"""
Unit tests for the 3 core battle strategy & deck tracking upgrades,
plus LiveScreenRecognizer card matching, GakumasRunner double-tap / loading handling,
and AdbController(AdbConfig(...)) positional initialization.
"""

from typing import Optional
import numpy as np
import pytest

from demo.agent.client import AgentConfig, AgentDecisionMaker
from demo.agent.fallback import HeuristicDecisionEngine
from demo.core.db import GakumasDB
from demo.core.prompt import PromptBuilder
from demo.core.tracker import GameStateTracker
from demo.maa_bridge.adb_controller import AdbConfig, AdbController
from demo.maa_bridge.base_controller import BaseController
from demo.maa_bridge.runner import GakumasRunner
from demo.schemas import BoundingBox, RecognizedCard, RecognizedDrink, ScreenObservation
from demo.vision.live_adb_recognizer import LiveScreenRecognizer


@pytest.fixture
def shared_db():
    db = GakumasDB("gakumas_library.db")
    yield db
    db.close()


class _DummyStubController(BaseController):
    """Non-MockController stub used to test live-mode GakumasRunner behavior offline."""

    def __init__(self):
        self._connected = True
        self.clicks = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def screencap(self) -> Optional[np.ndarray]:
        return np.zeros((1600, 720, 3), dtype=np.uint8)

    def click(self, x: int, y: int) -> bool:
        self.clicks.append((x, y))
        return True

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> bool:
        return True


def test_tracker_deck_persistence_multicopy_exhaustion_and_signature_top_deck(shared_db):
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    deck = tracker.init_produce_deck(db=shared_db)

    assert len(deck) >= 8
    assert 756 in deck  # 十王星南 [赤裸々] 私を超えて+
    assert tracker.upgrade_card_in_deck(1, 2) is True
    assert 2 in tracker.produce_deck
    assert tracker.remove_card_from_deck(2) is True
    tracker.add_card_to_deck(79)  # えいえいおー

    # Logic starter already has 2 copies of #13 (表現の基本, limit_count=1, base_id=13, genki+=4)
    assert tracker.produce_deck.count(13) == 2

    tracker.start_new_battle()
    # Playing 1st copy of #13 should NOT lock family 13 because 2 copies exist in deck
    c13 = shared_db.get_card_by_id(13)
    tracker.record_card_play(
        13,
        limit_count=c13["limit_count"],
        base_id=c13["base_id"],
        dsl_actions=c13["dsl_actions"],
        db=shared_db,
    )
    assert not tracker.is_card_exhausted(13, base_id=13)

    # Playing 2nd copy of #13 exhausts all copies in deck
    tracker.record_card_play(
        13,
        limit_count=c13["limit_count"],
        base_id=c13["base_id"],
        dsl_actions=c13["dsl_actions"],
        db=shared_db,
    )
    assert tracker.is_card_exhausted(13, base_id=13)

    # Playing signature #756 (私を超えて+) activates Genki->#757 top-deck generator
    c756 = shared_db.get_card_by_id(756)
    tracker.record_card_play(
        756,
        limit_count=c756["limit_count"],
        base_id=c756["base_id"],
        dsl_actions=c756["dsl_actions"],
        db=shared_db,
    )
    assert tracker.signature_genki_trigger_active is True

    # Play #3 (ポーズの基本, genki+=2) -> effectCounter=1 (odd) -> spawns #757 to pending_top_deck
    c3 = shared_db.get_card_by_id(3)
    tracker.record_card_play(
        3,
        limit_count=c3.get("limit_count"),
        base_id=c3.get("base_id"),
        dsl_actions=c3["dsl_actions"],
        db=shared_db,
    )
    assert 757 in tracker.pending_top_deck

    # Playing #757 itself (score+=genki*0.8, NO genki+=) must consume pending_top_deck, NOT re-trigger
    c757 = shared_db.get_card_by_id(757)
    before_count = len(tracker.battle_generated_cards)
    tracker.record_card_play(
        757,
        limit_count=c757["limit_count"],
        base_id=c757["base_id"],
        dsl_actions=c757["dsl_actions"],
        db=shared_db,
    )
    assert 757 not in tracker.pending_top_deck
    assert len(tracker.battle_generated_cards) == before_count

    # Next battle resets battle-local state while preserving produce_deck
    deck_snapshot = list(tracker.produce_deck)
    tracker.start_new_battle()
    assert tracker.produce_deck == deck_snapshot
    assert not tracker.used_limit_cards
    assert tracker.pending_top_deck == []


def test_tracker_sync_observed_hand_and_prompt_summary(shared_db):
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    tracker.init_produce_deck(db=shared_db)
    tracker.start_new_battle()

    # Simulate #757 pending top-deck and drawn in hand along with #756
    tracker.pending_top_deck.append(757)
    tracker.battle_generated_cards.append(757)
    deck_len_before = len(tracker.produce_deck)

    bbox = BoundingBox(xmin=50, ymin=1000, xmax=200, ymax=1250)
    hand = [
        RecognizedCard(card_id=756, name="私を超えて+", upgraded=True, bbox=bbox),
        RecognizedCard(card_id=757, name="私を超えて(翔)", bbox=bbox),
    ]
    tracker.sync_observed_hand(hand, db=shared_db)

    assert 757 not in tracker.pending_top_deck
    assert len(tracker.produce_deck) == deck_len_before  # token #757 does NOT pollute permanent produce_deck

    summary = tracker.get_deck_prompt_summary(db=shared_db, hand_cards=hand)
    assert "整局卡库总计" in summary
    assert "[固有⭐]" in summary
    assert "牌库中除手牌外的其余循环/待抽牌" in summary


def test_prompt_builder_strategy_rules_signature_priority_and_anti_skip(shared_db):
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    tracker.init_produce_deck(db=shared_db)
    tracker.start_new_battle()

    builder = PromptBuilder(db=shared_db, compact=False)
    bbox = BoundingBox(xmin=50, ymin=1000, xmax=200, ymax=1250)
    hand = [
        RecognizedCard(card_id=756, name="私を超えて+", upgraded=True, bbox=bbox),
        RecognizedCard(card_id=757, name="私を超えて(翔)", bbox=bbox),
    ]

    # Case 1: Lesson stage (week 5 追い込みレッスン) with stamina >= 10
    lesson_obs = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        scenario_id="hajime_regular",
        week=5,
        total_weeks=13,
        week_activity="中間追い込みレッスン (CLEARまで 50)",
        turn=1,
        turns_remaining=6,
        stamina=24,
        max_stamina=30,
        genki=18,
        score=0,
        target_score=50,
        buffs={"goodImpressionTurns": 2, "motivation": 3},
        cards=hand,
    )
    lesson_prompt, lesson_actions = builder.build_prompt(lesson_obs, tracker)
    assert "⭐固有卡+1使用次数(不耗行动点)" in lesson_prompt
    assert "不耗行动点" in lesson_prompt
    assert "剩余未用回合会自动兑换为体力回复" in lesson_prompt
    assert "非万不得已勿点SKIP浪费回合" in lesson_prompt
    assert "整局卡库总计" in lesson_prompt

    # Case 2: Exam stage (week 6 中間試験), late-turn burst warning, compact=True still stays concise
    exam_obs = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        scenario_id="hajime_regular",
        week=6,
        total_weeks=13,
        week_activity="中間試験",
        turn=7,
        turns_remaining=2,
        stamina=18,
        max_stamina=30,
        genki=35,
        score=300,
        target_score=800,
        buffs={"goodImpressionTurns": 6, "motivation": 5},
        cards=hand,
    )
    exam_prompt, _ = builder.build_prompt(exam_obs, tracker)
    assert "末盘抢分爆发期" in exam_prompt
    assert "本战为阶段考试，剩血不带入下周" in exam_prompt

    compact_prompt, compact_actions = builder.build_prompt(exam_obs, tracker, compact=True)
    assert (len(compact_prompt) // 2) < 800
    assert all(" | " not in a.description for a in compact_actions if a.action_type == "play_card")

    # Case 3: Produce schedule / card reward includes deck composition
    sched_obs = ScreenObservation(
        scene_type="produce_schedule",
        plan="logic",
        scenario_id="hajime_regular",
        week=4,
        total_weeks=13,
        stamina=20,
        max_stamina=30,
        options=[{"title": "相談 (Shop)", "description": "商店整备", "bbox": bbox}],
    )
    sched_prompt, _ = builder.build_prompt(sched_obs, tracker)
    assert "当前整局卡库构成" in sched_prompt
    assert "整局卡库总计" in sched_prompt


def test_heuristic_fallback_prioritizes_free_extra_use_signature(shared_db):
    builder = PromptBuilder(db=shared_db, compact=False)
    bbox = BoundingBox(xmin=50, ymin=1000, xmax=200, ymax=1250)
    obs = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        turn=1,
        turns_remaining=8,
        stamina=25,
        max_stamina=30,
        genki=10,
        cards=[
            RecognizedCard(card_id=3, name="ポーズの基本", bbox=bbox),
            RecognizedCard(card_id=756, name="私を超えて+", upgraded=True, bbox=bbox),
        ],
    )
    _, actions = builder.build_prompt(obs)
    dec = HeuristicDecisionEngine().decide(actions, obs)
    chosen = next(a for a in actions if a.choice_id == dec.choice_id)
    assert chosen.payload.get("card_id") == 756


def test_live_recognizer_parenthetical_card_match_preference(shared_db):
    reco = LiveScreenRecognizer(db=shared_db, plan="logic")
    m = reco.match_card_by_name("私を超えて", plan="logic", upgraded=True)
    assert m is not None
    assert m["id"] == 756  # Must match 私を超えて+, NOT #758 私を超えて(翔)+


def test_runner_double_tap_selected_skip_loading_and_reward_tracking(shared_db, monkeypatch):
    monkeypatch.setattr("demo.maa_bridge.runner.time.sleep", lambda *_a, **_kw: None)

    stub = _DummyStubController()
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    agent = AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True))
    runner = GakumasRunner(
        mode="adb",
        controller=stub,
        agent=agent,
        db=shared_db,
        tracker=tracker,
        plan="logic",
    )

    bbox_card = BoundingBox(xmin=100, ymin=1100, xmax=220, ymax=1350)
    bbox_reward = BoundingBox(xmin=200, ymin=1300, xmax=520, ymax=1400)

    scene_queue = [
        ScreenObservation(scene_type="loading", plan="logic"),
        ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            week=5,
            turn=1,
            turns_remaining=8,
            stamina=24,
            max_stamina=30,
            genki=5,
            cards=[RecognizedCard(card_id=756, name="私を超えて+", upgraded=True, selected=False, bbox=bbox_card)],
        ),
        ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            week=5,
            turn=1,
            turns_remaining=8,
            stamina=24,
            max_stamina=30,
            genki=5,
            cards=[RecognizedCard(card_id=3, name="ポーズの基本", selected=True, bbox=bbox_card)],
        ),
        ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            week=2,
            options=[{"choice_id": 0, "card_id": 79, "name": "受け取る", "title": "受け取る", "bbox": bbox_reward}],
        ),
    ]
    monkeypatch.setattr(runner.scene_routing_reco, "recognize", lambda _frame: scene_queue.pop(0))

    # Step 1 skips loading frame -> recognizes unselected #756 -> sends 2 clicks (double-tap)
    res1 = runner.run_once()
    assert res1.scene_type == "battle_turn"
    assert len(stub.clicks) == 2
    assert tracker.signature_genki_trigger_active is True

    # Step 2 recognizes already-selected card -> sends only 1 click
    res2 = runner.run_once()
    assert res2.scene_type == "battle_turn"
    assert len(stub.clicks) == 3  # +1 click only

    # Step 3 picks card reward "受け取る" for card #79 -> adds #79 to produce_deck
    res3 = runner.run_once()
    assert res3.scene_type == "card_reward"
    assert 79 in tracker.produce_deck


def test_adb_controller_positional_config_init():
    ctrl = AdbController(AdbConfig(host="127.0.0.1", port=5556))
    assert ctrl.host == "127.0.0.1"
    assert ctrl.port == 5556
    assert ctrl.address == "127.0.0.1:5556"


def test_system_and_user_prompts_aligned_without_hallucinated_passives():
    from demo.agent.prompts import get_system_prompt

    for plan in ("logic", "sense", "anomaly"):
        sp = get_system_prompt(plan=plan, scene_type="battle_turn", compact=False)
        assert "使用次数" in sp
        assert "CLEARまで" in sp
        assert "卡库" in sp
        assert "保留护盾与好印象，果断选择【结束回合】" not in sp

    logic_sp = get_system_prompt(plan="logic", scene_type="battle_turn", compact=False)
    assert "私を超えて" in logic_sp
    assert "初星笔记本" not in logic_sp
    assert "得意的笑容徽章" not in logic_sp

    shop_sp = get_system_prompt(plan="logic", scene_type="shop", compact=False)
    assert "研修案内" in shop_sp
    assert "整局卡库" in shop_sp


def test_fallback_skips_redraw_drink_at_high_stamina(shared_db):
    builder = PromptBuilder(db=shared_db, compact=False)
    bbox = BoundingBox(xmin=50, ymin=1000, xmax=200, ymax=1250)
    obs = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        turn=3,
        turns_remaining=5,
        stamina=22,
        max_stamina=31,
        genki=8,
        cards=[RecognizedCard(card_id=124, name="幸せな時間", bbox=bbox)],
        drinks=[RecognizedDrink(drink_id=9, name="ミックススムージー", bbox=bbox)],
    )
    _, actions = builder.build_prompt(obs)
    dec = HeuristicDecisionEngine().decide(actions, obs)
    chosen = next(a for a in actions if a.choice_id == dec.choice_id)
    assert chosen.action_type == "play_card"
    assert chosen.payload.get("card_id") == 124


def test_live_recognizer_week_digit_fix_and_story_exam_guard(shared_db, monkeypatch):
    reco = LiveScreenRecognizer(db=shared_db, plan="logic")
    reco._last_week = 9
    frame = np.full((1600, 720, 3), 240, dtype=np.uint8)

    # Story body talking about 最終試験 must NOT jump week to 13
    monkeypatch.setattr(reco, "_ocr_crop_upscaled", lambda *_a, **_kw: "")
    meta_story = reco._parse_week_and_stats(
        [{"text": "次の最終試験で必ず勝ちましょう", "bbox": [60, 1200, 620, 1240]}],
        frame,
    )
    assert meta_story["week"] == 9

    # Top-left "最終まで" + misread "ろ週" maps cleanly to "3週" -> Week 10
    def fake_crop(_frame, box, **_kw):
        x1, y1, _x2, _y2 = box
        if y1 < 40 and x1 < 40:
            return "最終まで"
        if 40 <= y1 <= 85 and x1 < 60:
            return "ろ週"
        return ""

    monkeypatch.setattr(reco, "_ocr_crop_upscaled", fake_crop)
    meta_w10 = reco._parse_week_and_stats([], frame)
    assert meta_w10["week"] == 10


def test_live_recognizer_exchange_modal_ignores_title_and_bright_close_popup(shared_db, monkeypatch):
    reco = LiveScreenRecognizer(db=shared_db, plan="logic")
    # Bright classroom top backdrop (>200 mean), modal header "交換確認" at y=900, real buttons at y=1460
    frame = np.full((1600, 720, 3), 210, dtype=np.uint8)
    fake_lines = [
        {"text": "交換確認", "bbox": [50, 895, 190, 925]},
        {"text": "Pポイントを使用して交換しますか?", "bbox": [60, 960, 560, 990]},
        {"text": "キャンセル", "bbox": [170, 1460, 310, 1495]},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "".join(l["text"] for l in fake_lines), "lines": fake_lines})
    monkeypatch.setattr(
        reco,
        "_ocr_bottom_orange_button",
        lambda *_a, **_kw: {
            "choice_id": 0,
            "name": "交換",
            "title": "交換",
            "description": "底部橙色主按钮: 交換",
            "bbox": {"xmin": 379, "ymin": 1445, "xmax": 623, "ymax": 1515},
        },
    )
    obs = reco.recognize(frame)
    assert obs.scene_type == "popup"
    assert obs.options[0]["name"] == "交換"
    assert obs.options[0]["bbox"]["ymin"] >= 1400
    assert all(o["name"] != "確認" for o in obs.options)

    # Bright tutorial modal with bottom-center 閉じる at y=1468 + diagram containing "残りターン数"
    tut_lines = [
        {"text": "パーフェクトを目指そう!", "bbox": [100, 450, 600, 500]},
        {"text": "残りターン数 2", "bbox": [150, 860, 300, 900]},
        {"text": "閉じる", "bbox": [342, 1468, 421, 1493]},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "".join(l["text"] for l in tut_lines), "lines": tut_lines})
    monkeypatch.setattr(reco, "_ocr_bottom_orange_button", lambda *_a, **_kw: None)
    tut_obs = reco.recognize(frame)
    assert tut_obs.scene_type == "popup"
    assert [o["name"] for o in tut_obs.options] == ["閉じる"]


def test_live_recognizer_cool_silver_drink_detection(shared_db):
    from pathlib import Path

    reco = LiveScreenRecognizer(db=shared_db, plan="logic")
    templates = reco._load_drink_templates()
    d1 = next((t for t in templates if t[0] == 1), None)  # #1 初星水
    assert d1 is not None
    row1 = shared_db.get_drink_by_id(1)
    assert row1 and row1.get("icon_path") and Path(row1["icon_path"]).exists()

    from PIL import Image

    frame = np.full((1600, 720, 3), 80, dtype=np.uint8)
    # Paint slot 0 circle at x=0.080w (~57), y=0.955h (~1528) with bright rim & 初星水 icon
    cx, cy = int(720 * 0.080), int(1600 * 0.955)
    yy, xx = np.ogrid[:1600, :720]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= 28 ** 2
    frame[circle] = (235, 245, 255)
    icon_rgba = Image.open(row1["icon_path"]).convert("RGBA").resize((46, 46), Image.Resampling.BILINEAR)
    bg_crop = Image.fromarray(frame[cy - 23 : cy + 23, cx - 23 : cx + 23]).convert("RGBA")
    comp = Image.alpha_composite(bg_crop, icon_rgba).convert("RGB")
    frame[cy - 23 : cy + 23, cx - 23 : cx + 23] = np.array(comp)

    drinks = reco._recognize_bottom_drinks(frame, plan="logic")
    assert any(d.drink_id == 1 for d in drinks)





def test_live_recognizer_clear_reward_card_not_event_choice(shared_db):
    import numpy as np
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    reco = LiveScreenRecognizer(db=shared_db)
    w, h = 720, 1600

    # 1. Dark translucent CLEAR reward card with OCR typos should never count as choice pills
    dark_frame = np.full((h, w, 3), 85, dtype=np.uint8)
    reward_lines = [
        {"text": "覆得pポイント", "bbox": [110, 935, 282, 989], "words": []},
        {"text": "得アイテム", "bbox": [154, 1025, 281, 1070], "words": []},
    ]
    assert reco._find_event_choice_pills(dark_frame, reward_lines, w, h) == []

    # 2. Even on a bright dress backdrop, reward summary tokens are excluded
    bright_frame = np.full((h, w, 3), 245, dtype=np.uint8)
    assert reco._find_event_choice_pills(bright_frame, reward_lines, w, h) == []

    # 3. Genuine white event choice pills remain recognized
    event_lines = [
        {"text": "ボーカルを重点的に", "bbox": [124, 1004, 408, 1034], "words": []},
        {"text": "ダンスを重点的に", "bbox": [126, 1105, 376, 1135], "words": []},
    ]
    pills = reco._find_event_choice_pills(bright_frame, event_lines, w, h)
    assert len(pills) == 2


def test_live_recognizer_turns_remaining_clamps_unrealistic_digits(shared_db, monkeypatch):
    import numpy as np
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    reco = LiveScreenRecognizer(db=shared_db)
    reco._last_turns_remaining = 4
    w, h = 720, 1600
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    meta = {"week": 12, "total_weeks": 13, "week_label": "12週", "stats": {"vocal": 500, "dance": 150, "visual": 130}, "stamina": 5, "max_stamina": 31, "p_points": 2}
    monkeypatch.setattr(reco, "_ocr_crop_upscaled", lambda *args, **kwargs: "60")
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *args, **kwargs: {"text": "65", "lines": []})
    obs_bad = reco._parse_battle_screen(frame, [{"text": "601", "bbox": [467, 5, 533, 40], "words": []}], "logic", meta)
    assert obs_bad.turns_remaining == 4

    monkeypatch.setattr(reco, "_ocr_crop_upscaled", lambda *args, **kwargs: "3")
    obs_ok = reco._parse_battle_screen(frame, [], "logic", meta)
    assert obs_ok.turns_remaining == 3


def test_live_recognizer_slot_drinks_hot_coffee_and_oolong_tea(shared_db):
    import numpy as np
    from PIL import Image
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    reco = LiveScreenRecognizer(db=shared_db)
    h, w = 1600, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cy = int(round(h * 0.9506))
    rad = int(round(w * 0.0458))
    for did, x_frac in ((6, 0.0833), (2, 0.2042)):
        d_row = shared_db.get_drink_by_id(did)
        assert d_row and d_row.get("icon_path")
        cx = int(round(w * x_frac))
        icon = Image.open(d_row["icon_path"]).convert("RGBA").resize((rad * 2, rad * 2), Image.Resampling.BILINEAR)
        arr = np.array(icon)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        comp = (arr[:, :, :3].astype(np.float32) * alpha + 245.0 * (1.0 - alpha)).astype(np.uint8)
        frame[cy - rad : cy + rad, cx - rad : cx + rad] = comp

    drinks = reco._recognize_bottom_drinks(frame, plan="logic")
    assert [d.drink_id for d in drinks] == [6, 2]


def test_live_recognizer_true_end_achievement_card_not_event_choice(shared_db, monkeypatch):
    import numpy as np
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    reco = LiveScreenRecognizer(db=shared_db)
    w, h = 720, 1600
    bright_frame = np.full((h, w, 3), 245, dtype=np.uint8)
    true_end_lines = [
        {"text": "TrueEnd条件", "bbox": [273, 725, 447, 751], "words": []},
        {"text": "・一5/10", "bbox": [164, 899, 395, 917], "words": []},
        {"text": "十王星南の親愛度Lvを9にする", "bbox": [162, 977, 438, 995], "words": []},
        {"text": "完全無欠", "bbox": [163, 1078, 256, 1100], "words": []},
        {"text": "TAPで次へ", "bbox": [275, 1450, 444, 1482], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda _arr: {"text": "\n".join(l["text"] for l in true_end_lines), "lines": true_end_lines})
    obs = reco.recognize(bright_frame, plan="logic")
    assert obs.scene_type != "produce_event"
    assert any("TAPで次へ" in (o.get("title") or "") for o in obs.options)

    # Achievement unlocked modal with bottom-center white circular [✕] close icon at (360, 1495)
    ach_frame = np.full((h, w, 3), 65, dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    circle_mask = ((xx - 360) ** 2 + (yy - 1495) ** 2) <= 38 ** 2
    ach_frame[circle_mask] = 248
    ach_frame[1482:1508, 355:365] = 85
    ach_frame[1490:1500, 347:373] = 85
    ach_lines = [
        {"text": "True End条件", "bbox": [200, 210, 520, 250], "words": []},
        {"text": "アチーブメント達成", "bbox": [120, 270, 600, 320], "words": []},
        {"text": "積み上げてきたもの", "bbox": [230, 900, 490, 930], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda _arr: {"text": "\n".join(l["text"] for l in ach_lines), "lines": ach_lines})
    obs_ach = reco.recognize(ach_frame, plan="logic")
    assert obs_ach.scene_type == "popup"
    assert obs_ach.options and obs_ach.options[0]["name"] == "閉じる"
    cy = (obs_ach.options[0]["bbox"]["ymin"] + obs_ach.options[0]["bbox"]["ymax"]) // 2
    assert 1490 <= cy <= 1500


def test_live_recognizer_resume_interrupt_popup_vs_retire(shared_db, monkeypatch):
    import numpy as np
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    reco = LiveScreenRecognizer(db=shared_db)
    w, h = 720, 1600
    frame = np.full((h, w, 3), 240, dtype=np.uint8)

    # 1. Interrupted-produce resume modal MUST expose 再開する cleanly (not blocked by "中断")
    resume_lines = [
        {"text": "中断中のプロデュースがあります", "bbox": [120, 620, 600, 660], "words": []},
        {"text": "再開しますか？", "bbox": [240, 680, 480, 710], "words": []},
        {"text": "再開する", "bbox": [410, 1180, 560, 1220], "words": []},
        {"text": "キャンセル", "bbox": [140, 1180, 290, 1220], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda _arr: {"text": "\n".join(l["text"] for l in resume_lines), "lines": resume_lines})
    obs_resume = reco.recognize(frame, plan="logic")
    assert obs_resume.scene_type == "popup"
    assert any("再開" in (o.get("name") or "") for o in obs_resume.options)

    # 2. Mid-run pause/retire modal ("一時中断") STILL withholds confirm and offers キャンセル
    pause_lines = [
        {"text": "一時中断確認", "bbox": [250, 620, 470, 660], "words": []},
        {"text": "リタイアして戻る", "bbox": [220, 680, 500, 710], "words": []},
        {"text": "決定", "bbox": [420, 1180, 520, 1220], "words": []},
        {"text": "キャンセル", "bbox": [140, 1180, 290, 1220], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda _arr: {"text": "\n".join(l["text"] for l in pause_lines), "lines": pause_lines})
    obs_pause = reco.recognize(frame, plan="logic")
    assert obs_pause.scene_type == "popup"
    assert [o.get("name") for o in obs_pause.options] == ["キャンセル"]


def test_live_recognizer_exam_heart_hp_and_fallback_drink_card_pacing(shared_db, monkeypatch):
    import numpy as np
    from PIL import ImageDraw, ImageFont
    from PIL import Image
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer
    from demo.core.prompt import PromptBuilder
    from demo.agent.fallback import HeuristicDecisionEngine
    from demo.schemas import BoundingBox, RecognizedCard, RecognizedDrink, ScreenObservation

    # 1. Green heart + dark-outlined '7' inside bright golden glare Exam HP ROI
    reco = LiveScreenRecognizer(db=shared_db)
    w, h = 720, 1600
    frame = np.full((h, w, 3), 220, dtype=np.uint8)
    frame[:, :, 2] = 135  # golden sunset glare background
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    hx = int(w * 0.76)
    hy = int(h * 0.145)
    draw.ellipse((hx, hy - 8, hx + 20, hy + 10), fill=(95, 225, 75))
    try:
        font = ImageFont.truetype("arialbd.ttf", 28)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except Exception:
            font = ImageFont.load_default(size=28)
    draw.text((hx + 26, hy - 15), "7", fill=(250, 250, 250), stroke_width=2, stroke_fill=(30, 30, 30), font=font)
    glare_arr = np.array(pil)
    hp_val = reco._ocr_heart_anchored_hp(glare_arr, (0.73, 0.130, 0.90, 0.165), 31)
    assert hp_val == 7

    # 2. Fallback skips #16 herbal tea when goodImpression == 0, and prefers Mental over Active early in W13
    builder = PromptBuilder(db=shared_db, compact=False)
    bbox = BoundingBox(xmin=50, ymin=1000, xmax=220, ymax=1350)
    obs_early = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=13,
        week_activity="最終試験",
        turn=2,
        turns_remaining=9,
        stamina=15,
        max_stamina=31,
        genki=6,
        buffs={"goodImpressionTurns": 0, "motivation": 2},
        cards=[
            RecognizedCard(card_id=2, name="アピールの基本+", upgraded=True, bbox=bbox),
            RecognizedCard(card_id=61, name="えいえいおー+", upgraded=True, bbox=bbox),
        ],
        drinks=[RecognizedDrink(drink_id=16, name="おしゃれハーブティー", bbox=bbox)],
    )
    _, acts_early = builder.build_prompt(obs_early)
    dec_early = HeuristicDecisionEngine().decide(acts_early, obs_early)
    chosen_early = next(a for a in acts_early if a.choice_id == dec_early.choice_id)
    assert chosen_early.action_type == "play_card"
    assert chosen_early.payload.get("card_id") == 61  # Mental card chosen, #16 skipped at 0 goodImpression

    # 3. In late exam turn (turns_remaining <= 4), flips to Active burst card
    obs_late = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=13,
        week_activity="最終試験",
        turn=10,
        turns_remaining=2,
        stamina=15,
        max_stamina=31,
        genki=40,
        buffs={"goodImpressionTurns": 6, "motivation": 12},
        cards=[
            RecognizedCard(card_id=61, name="えいえいおー+", upgraded=True, bbox=bbox),
            RecognizedCard(card_id=2, name="アピールの基本+", upgraded=True, bbox=bbox),
        ],
    )
    _, acts_late = builder.build_prompt(obs_late)
    dec_late = HeuristicDecisionEngine().decide(acts_late, obs_late)
    chosen_late = next(a for a in acts_late if a.choice_id == dec_late.choice_id)
    assert chosen_late.payload.get("card_id") == 2


def test_live_recognizer_reward_glow_split_and_genki_banner_digit(shared_db):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    reco = LiveScreenRecognizer(db=shared_db)
    w, h = 720, 1600
    frame = np.full((h, w, 3), 255, dtype=np.uint8)
    # Draw 3 card reward thumbnails where slot 1 has an orange glow touching slot 2 gap
    y1, y2 = int(h * 0.630), int(h * 0.710)
    for x1, x2 in ((145, 280), (294, 429), (443, 578)):
        frame[y1:y2, x1:x2] = (120, 155, 195)
    # Bright orange glow bridging gap between slot 1 and slot 2
    frame[y1 - 8 : y2 + 8, 138:296] = np.where(
        frame[y1 - 8 : y2 + 8, 138:296] == 255,
        np.array([245, 150, 50], dtype=np.uint8),
        frame[y1 - 8 : y2 + 8, 138:296],
    )
    slots = reco._find_reward_icon_slots(frame, w, h)
    assert len(slots) == 3
    assert slots[0][4] is True
    assert slots[1][4] is False and slots[2][4] is False

    # Synthetic blue shield with single white digit "4" -> _ocr_bw_banner_number must read 4 cleanly
    shield = np.full((60, 60, 3), (65, 185, 240), dtype=np.uint8)
    pil_s = Image.fromarray(shield)
    draw = ImageDraw.Draw(pil_s)
    try:
        font = ImageFont.truetype("arialbd.ttf", 34)
    except Exception:
        font = ImageFont.load_default(size=34)
    draw.text((20, 12), "4", fill=(252, 252, 252), font=font)
    arr_s = np.array(pil_s)
    mask = (arr_s[:, :, 0] > 200) & (arr_s[:, :, 1] > 200) & (arr_s[:, :, 2] > 200)
    assert reco._ocr_bw_banner_number(mask, prefix="GK:", max_val=999, scale=3) == 4


def test_live_recognizer_photo_saving_loading_and_shop_sold_out_filter(shared_db, monkeypatch):
    import numpy as np
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    reco = LiveScreenRecognizer(db=shared_db, plan="logic")
    w, h = 720, 1600

    # 1. Post-exam memory photo "保存中" spinner must resolve as loading (wait instead of clicking dimmed 次へ)
    save_frame = np.full((h, w, 3), 120, dtype=np.uint8)
    save_lines = [
        {"text": "メモリーにするフォトを選んでください", "bbox": [120, 420, 600, 460], "words": []},
        {"text": "保存中", "bbox": [310, 880, 410, 920], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "\n".join(l["text"] for l in save_lines), "lines": save_lines})
    obs_save = reco.recognize(save_frame, plan="logic")
    assert obs_save.scene_type == "loading"

    # 2. Shop screen excludes dark 交換済 slot while keeping bright affordable items & exit
    shop_frame = np.full((h, w, 3), 240, dtype=np.uint8)
    # Drink row slot 0 is sold-out dark overlay (<182 mean); slot 1 is bright unsold (>=200)
    d_y1, d_y2 = int(h * 0.512), int(h * 0.598)
    shop_frame[d_y1:d_y2, int(w * 0.08) : int(w * 0.27)] = 155
    shop_frame[d_y1:d_y2, int(w * 0.29) : int(w * 0.48)] = 215
    shop_lines = [
        {"text": "相談", "bbox": [60, 40, 140, 80], "words": []},
        {"text": "交換", "bbox": [260, 1180, 360, 1220], "words": []},
        {"text": "終了", "bbox": [600, 1210, 690, 1250], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "".join(l["text"] for l in shop_lines), "lines": shop_lines})
    monkeypatch.setattr(reco, "_ocr_crop_upscaled", lambda _f, box, **_kw: "50" if box[0] > int(w * 0.25) and box[1] > int(h * 0.55) else "")
    obs_shop = reco._parse_shop_screen(shop_frame, shop_lines, "logic", {"week": 11, "total_weeks": 13, "week_label": "最終まで 2週", "stats": {"vocal": 300, "dance": 150, "visual": 150}, "stamina": 29, "max_stamina": 31, "p_points": 60})
    names = [o["name"] for o in obs_shop.options]
    assert "Pドリンク1" not in names
    assert "Pドリンク2" in names
    assert "終了" in names

    # 3. Dimmed 交換する + orange-selected unaffordable slot (e.g. P=34 misread as 54 vs 35P SALE) must suppress >=35P items and leave 終了
    stall_frame = np.full((h, w, 3), 240, dtype=np.uint8)
    s_x1, s_x2 = int(w * 0.08), int(w * 0.27)
    s_y1, s_y2 = int(h * 0.512), int(h * 0.598)
    stall_frame[s_y1 - 6 : s_y1, s_x1 - 6 : s_x2 + 6] = (245, 145, 30)
    stall_frame[s_y2 : s_y2 + 6, s_x1 - 6 : s_x2 + 6] = (245, 145, 30)
    stall_frame[int(h * 0.78) : int(h * 0.86), 240:440] = (140, 120, 110)  # Dimmed 交換する (R < 218)
    stall_lines = [
        {"text": "相談", "bbox": [60, 40, 140, 80], "words": []},
        {"text": "交換する", "bbox": [260, 1300, 380, 1340], "words": []},
        {"text": "終了", "bbox": [600, 1310, 690, 1350], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "".join(l["text"] for l in stall_lines), "lines": stall_lines})
    monkeypatch.setattr(reco, "_ocr_crop_upscaled", lambda _f, box, **_kw: "35" if box[0] < int(w * 0.20) else "50")
    monkeypatch.setattr(reco, "_recognize_bottom_drinks", lambda *_a, **_kw: [])
    obs_dim = reco._parse_shop_screen(stall_frame, stall_lines, "logic", {"week": 4, "total_weeks": 13, "week_label": "中間まで 2週", "stats": {"vocal": 300, "dance": 130, "visual": 130}, "stamina": 2, "max_stamina": 31, "p_points": 54})
    assert [o["name"] for o in obs_dim.options] == ["終了"]


def test_tracker_four_zone_draw_wait_until_empty_exchange_drink_and_in_place_upgrade(shared_db):
    """Verify Discard only transfers to Draw Pile when Draw Pile empties, exchangeHand discards, and in-battle upgrade preserves produce_deck."""
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    # Minimal 5-card produce deck: #13 (limit=1), #1 (repeat), #21 (repeat), #60 (repeat), #120 (repeat)
    tracker.produce_deck = [13, 1, 21, 60, 120]
    tracker.start_new_battle(db=shared_db)
    bbox = BoundingBox(xmin=50, ymin=1050, xmax=200, ymax=1350)

    # Turn 1: observe [13, 1, 21] -> Draw Pile retains [60, 120] (2 cards)
    t1_cards = [
        RecognizedCard(card_id=13, name="表現の基本", bbox=bbox),
        RecognizedCard(card_id=1, name="アピールの基本", bbox=bbox),
        RecognizedCard(card_id=21, name="意識の基本", bbox=bbox),
    ]
    tracker.sync_observed_hand(t1_cards, db=shared_db, turn=1, turns_remaining=8)
    assert tracker.battle_hand == [13, 1, 21]
    assert tracker.battle_draw_pile == [60, 120]

    # Drink フレッシュビネガー upgrades #60/#13 style hand in-place ([13, 1, 21] -> [14, 2, 22]) without mutating produce_deck
    t1_upg_cards = [
        RecognizedCard(card_id=14, name="表現の基本+", upgraded=True, bbox=bbox),
        RecognizedCard(card_id=2, name="アピールの基本+", upgraded=True, bbox=bbox),
        RecognizedCard(card_id=22, name="意識の基本+", upgraded=True, bbox=bbox),
    ]
    tracker.sync_observed_hand(t1_upg_cards, db=shared_db, turn=1, turns_remaining=8)
    assert tracker.produce_deck == [13, 1, 21, 60, 120]
    assert tracker.battle_hand == [14, 2, 22]

    # Drink #9 ミックススムージー (exchangeHand): discards unplayed [14, 2, 22] -> Discard; draws remaining [60, 120] FIRST (0 reshuffles since Draw had 2 cards!)
    d9 = shared_db.get_drink_by_id(9)
    tracker.record_drink_use(9, dsl_actions=d9["dsl_actions"], db=shared_db)
    assert tracker.battle_hand == []
    assert tracker.battle_discard_pile == [14, 2, 22]
    assert tracker.battle_reshuffle_count == 0

    t1_redraw = [
        RecognizedCard(card_id=60, name="えいえいおー", bbox=bbox),
        RecognizedCard(card_id=120, name="あふれる思い出", bbox=bbox),
    ]
    tracker.sync_observed_hand(t1_redraw, db=shared_db, turn=1, turns_remaining=8)
    assert tracker.battle_reshuffle_count == 0  # Draw Pile had [60, 120], so Discard was NOT touched!
    assert tracker.battle_draw_pile == []
    assert tracker.battle_discard_pile == [14, 2, 22]

    # Explicit End Turn discards unplayed [60, 120] -> Discard now [14, 2, 22, 60, 120]
    tracker.record_turn_end(db=shared_db)
    assert tracker.battle_hand == []
    assert tracker.battle_discard_pile == [14, 2, 22, 60, 120]

    # Turn 2: Draw Pile is empty, so drawing [14, 2] triggers reshuffle #1 Discard -> Draw
    tracker.sync_observed_hand(
        [
            RecognizedCard(card_id=14, name="表現の基本+", upgraded=True, bbox=bbox),
            RecognizedCard(card_id=2, name="アピールの基本+", upgraded=True, bbox=bbox),
        ],
        db=shared_db,
        turn=2,
        turns_remaining=7,
    )
    assert tracker.battle_reshuffle_count == 1
    assert tracker.battle_discard_pile == []
    assert tracker.battle_draw_pile == [22, 60, 120]

    # Play #14 (limit=1): enters Exhaust Pile -> never playable or reshuffled again
    tracker.record_card_play(14, limit_count=1, base_id=13, dsl_actions="genki+=5", db=shared_db)
    assert tracker.battle_exhaust_pile == [14]
    assert tracker.is_card_exhausted(14, base_id=13) is True

    # Detailed non-session battle prompt also documents 4-zone rules + real pile state
    prompt, _ = PromptBuilder(db=shared_db, compact=False).build_prompt(
        ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            week=5,
            turn=2,
            turns_remaining=7,
            stamina=24,
            max_stamina=31,
            cards=[RecognizedCard(card_id=2, name="アピールの基本+", upgraded=True, bbox=bbox)],
        ),
        tracker,
    )
    assert "抽牌四区规则与完整卡库/本场牌堆实时追踪" in prompt
    assert "2) 🎴 【牌堆/待抽】(3张)" in prompt
    assert "4) 🚫 【除外】(1张，已打出限1牌，本场不可再使用): 《表現の基本+》×1" in prompt

    # Starting next battle resets all 4 zones cleanly back to produce_deck
    tracker.start_new_battle(db=shared_db)
    assert tracker.battle_draw_pile == [13, 1, 21, 60, 120]
    assert tracker.battle_hand == []
    assert tracker.battle_discard_pile == []
    assert tracker.battle_exhaust_pile == []
    assert tracker.battle_reshuffle_count == 0


def test_runner_center_play_double_tap_center_card_ocr_turn_guard_and_zero_hp_a_tag(shared_db, monkeypatch):
    """Verify live play_card taps bottom slot click_coords twice (once if already selected), ignores tooltip headers, guards turns_remaining, and tags 0-HP A-cards + <=3 turn clear directive."""
    monkeypatch.setattr("demo.maa_bridge.runner.time.sleep", lambda *_a, **_kw: None)

    stub = _DummyStubController()
    agent = AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True))
    runner = GakumasRunner(mode="adb", controller=stub, agent=agent, db=shared_db, plan="logic")

    bottom_bbox = BoundingBox(xmin=70, ymin=1180, xmax=220, ymax=1460)
    obs1 = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=5,
        turn=7,
        turns_remaining=2,
        stamina=18,
        max_stamina=31,
        genki=12,
        target_score=20,
        cards=[RecognizedCard(card_id=2, name="アピールの基本+", upgraded=True, selected=False, bbox=bottom_bbox)],
    )
    obs2 = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=5,
        turn=7,
        turns_remaining=2,
        stamina=18,
        max_stamina=31,
        genki=12,
        target_score=20,
        cards=[RecognizedCard(card_id=2, name="アピールの基本+", upgraded=True, selected=True, bbox=bottom_bbox)],
    )
    q = [obs1, obs2]
    monkeypatch.setattr(runner.scene_routing_reco, "recognize", lambda _f: q.pop(0))

    runner.run_once()
    assert stub.clicks[:2] == [bottom_bbox.center(), bottom_bbox.center()]

    runner.run_once()
    assert len(stub.clicks) == 3
    assert stub.clicks[2] == bottom_bbox.center()

    # Tooltip-header regression test + monotonic turns_remaining guard in LiveScreenRecognizer
    reco = LiveScreenRecognizer(db=shared_db, plan="logic")
    w, h = 720, 1600
    tip_frame = np.zeros((h, w, 3), dtype=np.uint8)
    tip_frame[int(h * 0.875) : int(h * 0.905), 260:460] = (230, 230, 230)
    sample_lines = [
        {"text": "えいえいおー", "bbox": [275, 921, 445, 946], "words": []},
        {"text": "アピールの基本+", "bbox": [280, 1410, 440, 1440], "words": []},
    ]
    meta = {"week": 5, "total_weeks": 13, "week_label": "5週", "stats": {"vocal": 250, "dance": 130, "visual": 130}, "stamina": 15, "max_stamina": 31, "p_points": 20}
    monkeypatch.setattr(reco, "_detect_hand_card_slots", lambda _f: [])
    tip_obs = reco._parse_battle_screen(tip_frame, sample_lines, "logic", meta)
    assert [c.card_id for c in tip_obs.cards] == [2]  # Tooltip header at y=921 must NOT create a ghost card!

    # Monotonic battle turn guard drops upward jump (4 -> 6) and huge drop (4 -> 1)
    reco._in_battle_scene = True
    reco._last_turns_remaining = 4
    monkeypatch.setattr(reco, "_ocr_crop_upscaled", lambda *_a, **_kw: "6")
    monkeypatch.setattr(reco, "_ocr_bw_banner_number", lambda *_a, **_kw: 6)
    assert reco._parse_battle_screen(np.zeros((h, w, 3), dtype=np.uint8), [], "logic", meta).turns_remaining == 4

    monkeypatch.setattr(reco, "_ocr_crop_upscaled", lambda *_a, **_kw: "1")
    monkeypatch.setattr(reco, "_ocr_bw_banner_number", lambda *_a, **_kw: 1)
    assert reco._parse_battle_screen(np.zeros((h, w, 3), dtype=np.uint8), [], "logic", meta).turns_remaining == 4

    monkeypatch.setattr(reco, "_ocr_crop_upscaled", lambda *_a, **_kw: "3")
    monkeypatch.setattr(reco, "_ocr_bw_banner_number", lambda *_a, **_kw: 3)
    assert reco._parse_battle_screen(np.zeros((h, w, 3), dtype=np.uint8), [], "logic", meta).turns_remaining == 3

    # Prompt directive + zero-HP free A tag on late regular lesson turn
    p_txt, _, _ = PromptBuilder(db=shared_db).build_session_prompt(obs1, GameStateTracker(plan="logic"), commit=False)
    assert "课程末盘清分死令" in p_txt
    assert "🔥0扣血免费A卡" in p_txt

    # PERFECTまで ring header normalization & blue shield anchored Genki fallback
    monkeypatch.setattr(reco, "_ocr_crop_upscaled", lambda _f, box, **_kw: "PERFE@T*で" if box[0] > int(w * 0.30) else "")
    perf_obs = reco._parse_battle_screen(np.zeros((h, w, 3), dtype=np.uint8), [], "logic", meta)
    assert "PERFECTまで" in perf_obs.week_activity

    gk_frame = np.zeros((h, w, 3), dtype=np.uint8)
    gk_frame[int(h * 0.080) : int(h * 0.095), int(w * 0.875) : int(w * 0.910)] = (40, 120, 220)
    gk_frame[int(h * 0.082) : int(h * 0.092), int(w * 0.920) : int(w * 0.960)] = (245, 245, 245)
    monkeypatch.setattr(reco, "_ocr_bw_banner_number", lambda *_a, **_kw: 83 if _kw.get("prefix") == "GK:" else None)
    assert reco._ocr_shield_anchored_genki(gk_frame, (0.855, 0.066, 0.992, 0.110)) == 83


def test_shop_insufficient_p_points_visited_filter_and_fallback_exit(shared_db, monkeypatch):
    """Shop collapses to Exit when P-points cannot buy any unvisited item and suppresses revisited slots."""
    reco = LiveScreenRecognizer(db=shared_db, plan="logic")
    w, h = 720, 1600
    shop_frame = np.full((h, w, 3), 240, dtype=np.uint8)
    # Mark drink slot 0 as currently orange-selected (visited) while slot 1 is affordable if P >= 50
    s_x1, s_x2 = int(w * 0.08), int(w * 0.27)
    s_y1, s_y2 = int(h * 0.512), int(h * 0.598)
    shop_frame[s_y1 - 6 : s_y1, s_x1 - 6 : s_x2 + 6] = (245, 145, 30)
    shop_frame[s_y2 : s_y2 + 6, s_x1 - 6 : s_x2 + 6] = (245, 145, 30)
    shop_lines = [
        {"text": "相談", "bbox": [60, 40, 140, 80], "words": []},
        {"text": "交換する", "bbox": [260, 1300, 380, 1340], "words": []},
        {"text": "終了", "bbox": [600, 1310, 690, 1350], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "".join(l["text"] for l in shop_lines), "lines": shop_lines})
    monkeypatch.setattr(reco, "_ocr_crop_upscaled", lambda *_a, **_kw: "80")
    monkeypatch.setattr(reco, "_recognize_bottom_drinks", lambda *_a, **_kw: [])

    # 1. With 30P (< 80P all items), shop must expose ONLY 終了 immediately
    obs_low_p = reco._parse_shop_screen(
        shop_frame,
        shop_lines,
        "logic",
        {"week": 4, "total_weeks": 13, "week_label": "中間まで 2週", "stats": {"vocal": 300, "dance": 130, "visual": 130}, "stamina": 20, "max_stamina": 31, "p_points": 30},
    )
    assert obs_low_p.p_points == 30
    assert [o["name"] for o in obs_low_p.options] == ["終了"]

    # 2. Visited slot ("Pドリンク:0") stays excluded across consecutive shop steps until leaving shop
    obs_more_p = reco._parse_shop_screen(
        shop_frame,
        shop_lines,
        "logic",
        {"week": 4, "total_weeks": 13, "week_label": "中間まで 2週", "stats": {"vocal": 300, "dance": 130, "visual": 130}, "stamina": 20, "max_stamina": 31, "p_points": 120},
    )
    names_more = [o["name"] for o in obs_more_p.options]
    assert "Pドリンク1" not in names_more
    assert "Pドリンク2" in names_more

    # 3. Fallback engine immediately chooses Exit when p_points < min item price
    shop_obs = ScreenObservation(
        scene_type="shop",
        p_points=25,
        options=[
            {"choice_id": 0, "name": "スキルカード1", "title": "切换预览 卡牌 #1 (50P)", "price": 50},
            {"choice_id": 1, "name": "終了", "title": "終了 (离开相談商店)"},
        ],
    )
    _, shop_acts = PromptBuilder(db=shared_db, compact=False).build_prompt(shop_obs)
    dec = HeuristicDecisionEngine().decide(shop_acts, shop_obs)
    assert dec.choice_id == 1


def test_runner_idle_watchdog_taps_blank_after_30s(shared_db, monkeypatch):
    """Watchdog fires blank-area tap at (670, 416) after >30s stuck and repeats at +60s, not earlier."""
    from demo.maa_bridge.runner import RunnerState

    stub = _DummyStubController()
    agent = AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True))
    runner = GakumasRunner(mode="adb", controller=stub, agent=agent, db=shared_db, plan="logic", idle_blank_tap_seconds=30.0)
    monkeypatch.setattr("demo.maa_bridge.runner.threading.Thread", lambda *a, **kw: type("NoOpThread", (), {"start": lambda self: None, "is_alive": lambda self: False, "join": lambda self, timeout=None: None})())

    runner.start()
    t0 = 1000.0
    runner._last_progress_ts = t0
    runner._last_heartbeat_ts = t0
    runner._last_blank_tap_ts = t0

    assert runner.check_idle_watchdog(now=t0 + 29.0) is False
    assert stub.clicks == []

    assert runner.check_idle_watchdog(now=t0 + 31.0) is True
    assert stub.clicks == [(670, 416)]
    assert runner.watchdog_tap_count == 1

    # No duplicate tap inside next 30s window, repeats once >= 60s
    assert runner.check_idle_watchdog(now=t0 + 50.0) is False
    assert runner.check_idle_watchdog(now=t0 + 62.0) is True
    assert stub.clicks == [(670, 416), (670, 416)]
    assert runner.watchdog_tap_count == 2
    runner.state = RunnerState.STOPPED


def test_shield_green_red_hp_tracker_ledger_and_limit_self_heal(shared_db):
    """Verify shield/green/red cost math, backend shield+motivation ledger, dropout recovery, and #757 self-heal."""
    from demo.core.rules import inspect_card_cost_and_conditions

    # 1. Green cost (cost-=6) absorbs into shield first; halfCostTurns halves ceil(6/2)=3
    c_green = shared_db.get_card_by_id(755)  # 私を超えて: cost-=6
    res_g = inspect_card_cost_and_conditions(
        c_green, stamina=10, genki=2, buffs={"halfCostTurns": 1}, card_uses_remaining=1, used_limit_cards=set()
    )
    assert res_g["cost_type"] == "green"
    assert res_g["effective_cost"] == 3
    assert res_g["shield_cost"] == 2
    assert res_g["hp_loss"] == 1
    assert res_g["playable"] is True

    # 2. Red cost (stamina-=1, #757 私を超えて(翔)) bypasses shield & requires genki >= 15
    c_red = shared_db.get_card_by_id(757)
    res_r_low = inspect_card_cost_and_conditions(
        c_red, stamina=11, genki=5, buffs={"halfCostTurns": 1}, card_uses_remaining=1, used_limit_cards=set()
    )
    assert res_r_low["cost_type"] == "red"
    assert res_r_low["shield_cost"] == 0
    assert res_r_low["hp_loss"] == 1
    assert res_r_low["playable"] is False
    assert "护盾(元気)≥15" in res_r_low["condition_desc"]

    res_r_ok = inspect_card_cost_and_conditions(
        c_red, stamina=11, genki=15, buffs={"halfCostTurns": 1}, card_uses_remaining=1, used_limit_cards=set()
    )
    assert res_r_ok["playable"] is True
    assert res_r_ok["hp_loss"] == 1
    assert "✅已满足" in res_r_ok["condition_desc"]

    # 3. Tracker backend Shield + Motivation ledger & dropout recovery
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    tracker.init_produce_deck(db=shared_db)
    tracker.start_new_battle(db=shared_db)
    tracker.stamina = 20
    tracker.max_stamina = 31
    tracker.buffs["motivation"] = 7
    # Play #155 テレビ出演+ (cost-=1 -> green 1->shield, then genki+=5 + 7 motivation = 12!)
    tracker.record_card_play(155, db=shared_db)
    assert tracker.genki == 12
    assert tracker.buffs["halfCostTurns"] == 5

    # If next frame OCR drops genki to 0 and drops motivation, reconcile_live_battle_state restores both
    obs_drop = ScreenObservation(scene_type="battle_turn", plan="logic", turn=2, turns_remaining=8, stamina=19, max_stamina=31, genki=0, buffs={})
    tracker.reconcile_live_battle_state(obs_drop)
    assert obs_drop.genki == 12
    assert obs_drop.buffs["motivation"] == 7
    assert "🛡️当前护盾(元気): 12" in tracker.get_shield_prompt_summary()

    # 4. Self-heal limit_count=1 card (#757) if accidentally recorded played while still observed in hand
    tracker.battle_generated_cards.append(757)
    tracker.battle_draw_pile.insert(0, 757)
    tracker.record_card_play(757, db=shared_db)
    assert tracker.is_card_exhausted(757, base_id=757) is True
    bbox = BoundingBox(xmin=44, ymin=1184, xmax=230, ymax=1456)
    tracker.sync_observed_hand([RecognizedCard(card_id=757, name="私を超えて(翔)", bbox=bbox)], db=shared_db, turn=2, turns_remaining=8)
    assert tracker.is_card_exhausted(757, base_id=757) is False
    assert 757 in tracker.battle_hand


def test_prompt_explains_red_green_costs_and_card_trigger_conditions(shared_db):
    """Verify prompt renders clear Chinese trigger conditions, #758 synthesized text, and red vs green cost breakdown."""
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    tracker.init_produce_deck(db=shared_db)
    tracker.start_new_battle(db=shared_db)
    bbox = BoundingBox(xmin=50, ymin=1100, xmax=220, ymax=1420)
    obs = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=6,
        week_activity="中間試験",
        turn=3,
        turns_remaining=7,
        stamina=11,
        max_stamina=31,
        genki=18,
        buffs={"motivation": 7, "halfCostTurns": 1},
        cards=[
            RecognizedCard(card_id=758, name="私を超えて(翔)+", upgraded=True, bbox=bbox),
            RecognizedCard(card_id=155, name="テレビ出演+", upgraded=True, bbox=bbox),
        ],
    )
    prompt, actions, _ = PromptBuilder(db=shared_db).build_session_prompt(obs, tracker, commit=False)
    assert "🟢绿血" in prompt
    assert "🔴红血" in prompt
    assert "需当前护盾(元気)≥15[✅已满足]" in prompt
    assert "触发机制: 本场后续每打出2张带元气(盾)效果的卡" in prompt
    assert any("🔴红血直扣1·无视护盾" in a.description for a in actions)
    assert any("🟢绿血" in a.description for a in actions)


def test_live_recognizer_exam_result_prefers_orange_next_over_disabled_retry(shared_db, monkeypatch):
    """Exam result screen with grey 〇再挑戦 + 再挑戦チケットがありません must expose only orange 次へ."""
    reco = LiveScreenRecognizer(db=shared_db, plan="logic")
    w, h = 720, 1600
    frame = np.full((h, w, 3), 120, dtype=np.uint8)
    frame[1435:1495, 50:215] = (190, 190, 190)  # Grey disabled retry pill
    frame[1418:1488, 237:482] = (245, 130, 35)  # Bright orange 次へ button
    res_lines = [
        {"text": "十王星南", "bbox": [209, 584, 320, 613], "words": []},
        {"text": "再挑戦チケットがありません", "bbox": [204, 782, 515, 819], "words": []},
        {"text": "〇再挑戦", "bbox": [67, 1451, 196, 1480], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "\n".join(l["text"] for l in res_lines), "lines": res_lines})
    monkeypatch.setattr(
        reco,
        "_ocr_bottom_orange_button",
        lambda _f, _w, _h: {
            "choice_id": 0,
            "name": "次へ",
            "title": "次へ",
            "description": "底部橙色主按钮: 次へ",
            "bbox": {"xmin": 237, "ymin": 1418, "xmax": 482, "ymax": 1488},
        },
    )
    obs = reco.recognize(frame, plan="logic")
    assert [o["name"] for o in obs.options] == ["次へ"]


def test_drink_cap_reward_exposes_discard_and_dont_take_without_selected_loop(shared_db, monkeypatch):
    """3/3 P-drink cap screen exposes held-drink discard entries, 受け取らない, and unvisited slots, never looping selected slot 1."""
    from demo.schemas import RecognizedDrink

    reco = LiveScreenRecognizer(db=shared_db, plan="logic")
    w, h = 720, 1600
    cap_frame = np.full((h, w, 3), 245, dtype=np.uint8)
    cap_frame[1220:1280, 240:480] = (175, 115, 85)  # dimmed 受け取る
    cap_lines = [
        {"text": "ホットコーヒー", "bbox": [259, 662, 460, 688], "words": []},
        {"text": "受け取らない", "bbox": [289, 1155, 430, 1178], "words": []},
        {"text": "受け取る", "bbox": [305, 1242, 413, 1268], "words": []},
        {"text": "Pドリンク所持上限", "bbox": [275, 1282, 446, 1301], "words": []},
    ]
    held = [
        RecognizedDrink(drink_id=2, name="烏龍茶", bbox=BoundingBox(xmin=27, ymin=1488, xmax=93, ymax=1554), dsl_summary="元気+7"),
        RecognizedDrink(drink_id=1, name="初星水", bbox=BoundingBox(xmin=114, ymin=1488, xmax=180, ymax=1554), dsl_summary="パラメータ+10"),
        RecognizedDrink(drink_id=6, name="ホットコーヒー", bbox=BoundingBox(xmin=199, ymin=1488, xmax=265, ymax=1554), dsl_summary="やる気+3"),
    ]
    monkeypatch.setattr(reco, "_recognize_bottom_drinks", lambda *_a, **_kw: list(held))
    monkeypatch.setattr(
        reco,
        "_find_reward_icon_slots",
        lambda *_a, **_kw: [(169, 1000, 273, 1148, True), (308, 1000, 412, 1148, False), (447, 1000, 551, 1148, False)],
    )
    slot_map = {169: {"id": 6, "name": "ホットコーヒー", "description": "やる気+3"}, 308: {"id": 2, "name": "烏龍茶", "description": "元気+7"}, 447: {"id": 5, "name": "ルイボスティー", "description": "好印象+3"}}
    monkeypatch.setattr(reco, "_match_square_drink_slot", lambda _f, x1, _y1, _x2, _y2: slot_map.get(x1))
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda img_in, *_a, **_kw: {"text": "ホットコーヒー" if img_in.shape[0] < 500 else "\n".join(l["text"] for l in cap_lines), "lines": [{"text": "ホットコーヒー", "bbox": [10, 10, 180, 35], "words": []}] if img_in.shape[0] < 500 else cap_lines})

    obs = reco.recognize(cap_frame, plan="logic")
    assert obs.scene_type == "card_reward"
    assert [d.drink_id for d in obs.drinks] == [2, 1, 6]
    names = [o["name"] for o in obs.options]
    assert "破棄候補_2" in names
    assert "破棄候補_1" in names
    assert "破棄候補_6" in names
    assert "受け取らない" in names
    assert "预览候选饮料2" in names
    assert "预览候选饮料3" in names
    assert "候选卡1" not in names and "预览候选饮料1" not in names


def test_drink_cap_checked_dont_take_exposes_confirm_action(shared_db, monkeypatch):
    """Once 受け取らない checkbox is checked orange, card_reward exposes [0] 決定 to confirm skipping."""
    reco = LiveScreenRecognizer(db=shared_db, plan="logic")
    w, h = 720, 1600
    frame = np.full((h, w, 3), 245, dtype=np.uint8)
    frame[1150:1182, 238:270] = (245, 130, 35)  # checked orange checkbox left of 受け取らない
    frame[1222:1282, 240:480] = (245, 135, 40)  # active orange 決定 button
    lines = [
        {"text": "受け取らない", "bbox": [289, 1155, 430, 1178], "words": []},
        {"text": "決定", "bbox": [315, 1242, 405, 1268], "words": []},
        {"text": "Pドリンク所持上限", "bbox": [275, 1282, 446, 1301], "words": []},
    ]
    monkeypatch.setattr(reco, "_recognize_bottom_drinks", lambda *_a, **_kw: [])
    monkeypatch.setattr(reco, "_find_reward_icon_slots", lambda *_a, **_kw: [])
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "\n".join(l["text"] for l in lines), "lines": lines})

    obs = reco.recognize(frame, plan="logic")
    assert obs.scene_type == "card_reward"
    assert [o["name"] for o in obs.options] == ["決定", "取消受け取らない"]


def test_drink_discard_popup_exposes_haki_and_protects_resume_retire(shared_db, monkeypatch):
    """P-drink detail/discard tooltip exposes 捨てる/破棄 outside battle, suppresses disabled 捨てる in battle, and protects abandon-save modals."""
    import openai
    from unittest.mock import MagicMock
    from demo.agent.client import AgentConfig, AgentDecisionMaker

    reco = LiveScreenRecognizer(db=shared_db, plan="logic")
    w, h = 720, 1600
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    # 1. Pドリンク詳細 overlay even with background 受け取る still OCR-visible
    discard_lines = [
        {"text": "Pドリンク詳細", "bbox": [250, 520, 470, 555], "words": []},
        {"text": "初星水", "bbox": [290, 590, 430, 620], "words": []},
        {"text": "破棄", "bbox": [220, 960, 310, 995], "words": []},
        {"text": "閉じる", "bbox": [430, 960, 520, 995], "words": []},
        {"text": "受け取る", "bbox": [305, 1242, 413, 1268], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "\n".join(l["text"] for l in discard_lines), "lines": discard_lines})
    monkeypatch.setattr(reco, "_ocr_bottom_orange_button", lambda *_a, **_kw: None)
    obs = reco.recognize(frame, plan="logic")
    assert obs.scene_type == "popup"
    assert [o["name"] for o in obs.options] == ["破棄", "閉じる"]

    # 2. Outside-battle real Gakumas layout: 捨てる at [577, 1248, 635, 1267] + Xキャンセル at [116, 1466, 309, 1495]
    suteru_lines = [
        {"text": "Pドリンク詳細", "bbox": [61, 1168, 270, 1198], "words": []},
        {"text": "烏龍茶", "bbox": [192, 1250, 277, 1277], "words": []},
        {"text": "捨てる", "bbox": [577, 1248, 635, 1267], "words": []},
        {"text": "Xキャンセル", "bbox": [116, 1466, 309, 1495], "words": []},
        {"text": "受け取る", "bbox": [305, 1242, 413, 1268], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "\n".join(l["text"] for l in suteru_lines), "lines": suteru_lines})
    obs_suteru = reco.recognize(frame, plan="logic")
    assert obs_suteru.scene_type == "popup"
    assert [o["name"] for o in obs_suteru.options] == ["捨てる", "キャンセル"]

    # 3. Mid-battle Pドリンク詳細 (_in_battle_scene=True + orange 使う) suppresses disabled 捨てる
    reco._in_battle_scene = True
    monkeypatch.setattr(
        reco,
        "_ocr_bottom_orange_button",
        lambda *_a, **_kw: {
            "choice_id": 0,
            "name": "使う",
            "title": "使う",
            "description": "底部橙色主按钮: 使う",
            "bbox": {"xmin": 370, "ymin": 1438, "xmax": 615, "ymax": 1508},
        },
    )
    obs_battle = reco.recognize(frame, plan="logic")
    assert obs_battle.scene_type == "popup"
    assert [o["name"] for o in obs_battle.options] == ["使う", "キャンセル"]
    reco._in_battle_scene = False
    monkeypatch.setattr(reco, "_ocr_bottom_orange_button", lambda *_a, **_kw: None)

    # 4. Title/resume abandon-save modal with 中断データ + 破棄 must NEVER expose 破棄 or 捨てる
    abandon_lines = [
        {"text": "再開確認", "bbox": [260, 500, 460, 540], "words": []},
        {"text": "中断データを破棄して新しく始めますか？", "bbox": [110, 720, 610, 760], "words": []},
        {"text": "破棄", "bbox": [410, 1180, 520, 1220], "words": []},
        {"text": "キャンセル", "bbox": [180, 1180, 310, 1220], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "\n".join(l["text"] for l in abandon_lines), "lines": abandon_lines})
    abandon_obs = reco.recognize(frame, plan="logic")
    assert abandon_obs.scene_type == "popup"
    assert all(o.get("name") not in ("破棄", "破棄する", "捨てる") for o in abandon_obs.options)

    # 5. xAI-style BadRequestError ('Error code: 400' with body '[input_too_large] The prompt is too long...') trims cleanly
    maker = AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True))
    maker.session_messages = [{"role": "system", "content": "SYS"}]
    maker._commit_session_turn("BOOTSTRAP【新卡牌档案收录】", 0, "init")
    for i in range(40):
        maker._commit_session_turn(f"U{i}" + ("【新卡牌档案收录】" if i == 5 else ""), 0, "ok")
    assert len(maker.session_messages) < 75
    assert any("U5" in m["content"] for m in maker.session_messages)
    xai_err = openai.BadRequestError(
        message="Error code: 400",
        response=MagicMock(status_code=400, text='{"code":"invalid-argument","error":"[input_too_large] The prompt is too long"}'),
        body={"code": "invalid-argument", "error": "[input_too_large] The prompt is too long"},
    )
    trimmed = maker._trim_session_on_context_overflow(
        list(maker.session_messages) + [{"role": "user", "content": "LATEST"}],
        xai_err,
    )
    assert trimmed is not None
    assert trimmed[1]["content"].startswith("BOOTSTRAP")
    assert trimmed[-1]["content"] == "LATEST"


def test_landscape_live_concert_skip_and_memory_regen_ignore(shared_db, monkeypatch):
    """Landscape Live Concert exposes LIVE_SKIP or tap-to-reveal HUD, and post-Produce Memory ignores 再生成."""
    reco = LiveScreenRecognizer(db=shared_db, plan="logic")

    # 1. Landscape 1600x720 with white |> icon in top-right (1510..1529, 64..95)
    lw, lh = 1600, 720
    live_hud = np.full((lh, lw, 3), 50, dtype=np.uint8)
    live_hud[64:95, 1510:1529] = (245, 245, 245)
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "", "lines": []})
    obs_skip = reco.recognize(live_hud, plan="logic")
    assert obs_skip.scene_type == "skip"
    assert [o["name"] for o in obs_skip.options] == ["LIVE_SKIP"]

    # 2. Landscape 1600x720 with hidden HUD returns tap-to-reveal control bar
    live_clean = np.full((lh, lw, 3), 50, dtype=np.uint8)
    obs_reveal = reco.recognize(live_clean, plan="logic")
    assert obs_reveal.scene_type == "skip"
    assert [o["name"] for o in obs_reveal.options] == ["唤出Live控制栏"]

    # 3. Post-Produce Memory screen ignores 再生成 and returns orange 完了
    pw, ph = 720, 1600
    mem_frame = np.full((ph, pw, 3), 230, dtype=np.uint8)
    mem_frame[1420:1490, 380:640] = (245, 130, 35)
    mem_lines = [
        {"text": "メモリー生成", "bbox": [240, 120, 480, 165], "words": []},
        {"text": "再生成", "bbox": [120, 1440, 240, 1475], "words": []},
        {"text": "完了", "bbox": [470, 1440, 550, 1475], "words": []},
    ]
    monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": "\n".join(l["text"] for l in mem_lines), "lines": mem_lines})
    monkeypatch.setattr(
        reco,
        "_ocr_bottom_orange_button",
        lambda *_a, **_kw: {
            "choice_id": 0,
            "name": "完了",
            "title": "完了",
            "description": "底部橙色主按钮: 完了",
            "bbox": {"xmin": 380, "ymin": 1420, "xmax": 640, "ymax": 1490},
        },
    )
    obs_mem = reco.recognize(mem_frame, plan="logic")
    assert [o["name"] for o in obs_mem.options] == ["完了"]







