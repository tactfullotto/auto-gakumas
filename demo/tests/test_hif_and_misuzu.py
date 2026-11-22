"""Tests for [ガラクタロード] 秦谷 美鈴 (#163), H.I.F 本戦, Good Impression / 再演 mechanics, and vision-only random P-drinks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from demo.agent.prompts import get_session_system_prompt, get_system_prompt
from demo.core.db import GakumasDB
from demo.core.idol_guides import (
    format_idol_system_snippet,
    format_idol_turn_nudge,
    get_memory_card_custom_meta,
    get_p_idol_guide,
    resolve_p_idol_id,
)
from demo.core.prompt import PromptBuilder
from demo.core.rules import (
    evaluate_card_cost_and_usability,
    format_card_conditions_zh,
    inspect_card_cost_and_conditions,
    summarize_dsl_actions,
)
from demo.core.scenarios import get_scenario_spec
from demo.core.tracker import GameStateTracker
from demo.maa_bridge.runner import GakumasRunner
from demo.schemas import BoundingBox, RecognizedCard, RecognizedDrink, ScreenObservation
from demo.vision.live_adb_recognizer import LiveScreenRecognizer


SCRATCH_HIF_PNG = Path(
    r"C:\Users\josha\.gemini\antigravity\brain\a4278b02-b7c3-44e7-80e5-b2a610c711ec\scratch\hif_current_screen2.png"
)
SCRATCH_HIF_R1_PNG = Path(
    r"C:\Users\josha\.gemini\antigravity\brain\a4278b02-b7c3-44e7-80e5-b2a610c711ec\scratch\current_emulator_check.png"
)
SCRATCH_HIF_RETRY_PNG = Path(
    r"C:\Users\josha\.gemini\antigravity\brain\a4278b02-b7c3-44e7-80e5-b2a610c711ec\scratch\hif_retry_modal.png"
)


def _dummy_obs(**kwargs) -> ScreenObservation:
    defaults = dict(
        scene_type="battle_turn",
        plan="logic",
        scenario_id="hif_honsen",
        week=7,
        total_weeks=9,
        stamina=30,
        max_stamina=44,
        genki=0,
        score=0,
        target_score=0,
        p_points=200,
    )
    defaults.update(kwargs)
    return ScreenObservation(**defaults)


def test_misuzu_163_guide_detection_and_prompt_isolation():
    guide = get_p_idol_guide(163)
    assert guide is not None
    assert guide.p_idol_id == 163
    assert "秦谷" in guide.idol_name and "ガラクタロード" in guide.title
    assert guide.plan == "logic"
    assert guide.recommended_effect == "goodImpressionTurns"
    assert "星々を統べる輝き" in guide.unique_p_item.get("name", "")

    # Auto-detect #163 from signature cards (#919..#922), without falling back to #126
    for sig_id in (919, 920, 921, 922):
        assert resolve_p_idol_id(plan="logic", deck_ids=[sig_id], default_logic_to_sekirara=False) == 163

    hand_920 = [
        RecognizedCard(
            card_id=920,
            name="ありのままのわたし+",
            playable=True,
            bbox=BoundingBox(xmin=10, ymin=1200, xmax=120, ymax=1400),
        )
    ]
    assert resolve_p_idol_id(plan="logic", hand_cards=hand_920, default_logic_to_sekirara=False) == 163

    snippet = format_idol_system_snippet(guide)
    assert "好印象" in snippet and "再演" in snippet and "除外" in snippet
    assert "私を超えて" not in snippet

    # Prompt injection for p_idol_id=163 should reflect 好印象 without leaking #126
    sys_p = get_system_prompt(plan="logic", scene_type="battle_turn", compact=False, p_idol_id=163)
    sess_p = get_session_system_prompt(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
    assert "ありのままのわたし" in sys_p and "認めてくれるひと" in sys_p
    assert "最初の鳥" not in sys_p and "最初の鳥" not in sess_p

    # Turn nudge progression across exhaust_count = 0, 7, 11
    obs0 = _dummy_obs(
        turn=1,
        turns_remaining=9,
        cards=[
            RecognizedCard(
                card_id=920,
                name="ありのままのわたし+",
                playable=True,
                bbox=BoundingBox(xmin=10, ymin=1200, xmax=120, ymax=1400),
            )
        ],
    )
    n0 = format_idol_turn_nudge(guide, obs0, exhaust_count=0)
    assert "ありのままのわたし" in n0

    obs7 = _dummy_obs(
        turn=5,
        turns_remaining=5,
        buffs={"goodImpressionTurns": 14},
        cards=[
            RecognizedCard(
                card_id=922,
                name="認めてくれるひと+",
                playable=True,
                bbox=BoundingBox(xmin=10, ymin=1200, xmax=120, ymax=1400),
            )
        ],
    )
    n7 = format_idol_turn_nudge(guide, obs7, exhaust_count=7)
    assert "認めてくれるひと" in n7 and "7" in n7

    obs11 = _dummy_obs(
        turn=8,
        turns_remaining=2,
        buffs={"goodImpressionTurns": 22},
        cards=[
            RecognizedCard(
                card_id=922,
                name="認めてくれるひと+",
                playable=True,
                bbox=BoundingBox(xmin=10, ymin=1200, xmax=120, ymax=1400),
            )
        ],
    )
    n11 = format_idol_turn_nudge(guide, obs11, exhaust_count=11)
    assert "11" in n11


def test_hif_scenario_and_memory_deck_init():
    spec = get_scenario_spec("hif_honsen")
    assert spec.total_weeks == 9
    assert spec.stat_cap == 3200
    assert get_scenario_spec("hif").scenario_id == "hif_honsen"

    with GakumasDB("gakumas_library.db") as db:
        tracker = GameStateTracker(plan="logic", p_idol_id=163)
        deck = tracker.init_produce_deck(db=db, plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        # H.I.F uses pre-built memory deck instead of Hajime N starter padding
        assert deck == [920, 922]
        assert len(deck) < 10

        # Runner in hif_honsen without explicit p_idol_id must NOT wrongly default Logic -> #126
        runner = GakumasRunner(mode="mock", plan="logic", scenario_id="hif_honsen", db=db)
        assert runner.tracker.p_idol_id is None


def test_misuzu_rules_and_tracker_gi_recast_and_removed_thresholds():
    with GakumasDB("gakumas_library.db") as db:
        c920 = db.get_card_by_id(920)
        c922 = db.get_card_by_id(922)
        assert c920 is not None and c922 is not None
        assert c920["limit_count"] == 1
        assert c922["limit_count"] is None  # repeatable

        sum922 = summarize_dsl_actions(c922.get("dsl_actions", ""))
        assert "好印象" in sum922 and "除外" in sum922

        cond_below = format_card_conditions_zh(
            c922.get("dsl_conditions", ""),
            dsl_cost=c922.get("dsl_cost", ""),
            dsl_actions=c922.get("dsl_actions", ""),
            context={"removedCards": 4, "goodImpressionTurns": 10},
            dsl_effects=c922.get("dsl_effects", ""),
        )
        assert "❌未达" in cond_below

        cond_11 = format_card_conditions_zh(
            c922.get("dsl_conditions", ""),
            dsl_cost=c922.get("dsl_cost", ""),
            dsl_actions=c922.get("dsl_actions", ""),
            context={"removedCards": 11, "goodImpressionTurns": 18},
            dsl_effects=c922.get("dsl_effects", ""),
        )
        assert "✅已满足" in cond_11

        # Evaluate cost usability & buff-cost deduction
        can_pay_missing_mot, _, _ = evaluate_card_cost_and_usability(
            card={"id": 99999, "name": "TestMotCost", "dsl_cost": "motivation-=2", "dsl_conditions": "", "dsl_actions": ""},
            stamina=30,
            genki=0,
            buffs={"motivation": 1},
            card_uses_remaining=1,
            used_limit_cards=set(),
        )
        assert not can_pay_missing_mot

        info = inspect_card_cost_and_conditions(
            c920,
            stamina=30,
            genki=0,
            buffs={"goodImpressionTurns": 6},
            card_uses_remaining=1,
            used_limit_cards=set(),
            removed_count=7,
        )
        assert info["playable"] is True

        # Tracker: #920 enters exhaust, activates +50% buff, and auto-recasts once per turn every 3 direct GI increases
        tr = GameStateTracker(plan="logic", p_idol_id=163)
        tr.init_produce_deck(db=db, plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tr.start_new_battle(db=db)
        tr.stamina = 44
        tr.genki = 0
        tr.turns_remaining = 9

        tr.record_card_play(920, db=db)
        assert 920 in tr.battle_exhaust_pile
        assert tr.good_impression_effect_buff == pytest.approx(0.5)

        # 3 direct GI increases in one turn -> triggers 1st recast (1/turn limit, remaining 2 -> 1)
        tr.genki = 30
        for _ in range(3):
            tr.record_card_play(922, db=db)  # repeatable, cost-=6 -> Discard, +GI
        assert 922 in tr.battle_discard_pile
        assert 922 not in tr.battle_exhaust_pile
        assert tr.signature_gi_recasts_remaining == 1
        assert tr.signature_gi_recast_used_turn is True
        assert tr.genki == 16  # 30 - 3*6 + 4 (再演) = 16
        assert tr.buffs.get("goodImpressionTurns", 0) >= 12  # 3 * floor(3 * 1.5) = 12

        # Further 3 GI increases same turn must wait until next turn for 2nd recast
        for _ in range(3):
            tr.record_card_play(922, db=db)
        assert tr.signature_gi_recasts_remaining == 1

        # Next turn: 3 GI increases triggers 2nd recast (1 -> 0)
        tr.start_new_turn(turns_remaining=8, db=db)
        for _ in range(3):
            tr.record_card_play(922, db=db)
        assert tr.signature_gi_recasts_remaining == 0

        # Also check buff cost deduction on motivation-= / goodImpressionTurns-=
        tr.buffs["motivation"] = 5
        tr.record_card_play(99999, limit_count=None, dsl_actions="", dsl_cost="motivation-=2")
        assert tr.buffs["motivation"] == 3


def test_vision_driven_random_p_drink_detection_0_to_4(monkeypatch):
    with GakumasDB("gakumas_library.db") as db:
        all_drinks = db.get_drinks_for_plan("logic")[:3]
        assert len(all_drinks) == 3

        frame = np.full((1600, 720, 3), 245, dtype=np.uint8)
        slots = [
            (90, 960, 210, 1080, False),
            (300, 960, 420, 1080, False),
            (510, 960, 630, 1080, False),
        ]

        # Test at multiple arbitrary weeks (e.g. week 3 & week 8) and missing "Pドリンク" text
        rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hajime_regular")
        monkeypatch.setattr(rec, "_find_reward_icon_slots", lambda *a, **kw: list(slots))
        monkeypatch.setattr(
            rec,
            "_match_square_drink_slot_with_score",
            lambda _f, x1, y1, x2, y2: (all_drinks[0] if x1 < 200 else (all_drinks[1] if x1 < 400 else all_drinks[2]), 0.85),
        )
        # Stub OCR without any "Pドリンク" string, varying weeks, 0 held drinks
        fake_lines = [{"text": "獲得したいものを選んでください", "bbox": [80, 320, 620, 360], "words": []}]
        monkeypatch.setattr(rec.ocr, "recognize_array", lambda *a, **kw: {"text": "獲得", "lines": fake_lines})
        monkeypatch.setattr(rec.en_ocr, "recognize_array", lambda *a, **kw: {"text": "", "lines": []})

        for w_num in (3, 8):
            monkeypatch.setattr(rec, "_recognize_bottom_drinks", lambda *a, **kw: [])
            obs0 = rec._parse_card_reward_screen(
                frame,
                fake_lines,
                "logic",
                {"week": w_num, "total_weeks": 13, "week_label": f"Week {w_num}", "header_label": "", "stamina": 25, "max_stamina": 31, "p_points": 120, "stats": {"vocal": 300, "dance": 300, "visual": 300}},
            )
            assert obs0.scene_type == "card_reward"
            assert obs0.week == w_num
            assert any(o.get("drink_id") == int(all_drinks[0]["id"]) for o in obs0.options)

        # In Hajime, 3 held drinks + "受け取らない" triggers cap mode (破棄候補 + 受け取らない)
        held3 = [
            RecognizedDrink(
                drink_id=int(d["id"]),
                name=d["name"],
                playable=True,
                bbox=BoundingBox(xmin=30 + i * 90, ymin=1480, xmax=96 + i * 90, ymax=1550),
                dsl_summary=d.get("description") or "",
            )
            for i, d in enumerate(all_drinks)
        ]
        dont_take_line = [{"text": "受け取らない", "bbox": [240, 1140, 460, 1185], "words": []}]
        monkeypatch.setattr(rec, "_recognize_bottom_drinks", lambda *a, **kw: held3)
        obs_hajime_cap = rec._parse_card_reward_screen(
            frame,
            fake_lines + dont_take_line,
            "logic",
            {"week": 4, "total_weeks": 13, "week_label": "Week 4", "header_label": "", "stamina": 25, "max_stamina": 31, "p_points": 120, "stats": {"vocal": 300, "dance": 300, "visual": 300}},
        )
        names_hajime = [o["name"] for o in obs_hajime_cap.options]
        assert any(n.startswith("破棄候補_") for n in names_hajime)
        assert "受け取らない" in names_hajime

        # In H.I.F 本戦 (cap_count=4), 3 held drinks is NOT capped yet, whereas 4 held drinks IS capped
        rec_hif = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        monkeypatch.setattr(rec_hif, "_find_reward_icon_slots", lambda *a, **kw: list(slots))
        monkeypatch.setattr(
            rec_hif,
            "_match_square_drink_slot_with_score",
            lambda _f, x1, y1, x2, y2: (all_drinks[0], 0.85),
        )
        monkeypatch.setattr(rec_hif.ocr, "recognize_array", lambda *a, **kw: {"text": "", "lines": []})
        monkeypatch.setattr(rec_hif.en_ocr, "recognize_array", lambda *a, **kw: {"text": "", "lines": []})

        monkeypatch.setattr(rec_hif, "_recognize_bottom_drinks", lambda *a, **kw: held3)
        obs_hif_3 = rec_hif._parse_card_reward_screen(
            frame,
            fake_lines + dont_take_line,
            "logic",
            {"week": 5, "total_weeks": 9, "week_label": "5", "header_label": "", "stamina": 40, "max_stamina": 44, "p_points": 200, "stats": {"vocal": 2400, "dance": 1200, "visual": 1700}},
        )
        assert not any(o["name"].startswith("破棄候補_") for o in obs_hif_3.options)

        held4 = held3 + [
            RecognizedDrink(
                drink_id=int(all_drinks[0]["id"]),
                name=all_drinks[0]["name"],
                playable=True,
                bbox=BoundingBox(xmin=300, ymin=1480, xmax=366, ymax=1550),
                dsl_summary=all_drinks[0].get("description") or "",
            )
        ]
        monkeypatch.setattr(rec_hif, "_recognize_bottom_drinks", lambda *a, **kw: held4)
        obs_hif_4 = rec_hif._parse_card_reward_screen(
            frame,
            fake_lines + dont_take_line,
            "logic",
            {"week": 5, "total_weeks": 9, "week_label": "5", "header_label": "", "stamina": 40, "max_stamina": 44, "p_points": 200, "stats": {"vocal": 2400, "dance": 1200, "visual": 1700}},
        )
        names_hif4 = [o["name"] for o in obs_hif_4.options]
        assert any(n.startswith("破棄候補_") for n in names_hif4)
        assert "受け取らない" in names_hif4


def test_hif_day1_live_screenshot_recognition():
    if not SCRATCH_HIF_PNG.exists():
        pytest.skip(f"Live screenshot not present: {SCRATCH_HIF_PNG}")
    frame = np.array(Image.open(SCRATCH_HIF_PNG))[:, :, :3]
    rec = LiveScreenRecognizer(plan="logic", scenario_id="hif_honsen")
    obs = rec.recognize(frame)
    assert obs.scene_type == "produce_schedule"
    assert obs.week == 1
    assert obs.total_weeks == 9
    assert obs.stamina == 44 and obs.max_stamina == 44
    assert obs.p_points == 200
    assert obs.idol_stats == {"vocal": 2393, "dance": 1108, "visual": 1644}
    assert [o["name"] for o in obs.options] == ["Vo.授業", "Da.授業", "Vi.授業"]


def test_left_buff_diamonds_classify_good_impression_vs_motivation():
    """👍 (single thumb top + solid palm) -> goodImpressionTurns; ✊ (separated knuckles) -> motivation."""
    frame = np.full((1600, 720, 3), 60, dtype=np.uint8)

    def _stamp_diamond_with_number(arr: np.ndarray, y0: int, x0: int, kind: str, label: str) -> None:
        # 36x36 rotated-square blue diamond inside y0..y0+36, x0..x0+36
        cx, cy, rad = 18, 18, 17
        for r in range(36):
            for c in range(36):
                if abs(r - cy) + abs(c - cx) <= rad:
                    arr[y0 + r, x0 + c] = (35, 145, 240)
        # Inner white glyph strictly bounded inside blue diamond
        if kind == "thumb":
            # Raised single thumb top (rows 6..18, cols 14..21) + solid palm bottom (rows 18..29, cols 9..27)
            arr[y0 + 6 : y0 + 18, x0 + 14 : x0 + 21] = (255, 255, 255)
            arr[y0 + 18 : y0 + 29, x0 + 9 : x0 + 27] = (255, 255, 255)
        else:
            # 4 separated knuckle bars (rows 7..18) + compact bottom palm (rows 19..27)
            for kx in (9, 14, 19, 24):
                arr[y0 + 7 : y0 + 18, x0 + kx : x0 + kx + 3] = (255, 255, 255)
            arr[y0 + 19 : y0 + 27, x0 + 10 : x0 + 25] = (255, 255, 255)

        # Draw crisp black-outlined white stack count immediately to the right of diamond
        pil = Image.fromarray(arr)
        dr = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("arialbd.ttf", 28)
        except Exception:
            font = ImageFont.load_default()
        tx, ty = x0 + 44, y0 + 2
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx != 0 or dy != 0:
                    dr.text((tx + dx, ty + dy), label, fill=(20, 20, 20), font=font)
        dr.text((tx, ty), label, fill=(255, 255, 255), font=font)
        arr[:] = np.array(pil)

    # Top blue diamond in y=200..236: 👍 Good Impression = 14
    _stamp_diamond_with_number(frame, y0=200, x0=24, kind="thumb", label="14")
    # Second blue diamond in y=280..316: ✊ Motivation = 7
    _stamp_diamond_with_number(frame, y0=280, x0=24, kind="fist", label="7")

    rec = LiveScreenRecognizer(plan="logic", scenario_id="hif_honsen")
    buffs = rec._parse_left_buff_diamonds(frame)
    assert buffs.get("goodImpressionTurns") == 14
    assert buffs.get("motivation") == 7


def _load_jp_font(size: int = 24):
    for p in (
        r"C:\Windows\Fonts\meiryob.ttc",
        r"C:\Windows\Fonts\YuGothB.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    pytest.skip("No Japanese TrueType font installed for WinRT OCR rendering test")


def test_change_acquire_card_description_end_of_turn_routes_to_card_reward(monkeypatch):
    """チェンジで獲得 + '以降、ターン終了時' card text must route to card_reward, not battle_turn."""
    frame = np.full((1600, 720, 3), 90, dtype=np.uint8)
    # Paint an active orange '決定' confirm pill near bottom center (y=1270..1350, x=210..510)
    frame[1270:1350, 210:510] = (245, 130, 35)

    rec = LiveScreenRecognizer(plan="logic", scenario_id="hif_honsen")
    fake_lines = [
        {"text": "チェンジで獲得するスキルカードを選んでください", "bbox": [50, 440, 670, 485]},
        {"text": "虹色ドリーマー", "bbox": [120, 620, 390, 660]},
        {"text": "以降、ターン終了時、好印象が3以上の場合、好印象+3", "bbox": [110, 670, 640, 715]},
        {"text": "決定", "bbox": [310, 1290, 410, 1335]},
    ]
    pv_lines = [
        {"text": "虹色ドリーマー", "bbox": [40, 25, 310, 65]},
        {"text": "以降、ターン終了時、好印象が3以上の場合、好印象+3", "bbox": [40, 75, 560, 115]},
    ]
    monkeypatch.setattr(
        rec.ocr,
        "recognize_array",
        lambda arr: (
            {"text": "\n".join(l["text"] for l in fake_lines), "lines": fake_lines}
            if arr.shape[0] >= 1200
            else {"text": "\n".join(l["text"] for l in pv_lines), "lines": pv_lines}
        ),
    )
    monkeypatch.setattr(rec, "_ocr_crop_upscaled", lambda *a, **kw: "")

    obs = rec.recognize(frame)
    assert obs.scene_type == "card_reward"
    assert any(o.get("name") in ("受け取る", "決定") for o in obs.options)


def test_acquisition_guide_modal_routes_to_popup(monkeypatch):
    """獲得ガイド表示設定 modal must classify as popup with 決定 option."""
    frame = np.full((1600, 720, 3), 90, dtype=np.uint8)
    frame[1260:1345, 220:500] = (245, 130, 35)

    rec = LiveScreenRecognizer(plan="logic", scenario_id="hif_honsen")
    fake_lines = [
        {"text": "獲得ガイド表示設定", "bbox": [190, 560, 530, 605]},
        {"text": "メニュー>設定からいつでも変更できます", "bbox": [120, 660, 600, 700]},
        {"text": "決定", "bbox": [315, 1280, 405, 1325]},
    ]
    pv_lines = [
        {"text": "虹色ドリーマー", "bbox": [40, 25, 310, 65]},
        {"text": "以降、ターン終了時、好印象が3以上の場合、好印象+3", "bbox": [40, 75, 560, 115]},
    ]
    monkeypatch.setattr(
        rec.ocr,
        "recognize_array",
        lambda arr: (
            {"text": "\n".join(l["text"] for l in fake_lines), "lines": fake_lines}
            if arr.shape[0] >= 1200
            else {"text": "\n".join(l["text"] for l in pv_lines), "lines": pv_lines}
        ),
    )
    monkeypatch.setattr(rec, "_ocr_crop_upscaled", lambda *a, **kw: "")

    obs = rec.recognize(frame)
    assert obs.scene_type == "popup"
    assert any(o.get("name") == "決定" for o in obs.options)


def test_find_event_choice_pills_detects_orange_text_dialogue_choices():
    """Orange-text choices on white pills must be recovered by _find_event_choice_pills while skipping 早送り."""
    frame = np.full((1600, 720, 3), 45, dtype=np.uint8)
    pill_specs = [
        (960, "長い道のりでした"),
        (1060, "必ず勝ちます"),
        (1160, "楽しんできます"),
    ]
    font = _load_jp_font(26)
    pil = Image.fromarray(frame)
    dr = ImageDraw.Draw(pil)
    for cy, label in pill_specs:
        dr.rounded_rectangle((110, cy - 34, 640, cy + 34), radius=24, fill=(250, 250, 250))
        dr.text((160, cy - 16), label, fill=(245, 112, 35), font=font)
    frame = np.array(pil)

    rec = LiveScreenRecognizer(plan="logic", scenario_id="hif_honsen")
    # Simulate full-frame OCR missing the orange-text white pills and only catching HUD 早送り
    sparse_lines = [{"text": "早送り", "bbox": [580, 1480, 680, 1520]}]
    pills = rec._find_event_choice_pills(frame, sparse_lines, 720, 1600)
    names = [p["text"] for p in pills]
    assert len(pills) >= 3
    assert any("長い" in n for n in names)
    assert any("勝ち" in n for n in names)
    assert any("楽し" in n for n in names)
    assert all("早送り" not in n for n in names)


def test_scan_deck_grid_slots_excludes_monochrome_trouble_card():
    """Monochrome dark-gray 眠気 Trouble cells (mean_v < 118, sat < 32) must be excluded from deck grid."""
    frame = np.full((1600, 720, 3), 30, dtype=np.uint8)
    # Row 1 in _scan_deck_grid_slots (720x1600): y=1200..1336; cols: [79..198, 223..342, 367..486]
    frame[1200:1336, 79:198] = (220, 120, 45)
    frame[1200:1336, 223:342] = (65, 165, 225)
    frame[1200:1336, 367:486] = (90, 92, 95)

    rec = LiveScreenRecognizer(plan="logic", scenario_id="hif_honsen")
    slots = rec._scan_deck_grid_slots(frame, 720, 1600)
    assert len(slots) == 2
    cx_list = [(x1 + x2) // 2 for x1, _, x2, _, _, _ in slots]
    assert all(not (367 <= cx <= 486) for cx in cx_list)


def test_dimmed_four_slot_drink_cap_recognition(monkeypatch):
    """Dimmed (*0.45) 4-slot H.I.F drinks + Pドリンク所持上限 must expose 破棄候補_<id> and 受け取らない."""
    rec = LiveScreenRecognizer(plan="logic", scenario_id="hif_honsen")
    templates = rec._load_drink_templates()
    target_ids = [25, 12, 5, 14]
    chosen_drinks = []
    for did in target_ids:
        row = rec.db.get_drink_by_id(did)
        if row and row.get("icon_path") and Path(row["icon_path"]).exists():
            chosen_drinks.append(row)
    if len(chosen_drinks) < 4:
        seen = {int(r["id"]) for r in chosen_drinks}
        for row in rec.db.get_drinks_for_plan("logic"):
            if int(row["id"]) not in seen and row.get("icon_path") and Path(row["icon_path"]).exists():
                chosen_drinks.append(row)
                seen.add(int(row["id"]))
            if len(chosen_drinks) >= 4:
                break
    assert len(chosen_drinks) == 4 and len(templates) >= 4

    frame = np.full((1600, 720, 3), 24, dtype=np.uint8)
    h, w = frame.shape[:2]
    rad = int(round(w * 0.0458))
    cy = int(round(h * 0.9545))
    x_fracs = (0.0833, 0.2042, 0.3222, 0.4403)
    for row, x_frac in zip(chosen_drinks, x_fracs):
        cx = int(round(w * x_frac))
        arr_d = np.array(Image.open(row["icon_path"]).convert("RGBA"))
        alpha = arr_d[:, :, 3:4].astype(np.float32) / 255.0
        comp = (arr_d[:, :, :3].astype(np.float32) * alpha + 245.0 * (1.0 - alpha)) * 0.45
        tile = np.array(
            Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8)).resize((rad * 2, rad * 2), Image.Resampling.BILINEAR)
        )
        frame[cy - rad : cy + rad, cx - rad : cx + rad] = tile

    # Paint a candidate drink slot in the upper selection band too (y=0.64..0.72, x=240..310)
    cand_row = chosen_drinks[0]
    arr_c = np.array(Image.open(cand_row["icon_path"]).convert("RGBA"))
    alpha_c = arr_c[:, :, 3:4].astype(np.float32) / 255.0
    comp_c = np.clip(arr_c[:, :, :3].astype(np.float32) * alpha_c + 245.0 * (1.0 - alpha_c), 0, 255).astype(np.uint8)
    cand_tile = np.array(Image.fromarray(comp_c).resize((66, 66), Image.Resampling.BILINEAR))
    frame[1035:1101, 240:306] = cand_tile
    # Render crisp white "Pドリンク所持上限" in cap band (0.28w..0.72w, 0.785h..0.84h) -> (201..518, 1256..1344)
    font = _load_jp_font(24)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((235, 1282), "Pドリンク所持上限", fill=(255, 255, 255), font=font)
    frame = np.array(pil)

    held = rec._recognize_bottom_drinks(frame, "logic")
    assert len(held) == 4
    assert {d.drink_id for d in held} == {int(r["id"]) for r in chosen_drinks}

    fake_lines = [
        {"text": "獲得するPドリンクを選んでください", "bbox": [90, 460, 630, 505]},
        {"text": "受け取らない", "bbox": [270, 1140, 450, 1175]},
    ]
    monkeypatch.setattr(
        rec.ocr,
        "recognize_array",
        lambda arr: {"text": "", "lines": []},
    )
    obs = rec._parse_card_reward_screen(
        frame,
        fake_lines,
        "logic",
        {
            "week": 5,
            "total_weeks": 9,
            "week_label": "H.I.F本戦まで 2日",
            "header_label": "",
            "stamina": 36,
            "max_stamina": 44,
            "p_points": 300,
            "stats": {"vocal": 2800, "dance": 1200, "visual": 1900},
        },
    )
    opt_names = [o["name"] for o in obs.options]
    for did in {int(r["id"]) for r in chosen_drinks}:
        assert f"破棄候補_{did}" in opt_names
    assert "受け取らない" in opt_names


def test_mid_battle_transient_skip_preserves_in_battle_and_exhaust(monkeypatch):
    """A transient mid-battle skip frame must NOT reset in_battle or wipe battle_exhaust_pile."""
    from demo.schemas import DecisionResult

    class _DummyCtrl:
        is_connected = True

        def connect(self) -> bool:
            return True

        def disconnect(self) -> None:
            self.is_connected = False

        def screencap(self) -> np.ndarray:
            return np.full((1600, 720, 3), 120, dtype=np.uint8)

        def click(self, x: int, y: int) -> bool:
            return True

    db = GakumasDB("gakumas_library.db")
    dummy_agent = SimpleNamespace(
        decide=lambda **kw: DecisionResult(choice_id=0, reason="test", source="test"),
        reset_session=lambda: None,
        export_session=lambda: {},
        load_session=lambda d: True,
    )
    runner = GakumasRunner(
        mode="mock",
        controller=_DummyCtrl(),
        agent=dummy_agent,
        db=db,
        plan="logic",
        scenario_id="hif_honsen",
        p_idol_id=163,
        session_mode=True,
    )
    seq_obs = [
        _dummy_obs(
            scene_type="battle_turn",
            week=7,
            turn=1,
            turns_remaining=9,
            stamina=42,
            max_stamina=44,
            genki=9,
            score=57,
            cards=[
                RecognizedCard(
                    card_id=158,
                    name="アイドル魂",
                    playable=True,
                    bbox=BoundingBox(xmin=200, ymin=1200, xmax=360, ymax=1440),
                )
            ],
            options=[{"choice_id": 0, "name": "SKIP", "action_type": "end_turn", "bbox": {"xmin": 630, "ymin": 1080, "xmax": 710, "ymax": 1165}}],
        ),
        _dummy_obs(
            scene_type="skip",
            week=7,
            turn=1,
            turns_remaining=9,
            options=[{"choice_id": 0, "name": "点击屏幕继续", "bbox": {"xmin": 260, "ymin": 1240, "xmax": 460, "ymax": 1400}}],
            raw_text="",
        ),
        _dummy_obs(
            scene_type="battle_turn",
            week=7,
            turn=2,
            turns_remaining=8,
            stamina=41,
            max_stamina=44,
            genki=12,
            score=2276,
            cards=[
                RecognizedCard(
                    card_id=920,
                    name="ありのままのわたし+",
                    upgraded=True,
                    playable=True,
                    bbox=BoundingBox(xmin=200, ymin=1200, xmax=360, ymax=1440),
                )
            ],
            options=[{"choice_id": 0, "name": "SKIP", "action_type": "end_turn", "bbox": {"xmin": 630, "ymin": 1080, "xmax": 710, "ymax": 1165}}],
        ),
    ]
    idx_holder = {"i": 0}

    def _next_obs(_frame):
        o = seq_obs[idx_holder["i"]]
        idx_holder["i"] += 1
        return o

    monkeypatch.setattr(runner.scene_routing_reco, "recognize", _next_obs)
    runner.start()
    try:
        r1 = runner.run_once()
        assert r1.success
        assert runner.tracker.in_battle is True
        assert 158 in runner.tracker.battle_exhaust_pile

        r2 = runner.run_once()
        assert r2.success
        assert runner.tracker.in_battle is True
        assert runner.tracker.battle_exhaust_pile == [158]

        r3 = runner.run_once()
        assert r3.success
        assert runner.tracker.in_battle is True
        assert 158 in runner.tracker.battle_exhaust_pile
    finally:
        runner.stop()

    # Also verify LiveScreenRecognizer._in_battle_scene debounces 1-2 blank fallback frames, but clears on CLEAR
    rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
    rec._in_battle_scene = True
    rec._last_p_points = 300
    blank = np.full((1600, 720, 3), 120, dtype=np.uint8)
    monkeypatch.setattr(rec.ocr, "recognize_array", lambda *a, **kw: {"text": "", "lines": []})
    monkeypatch.setattr(rec.en_ocr, "recognize_array", lambda *a, **kw: {"text": "", "lines": []})
    monkeypatch.setattr(rec, "_ocr_crop_upscaled", lambda *a, **kw: "")
    rec.recognize(blank)
    assert rec._in_battle_scene is True
    assert rec._last_p_points == 300
    monkeypatch.setattr(rec.ocr, "recognize_array", lambda *a, **kw: {"text": "CLEAR", "lines": [{"text": "CLEAR", "bbox": [260, 420, 460, 480]}]})
    rec.recognize(blank)
    assert rec._in_battle_scene is False


def test_hif_round1_battle_hud_live_screenshot():
    if not SCRATCH_HIF_R1_PNG.exists():
        pytest.skip(f"Live R1 screenshot not present: {SCRATCH_HIF_R1_PNG}")
    frame = np.array(Image.open(SCRATCH_HIF_R1_PNG))[:, :, :3]
    rec = LiveScreenRecognizer(plan="logic", scenario_id="hif_honsen")
    rec._last_p_points = 300
    obs = rec.recognize(frame)
    assert obs.scene_type == "battle_turn"
    assert obs.week == 7
    assert obs.turn == 3
    assert obs.turns_remaining == 7
    assert obs.stamina == 37
    assert obs.genki == 11
    assert obs.score == 13260
    assert obs.p_points == 300
    assert obs.buffs == {"goodImpressionTurns": 41, "motivation": 4}
    assert [d.drink_id for d in obs.drinks] == [25, 12, 5]
    assert {920, 521}.issubset({c.card_id for c in obs.cards})


def test_hif_memory_card_exile_nemuke_and_free_extra_action():
    db = GakumasDB("gakumas_library.db")
    for cid in (296, 297, 298, 299, 128, 129):
        meta = get_memory_card_custom_meta("hif_honsen", 163, cid)
        assert meta["exiles_trouble"] is True
        assert not get_memory_card_custom_meta("hajime_regular", 163, cid).get("exiles_trouble")

    tracker = GameStateTracker()
    tracker.init_produce_deck(db=db, plan="logic", p_idol_id=163, scenario_id="hif_honsen")
    tracker.produce_deck = [297, 23, 137, 922]
    tracker.start_new_battle(db=db)
    tracker.battle_hand = [137, 297, 23]
    tracker.card_uses_remaining = 1

    c137 = db.get_card_by_id(137)
    assert c137 is not None
    tracker.record_card_play(
        137,
        limit_count=c137.get("limitCount"),
        base_id=c137.get("baseId"),
        dsl_actions=c137.get("dslActions", ""),
        db=db,
    )
    assert tracker.card_uses_remaining == 1
    assert 137 in tracker.battle_exhaust_pile
    assert get_memory_card_custom_meta("hif_honsen", 163, 137, tracker.battle_played_counts)["grant_extra_use_now"] is False

    c297 = db.get_card_by_id(297)
    assert c297 is not None
    tracker.record_card_play(
        297,
        limit_count=c297.get("limitCount"),
        base_id=c297.get("baseId"),
        dsl_actions=c297.get("dslActions", ""),
        db=db,
    )
    assert 23 not in tracker.battle_hand
    assert 23 in tracker.battle_exhaust_pile


def test_hif_drink_25_timing_and_778_combo_in_prompt():
    db = GakumasDB("gakumas_library.db")
    builder = PromptBuilder(db=db)
    tracker = GameStateTracker()
    tracker.init_produce_deck(db=db, plan="logic", p_idol_id=163, scenario_id="hif_honsen")
    tracker.start_new_battle(db=db)

    obs_a = _dummy_obs(
        week=7,
        turns_remaining=2,
        stamina=25,
        max_stamina=44,
        buffs={"goodImpressionTurns": 35, "motivation": 6},
        drinks=[
            RecognizedDrink(
                drink_id=25,
                name="特製ハツボシエキス",
                playable=True,
                bbox=BoundingBox(xmin=0, ymin=1000, xmax=80, ymax=1080),
            )
        ],
        cards=[
            RecognizedCard(
                card_id=299,
                name="オトメゴコロ+",
                upgraded=True,
                type="mental",
                playable=True,
                hp_loss=0,
                cost_type="green",
                raw_cost=0,
                effective_cost=0,
                bbox=BoundingBox(xmin=100, ymin=1100, xmax=250, ymax=1350),
            )
        ],
    )
    prompt_a, _, _ = builder.build_session_prompt(obs_a, tracker, commit=False)
    assert "严禁空喝" in prompt_a
    assert "终盘必喝" not in prompt_a

    obs_b = _dummy_obs(
        week=7,
        turns_remaining=4,
        stamina=25,
        max_stamina=44,
        buffs={"goodImpressionTurns": 45, "motivation": 6},
        drinks=[
            RecognizedDrink(
                drink_id=25,
                name="特製ハツボシエキス",
                playable=True,
                bbox=BoundingBox(xmin=0, ymin=1000, xmax=80, ymax=1080),
            )
        ],
        cards=[
            RecognizedCard(
                card_id=778,
                name="びしっとキメ顔+",
                upgraded=True,
                type="active",
                rarity="SSR",
                playable=True,
                hp_loss=5,
                cost_type="green",
                raw_cost=5,
                effective_cost=5,
                bbox=BoundingBox(xmin=100, ymin=1100, xmax=250, ymax=1350),
            ),
            RecognizedCard(
                card_id=922,
                name="認めてくれるひと+",
                upgraded=True,
                type="active",
                rarity="R",
                playable=True,
                hp_loss=0,
                cost_type="green",
                raw_cost=0,
                effective_cost=0,
                bbox=BoundingBox(xmin=280, ymin=1100, xmax=430, ymax=1350),
            ),
        ],
    )
    prompt_b, _, _ = builder.build_session_prompt(obs_b, tracker, commit=False)
    assert "びしっとキメ顔" in prompt_b
    assert "王炸" in prompt_b
    assert "🔥0扣血免费A卡" not in prompt_b


def test_hif_popup_and_shop_vision_regressions():
    db = GakumasDB("gakumas_library.db")
    rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")

    if SCRATCH_HIF_RETRY_PNG.exists():
        retry_frame = np.array(Image.open(SCRATCH_HIF_RETRY_PNG))[:, :, :3]
        retry_obs = rec.recognize(retry_frame)
        retry_names = [o.get("name") for o in retry_obs.options]
        assert retry_obs.scene_type == "popup"
        assert "再挑戦" in retry_names
        assert "プロデュース終了" not in retry_names

    meta = {
        "week": 8,
        "total_weeks": 9,
        "week_label": "8日目",
        "stats": {"vocal": 500, "dance": 500, "visual": 500},
        "p_points": 250,
        "stamina": 30,
        "max_stamina": 44,
        "drinks": [],
    }
    blank_frame = np.zeros((1600, 720, 3), dtype=np.uint8)
    rec._in_battle_scene = True
    ink_lines = [
        {"text": "ンク詳細", "bbox": (220, 380, 500, 430)},
        {"text": "使う", "bbox": (420, 1140, 620, 1200)},
    ]
    ink_obs = rec._parse_popup_screen(blank_frame, ink_lines, "logic", meta)
    assert ink_obs.scene_type == "popup"
    assert "使う" in [o.get("name") for o in ink_obs.options]

    rec._in_battle_scene = False
    bright_shop = np.full((1600, 720, 3), 235, dtype=np.uint8)
    shop_lines = [
        {"text": "相談", "bbox": (100, 120, 300, 160)},
        {"text": "セレクトチェンジ", "bbox": (60, 1120, 320, 1180)},
        {"text": "カスタマイズ", "bbox": (380, 1120, 660, 1180)},
        {"text": "終了", "bbox": (550, 1280, 680, 1330)},
    ]
    shop_obs = rec._parse_shop_screen(bright_shop, shop_lines, "logic", meta)
    shop_by_name = {o.get("name"): o for o in shop_obs.options}
    assert "セレクトチェンジ" in shop_by_name
    assert "カスタマイズ" in shop_by_name
    assert shop_by_name["セレクトチェンジ"].get("shop_slot_key")
    assert shop_by_name["カスタマイズ"].get("shop_slot_key")
    assert "強化" not in shop_by_name

    rec._shop_visited_slots.add("セレクトチェンジ")
    shop_obs2 = rec._parse_shop_screen(bright_shop, shop_lines, "logic", meta)
    assert "セレクトチェンジ" not in [o.get("name") for o in shop_obs2.options]


def test_shop_never_misclassified_as_battle_turn_and_detects_pink_upgrade():
    db = GakumasDB("gakumas_library.db")
    rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
    orig_ocr = rec.ocr
    orig_en_ocr = rec.en_ocr

    shop_full_lines = [
        {"text": "相談", "bbox": [90, 60, 210, 100]},
        {"text": "Pポイントと交換するものを選んでください", "bbox": [110, 420, 610, 460]},
        {"text": "スキルカード使用数追加+1", "bbox": [480, 740, 680, 780]},
        {"text": "好印象増加量増加+50%（5ターン）", "bbox": [180, 740, 470, 780]},
        {"text": "あと1回", "bbox": [65, 1185, 145, 1215]},
        {"text": "リフレッシュ", "bbox": [45, 1225, 195, 1265]},
        {"text": "終了", "bbox": [585, 1220, 680, 1265]},
    ]
    rec.ocr = SimpleNamespace(
        recognize_array=lambda arr, *a, **kw: {"text": "", "lines": shop_full_lines if arr.shape[:2] == (1600, 720) else []}
    )
    rec.en_ocr = SimpleNamespace(recognize_array=lambda *a, **kw: {"text": "", "lines": []})
    shop_frame = np.full((1600, 720, 3), 235, dtype=np.uint8)
    obs = rec.recognize(shop_frame)
    assert obs.scene_type == "shop"

    rec.ocr = orig_ocr
    rec.en_ocr = orig_en_ocr
    # Paint vivid pink 強化 button (mean ~156.7 < old 165 threshold, p85=240) and cyan 削除 button
    pink_frame = np.full((1600, 720, 3), 235, dtype=np.uint8)
    pink_frame[1090:1165, 55:340] = (240, 95, 135)
    pink_frame[1090:1165, 380:665] = (80, 195, 250)
    pink_lines = [
        {"text": "相談", "bbox": (90, 60, 210, 100)},
        {"text": "強化", "bbox": (110, 1110, 190, 1148)},
        {"text": "100", "bbox": (210, 1110, 275, 1148)},
        {"text": "削除", "bbox": (435, 1110, 515, 1148)},
        {"text": "125", "bbox": (535, 1110, 600, 1148)},
        {"text": "終了", "bbox": (585, 1220, 680, 1265)},
    ]
    meta6 = {
        "week": 6,
        "total_weeks": 9,
        "week_label": "6日目",
        "stats": {"vocal": 2730, "dance": 1572, "visual": 1849},
        "p_points": 250,
        "stamina": 33,
        "max_stamina": 44,
        "drinks": [],
    }
    obs_pink = rec._parse_shop_screen(pink_frame, pink_lines, "logic", meta6)
    by_name = {o.get("name"): o for o in obs_pink.options}
    assert "強化" in by_name and by_name["強化"].get("price") == 100
    assert "削除" in by_name and by_name["削除"].get("price") == 125


def test_sp_lesson_recognition_recovery_and_live_day1_no_false_sp():
    db = GakumasDB("gakumas_library.db")
    rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")

    day1_path = Path(
        r"C:\Users\josha\.gemini\antigravity\brain\a4278b02-b7c3-44e7-80e5-b2a610c711ec\scratch\phase1_precheck.png"
    )
    if day1_path.exists():
        day1_frame = np.array(Image.open(day1_path))[:, :, :3]
        day1_obs = rec.recognize(day1_frame)
        assert day1_obs.scene_type == "produce_schedule"
        assert [o["name"] for o in day1_obs.options] == ["Vo.授業", "Da.授業", "Vi.授業"]

    frame = np.full((1600, 720, 3), 40, dtype=np.uint8)
    for slot_cx in (118, 277, 435, 595):
        frame[1150:1430, max(0, slot_cx - 65) : min(720, slot_cx + 65)] = (235, 235, 235)
    # Rainbow gold + magenta SP top-left badge on Slot 0 (cx=118) and Slot 2 (cx=435)
    for sp_cx in (118, 435):
        frame[1190:1212, max(0, sp_cx - 58) : sp_cx - 22] = (245, 200, 50)
        frame[1215:1237, max(0, sp_cx - 58) : sp_cx - 22] = (240, 80, 210)
    # Blue Da attribute icon on Slot 1 (cx=277)
    frame[1305:1355, 277:325] = (55, 145, 245)

    rec.ocr = SimpleNamespace(recognize_array=lambda *a, **kw: {"text": "", "lines": []})
    pass1_lines = [
        {"text": "Da.公開レッスン", "bbox": [220, 1380, 334, 1412]},
        {"text": "休む", "bbox": [555, 1380, 635, 1412]},
    ]
    opts = rec._find_keyword_options(frame, pass1_lines, 720, 1600)
    assert [o["name"] for o in opts] == ["Vo.SP公開レッスン", "Da.公開レッスン", "Vi.SP公開レッスン", "休む"]
    assert "🌟SP" in opts[0]["description"]
    assert "🌟SP" not in opts[1]["description"]
    assert "🌟SP" in opts[2]["description"]
    assert "SP" in get_session_system_prompt("logic", scenario_id="hif_honsen", p_idol_id=163)


def test_four_digit_stat_truncation_guard():
    db = GakumasDB("gakumas_library.db")
    rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
    rec._last_week = 5
    rec._last_stats = {"vocal": 2393, "dance": 1288, "visual": 1644}

    def _fake_crop(frame, box, *args, **kwargs):
        x1, y1, _, _ = box
        if y1 > 400:
            if x1 < 250:
                return "2444/3200"
            if x1 < 400:
                return "1468/3200"
            return "14/3200"
        return ""

    rec._ocr_crop_upscaled = _fake_crop  # type: ignore[method-assign]
    blank = np.zeros((1600, 720, 3), dtype=np.uint8)
    lines = [
        {"text": "H.I.F本戦まで 2日", "bbox": [30, 30, 200, 70]},
        {"text": "/3200", "bbox": [200, 1040, 280, 1070]},
    ]
    meta = rec._parse_week_and_stats(lines, blank)
    assert meta["stats"]["vocal"] == 2444
    assert meta["stats"]["dance"] == 1468
    assert meta["stats"]["visual"] == 1644
