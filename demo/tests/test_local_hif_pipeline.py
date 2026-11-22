"""Tests for zero-LLM H.I.F 本戦 local automation pipeline (LoadoutScanner, LocalAgent, LiveBattlePlanner, Runner)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from demo.agent.local_agent import LocalHybridDecisionAgent
from demo.core.db import GakumasDB
from demo.core.prompt import PromptBuilder
from demo.core.tracker import GameStateTracker
from demo.maa_bridge.mock_controller import MockController
from demo.maa_bridge.runner import GakumasRunner
from demo.schemas import BoundingBox, DecisionResult, LegalAction, RecognizedCard, RecognizedDrink, ScreenObservation
from demo.sim.live_battle_planner import (
    LiveBattlePlanner,
    LiveBattlePlannerConfig,
    build_live_battle_config,
)
from demo.vision.loadout_scanner import HifLoadoutScanner, match_p_item_by_text


def test_tracker_loadout_library_and_checkpoint_roundtrip(tmp_path: Path):
    db = GakumasDB("gakumas_library.db")
    try:
        tracker = GameStateTracker(plan="logic")
        p_items = [
            {"item_id": 509, "canonical_name": "星々を統べる輝き+", "mode": "stage"},
            {"item_id": 404, "canonical_name": "咲季の完全食・改", "mode": "stage"},
            {"item_id": 447, "canonical_name": "H.I.Fワッペン", "mode": "produce"},
        ]
        deck_ids = [920, 922, 129, 127, 137, 137, 23, 23]
        hooks = {127: {"force_initial_hand": True}, 129: {"force_initial_hand": True}}
        tracker.set_initial_loadout(
            p_idol_id=163,
            plan="logic",
            recommended_effect="goodImpressionTurns",
            deck_ids=deck_ids,
            p_items=p_items,
            card_memory_hooks=hooks,
            db=db,
        )

        lib = tracker.export_produce_library(db=db)
        assert lib["p_idol_id"] == 163
        assert lib["recommended_effect"] == "goodImpressionTurns"
        assert lib["deck_ids"] == deck_ids
        assert len(lib["p_items"]) == 3
        assert tracker.initial_loadout_captured is True
        assert any(c["card_id"] == 129 and c["memory_hooks"].get("force_initial_hand") for c in lib["cards"])

        ckpt_file = tmp_path / "ckpt.json"
        saved = tracker.export_checkpoint()
        ckpt_file.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

        restored = GameStateTracker()
        restored.load_checkpoint(json.loads(ckpt_file.read_text(encoding="utf-8")), db=db)
        assert restored.p_idol_id == 163
        assert restored.recommended_effect == "goodImpressionTurns"
        assert restored.produce_deck == deck_ids
        assert len(restored.equipped_p_items) == 3
        assert restored.card_memory_hooks[127]["force_initial_hand"] is True

        restored.reset(keep_deck=False)
        assert restored.initial_loadout_captured is False
        assert restored.produce_deck == []
    finally:
        db.close()



def test_loadout_scanner_disambiguation_duplicates_and_body_fallback():
    # Verify body-text fallback on garbled P-item OCR titles
    p510 = match_p_item_by_text(";咼ーつ午コイフ", "ロジックのスキルカード獲得時 | ボーカル上昇+25 | ランダムなスキルカードを強化")
    assert p510["item_id"] == 510
    assert p510["canonical_name"] == "過激め？チョイス"

    db = GakumasDB("gakumas_library.db")
    scanner = HifLoadoutScanner(db=db)
    try:
        # Build a textured 2200x720 virtual scroll document so template matching measures exact 420px scroll offset
        rng = np.random.default_rng(7)
        doc = np.full((2200, 720, 3), 255, dtype=np.uint8)
        # Box Y coords: 726 (rel 320), 980 (rel 574) on fr0; 1230 (rel 404), 1480 (rel 654), 1710 (rel 884) on fr1
        for box_y in (726, 980, 1230, 1480, 1710):
            doc[box_y : box_y + 115, 55:170] = rng.integers(30, 140, size=(115, 115, 3), dtype=np.uint8)
            doc[box_y + 10 : box_y + 105, 200:600] = rng.integers(180, 240, size=(95, 400, 3), dtype=np.uint8)
        fr0 = doc[:1600].copy()
        fr1 = doc[420:2020].copy()

        def _fake_ja_ocr(arr: np.ndarray, *a, **kw):
            ah, aw = arr.shape[:2]
            # Header band 0.254..0.415 -> h ~ 257
            if 220 <= ah <= 300:
                return {
                    "text": "SSR ガラクタロード 秦谷 美鈴",
                    "lines": [
                        {"text": "SSR ガラクタロード", "bbox": [180, 20, 460, 50]},
                        {"text": "秦谷 美鈴", "bbox": [180, 55, 360, 85]},
                    ],
                }
            # Page 0 full viewport (h=966, offset 406)
            if ah >= 900 and np.array_equal(arr, fr0[int(1600 * 0.254) : int(1600 * 0.858), :]):
                return {
                    "text": "",
                    "lines": [
                        {"text": "Pアイテム", "bbox": [24, 270, 140, 295]},
                        {"text": "星々を統べる輝き+", "bbox": [200, 325, 480, 355]},
                        {"text": "直接効果で好印象が4回増加時", "bbox": [200, 370, 560, 395]},
                        {"text": "スキルカード", "bbox": [24, 525, 160, 550]},
                        {"text": "ワクワクが止まらない+", "bbox": [200, 580, 480, 610]},
                        {"text": "レッスン開始時手札に入る / トラブルカードを除外", "bbox": [200, 620, 620, 650]},
                    ],
                }
            # Page 1 full viewport (offset 420+406=826 -> rel y=404, 654, 884)
            if ah >= 900:
                return {
                    "text": "",
                    "lines": [
                        {"text": "やる気は満点+", "bbox": [200, 410, 450, 440]},
                        {"text": "スキルカード使用数追加+1", "bbox": [200, 450, 520, 480]},
                        {"text": "眠気", "bbox": [200, 660, 310, 690]},
                        {"text": "眠気", "bbox": [200, 890, 310, 920]},
                    ],
                }
            return {"text": "", "lines": []}

        def _fake_en_ocr(arr: np.ndarray, *a, **kw):
            aw = arr.shape[1]
            # 3 upscaled stat columns: Vo=2393, Da=1108, Vi=1644
            if aw < 400:
                # Distinguish via caller x crop isn't passed; use sequential counter
                _fake_en_ocr.idx = (_fake_en_ocr.idx + 1) % 3
                vals = ["2393", "1108", "1644"]
                return {"text": vals[_fake_en_ocr.idx], "lines": []}
            return {"text": "2393 1108 1644", "lines": []}

        _fake_en_ocr.idx = -1
        scanner.ocr = SimpleNamespace(recognize_array=_fake_ja_ocr)
        scanner.reco.en_ocr = SimpleNamespace(recognize_array=_fake_en_ocr)

        res = scanner.parse_detail_frames([fr0, fr1], plan="logic")
        assert res.p_idol_id == 163
        assert res.recommended_effect == "goodImpressionTurns"
        assert res.idol_stats == {"vocal": 2393, "dance": 1108, "visual": 1644}
        assert [p["item_id"] for p in res.p_items] == [509]
        assert res.deck_card_ids.count(129) == 1
        assert res.deck_card_ids.count(137) == 1
        assert res.deck_card_ids.count(23) == 2
        assert res.card_memory_hooks[129]["force_initial_hand"] is True
        assert res.card_memory_hooks[129]["exiles_trouble_on_play"] is True
    finally:
        scanner.close()
        db.close()


def test_local_agent_sp_priority_recommended_effect_and_no_new_card_shop():
    db = GakumasDB("gakumas_library.db")
    try:
        tracker = GameStateTracker(plan="logic", p_idol_id=163)
        tracker.recommended_effect = "goodImpressionTurns"
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        agent.bind_tracker(tracker)

        # 1. Produce schedule: SP course always beats higher/normal courses
        sched_obs = ScreenObservation(
            scene_type="produce_schedule",
            plan="logic",
            scenario_id="hif_honsen",
            stamina=24,
            max_stamina=44,
            idol_stats={"vocal": 2393, "dance": 1108, "visual": 1644},
        )
        sched_acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="选择: 追い込みレッスン",
                bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10),
                payload={"name": "追い込みレッスン"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="选择: Da.SPレッスン",
                bbox=BoundingBox(xmin=10, ymin=0, xmax=20, ymax=10),
                payload={"name": "Da.SPレッスン", "is_sp": True},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="选择: 休む",
                bbox=BoundingBox(xmin=20, ymin=0, xmax=30, ymax=10),
                payload={"name": "休む"},
            ),
        ]
        dec_sched = agent.decide("", sched_acts, obs=sched_obs)
        assert dec_sched.choice_id == 1

        # 2. Card reward: prefers `goodImpressionTurns` card (#299 オトメゴコロ+) over motivation-only & avoids #23 眠気
        reward_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
        )
        reward_acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="选择: 眠気",
                bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10),
                payload={"name": "眠気", "card_id": 23},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="选择: 深呼吸 (やる気のみ)",
                bbox=BoundingBox(xmin=10, ymin=0, xmax=20, ymax=10),
                payload={"name": "深呼吸", "card_id": 49},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="选择: オトメゴコロ+",
                bbox=BoundingBox(xmin=20, ymin=0, xmax=30, ymax=10),
                payload={"name": "オトメゴコロ+", "card_id": 299},
            ),
        ]
        dec_rew = agent.decide("", reward_acts, obs=reward_obs, p_idol_id=163)
        assert dec_rew.choice_id == 2

        # 3. Shop: NEVER buys skill cards (`スキルカード:0` slot or `交換する` with card_id), picks 強化/削除/ドリンク/終了
        shop_obs = ScreenObservation(
            scene_type="shop",
            plan="logic",
            scenario_id="hif_honsen",
            p_points=250,
            drinks=[RecognizedDrink(drink_id=13, name="パワフル漢方ドリンク", bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10))],
        )
        shop_acts_card_only = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="选择: 【确认购买】交換する《オトメゴコロ+》",
                bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10),
                payload={"name": "交換する", "card_id": 299, "price": 80},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="选择: 切换预览 スキルカード #1 (50P)",
                bbox=BoundingBox(xmin=10, ymin=0, xmax=20, ymax=10),
                payload={"name": "スキルカード1", "shop_slot_key": "スキルカード:0", "price": 50},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="选择: 終了",
                bbox=BoundingBox(xmin=20, ymin=0, xmax=30, ymax=10),
                payload={"name": "終了"},
            ),
        ]
        dec_shop = agent.decide("", shop_acts_card_only, obs=shop_obs)
        assert dec_shop.choice_id == 2

        # 4. Popup: cancels card exchange confirmation & avoids ジュエル / リタイア
        popup_obs = ScreenObservation(
            scene_type="popup",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="このスキルカードと交換しますか？",
        )
        popup_acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="选择: 決定",
                bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10),
                payload={"name": "決定"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="选择: キャンセル",
                bbox=BoundingBox(xmin=10, ymin=0, xmax=20, ymax=10),
                payload={"name": "キャンセル"},
            ),
        ]
        dec_pop = agent.decide("", popup_acts, obs=popup_obs)
        assert dec_pop.choice_id == 1
    finally:
        db.close()



def test_runner_home_produce_home_stop_and_no_second_entry():
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(db=db)
        ctrl = MockController(db=db, plan="logic")
        runner = GakumasRunner(
            mode="mock",
            controller=ctrl,
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )

        # Feed deterministic observations: home -> produce_schedule -> home
        seq = iter([
            ScreenObservation(
                scene_type="home",
                plan="logic",
                scenario_id="hif_honsen",
                options=[{"choice_id": 0, "name": "プロデュース", "title": "プロデュース", "bbox": {"xmin": 210, "ymin": 1272, "xmax": 510, "ymax": 1368}}],
            ),
            ScreenObservation(
                scene_type="produce_schedule",
                plan="logic",
                scenario_id="hif_honsen",
                stamina=40,
                max_stamina=44,
                options=[{"choice_id": 0, "name": "Vo.SPレッスン", "title": "Vo.SPレッスン", "is_sp": True, "bbox": {"xmin": 50, "ymin": 1000, "xmax": 200, "ymax": 1150}}],
            ),
            ScreenObservation(
                scene_type="home",
                plan="logic",
                scenario_id="hif_honsen",
                options=[{"choice_id": 0, "name": "プロデュース", "title": "プロデュース", "bbox": {"xmin": 210, "ymin": 1272, "xmax": 510, "ymax": 1368}}],
            ),
        ])
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: next(seq))

        results = runner.run_loop(max_steps=10, delay_between_steps=0.0, stop_on_home_return=True)
        assert [r.scene_type for r in results] == ["home", "produce_schedule", "home"]
        # First Home clicked プロデュース, Schedule clicked Vo.SPレッスン, second Home did NOT click!
        assert results[0].click_coords is not None
        assert results[1].click_coords is not None
        assert results[2].click_coords is None
    finally:
        db.close()



def test_live_battle_planner_dynamic_r1_vs_r2_turns():
    db = GakumasDB("gakumas_library.db")
    try:
        tracker = GameStateTracker(plan="logic", p_idol_id=163)
        tracker.set_initial_loadout(
            p_idol_id=163,
            plan="logic",
            recommended_effect="goodImpressionTurns",
            deck_ids=[920, 922, 159, 129, 127, 137, 299, 521, 179, 23],
            p_items=[{"item_id": 509, "canonical_name": "星々を統べる輝き+", "mode": "stage"}],
            card_memory_hooks={127: {"force_initial_hand": True}, 129: {"force_initial_hand": True}},
            db=db,
        )

        # Verify build_live_battle_config scales schedule to 9T vs 12T
        cfg9 = build_live_battle_config(
            db=db,
            deck_card_ids=tracker.produce_deck,
            equipped_p_items=tracker.equipped_p_items,
            total_turns=9,
        )
        cfg12 = build_live_battle_config(
            db=db,
            deck_card_ids=tracker.produce_deck,
            equipped_p_items=tracker.equipped_p_items,
            total_turns=12,
        )
        assert cfg9.total_turns == 9 and len(cfg9.turn_schedule) == 9
        assert cfg12.total_turns == 12 and len(cfg12.turn_schedule) == 12

        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_hidden_dim=32, ppo_episodes=32, ppo_episodes_per_update=16, ismcts_simulations=6),
        )
        pb = PromptBuilder(db)

        # Round 1: 9 turns, even transient turns_remaining=0 must be safe later
        obs_r1 = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=7,
            turn=1,
            turns_remaining=9,
            stamina=40,
            max_stamina=44,
            idol_stats={"vocal": 2393, "dance": 1108, "visual": 1644},
            buffs={"goodImpressionTurns": 6, "motivation": 6},
            cards=[
                RecognizedCard(card_id=129, name="ワクワクが止まらない+", upgraded=True, bbox=BoundingBox(xmin=100, ymin=1200, xmax=220, ymax=1450)),
                RecognizedCard(card_id=299, name="オトメゴコロ+", upgraded=True, bbox=BoundingBox(xmin=240, ymin=1200, xmax=360, ymax=1450)),
            ],
            drinks=[RecognizedDrink(drink_id=13, name="パワフル漢方ドリンク", bbox=BoundingBox(xmin=40, ymin=1480, xmax=110, ymax=1550))],
        )
        acts_r1 = pb.generate_legal_actions_and_update_obs(obs_r1, tracker)
        cid_r1, reason_r1, _ = planner.decide_battle_action(obs_r1, acts_r1, tracker=tracker)
        assert cid_r1 in [a.choice_id for a in acts_r1]
        assert "PPO+ISMCTS" in reason_r1
        sig_r1 = planner._active_signature

        # Simulate +1 extra turn (#171 私がスター+) mid-battle: must NOT retrain a brand new model!
        obs_r1_plus = obs_r1.model_copy(update={"turns_remaining": 10})
        cid_r1_p, _, _ = planner.decide_battle_action(obs_r1_plus, acts_r1, tracker=tracker)
        assert cid_r1_p in [a.choice_id for a in acts_r1]
        assert planner._active_signature == sig_r1

        # Transient turns_remaining=0 frame safety
        obs_zero = obs_r1.model_copy(update={"turns_remaining": 0})
        cid_zero, _, _ = planner.decide_battle_action(obs_zero, acts_r1, tracker=tracker)
        assert cid_zero in [a.choice_id for a in acts_r1]

        # Round 2 (Week 9, 12 turns): MUST detect new stage & retrain 12T battle model
        obs_r2 = obs_r1.model_copy(update={"week": 9, "turns_remaining": 12, "stamina": 33})
        acts_r2 = pb.generate_legal_actions_and_update_obs(obs_r2, tracker)
        cid_r2, _, _ = planner.decide_battle_action(obs_r2, acts_r2, tracker=tracker)
        assert cid_r2 in [a.choice_id for a in acts_r2]
        assert planner._battle_initial_turns == 12
        assert planner._active_signature != sig_r1
    finally:
        db.close()



def test_produce_event_tooltip_not_misclassified_as_card_reward():
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
    try:
        frame = np.full((1600, 720, 3), 235, dtype=np.uint8)
        # Highlighted pink choice pill around y=975..1055
        frame[975:1055, 60:660] = (230, 80, 140)
        lines = [
            {"text": "授業", "bbox": [45, 25, 110, 55]},
            {"text": "審査基準", "bbox": [520, 20, 640, 48]},
            {"text": "トラブルカード以外のスキルカードを選択して異なる好印象関係のスキルカードにセレクトチェンジ", "bbox": [110, 790, 630, 875]},
            {"text": "飲食物のチェック", "bbox": [150, 1000, 390, 1036]},
            {"text": "基礎の確認", "bbox": [150, 1098, 340, 1134]},
        ]
        rec.ocr = SimpleNamespace(
            recognize_array=lambda arr, *a, **kw: {"text": "", "lines": lines if arr.shape[:2] == (1600, 720) else []}
        )
        rec.en_ocr = SimpleNamespace(recognize_array=lambda _arr, *a, **kw: {"text": "", "lines": []})

        obs = rec.recognize(frame)
        assert obs.scene_type == "produce_event"
        names = [o.get("name") for o in obs.options]
        assert any("飲食物のチェック" in n for n in names)

        # Drink reward description containing "スキルカード使用数追加+1" must stay `card_reward`, NOT `battle_turn`
        drink_lines = [
            {"text": "差し入れ", "bbox": [70, 40, 150, 60]},
            {"text": "審査基準", "bbox": [550, 30, 630, 50]},
            {"text": "初星ホエイプロテイン", "bbox": [210, 660, 505, 690]},
            {"text": "スキルカード使用数追加+1", "bbox": [125, 718, 458, 742]},
            {"text": "受け取る", "bbox": [305, 1242, 413, 1268]},
        ]
        rec.ocr = SimpleNamespace(
            recognize_array=lambda arr, *a, **kw: {"text": "", "lines": drink_lines if arr.shape[:2] == (1600, 720) else []}
        )
        obs2 = rec.recognize(np.full((1600, 720, 3), 235, dtype=np.uint8))
        assert obs2.scene_type == "card_reward"
    finally:
        db.close()



def test_shop_repeated_upgrade_advances_to_drink_or_exit_and_mid_battle_popup_keeps_state():
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        shop_obs = ScreenObservation(scene_type="shop", plan="logic", scenario_id="hif_honsen", p_points=300)
        shop_acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="选择: 強化",
                bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10),
                payload={"name": "強化", "shop_slot_key": "強化", "price": 100},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="选择: 切换预览 Pドリンク #1",
                bbox=BoundingBox(xmin=10, ymin=0, xmax=20, ymax=10),
                payload={"name": "Pドリンク1", "shop_slot_key": "Pドリンク:0", "price": 50},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="选择: 終了",
                bbox=BoundingBox(xmin=20, ymin=0, xmax=30, ymax=10),
                payload={"name": "終了"},
            ),
        ]
        # Step 1 tries 強化; when P-points unchanged on next step, 強化 is skipped and advances to Pドリンク1
        d1 = agent.decide("", shop_acts, obs=shop_obs)
        assert d1.choice_id == 0
        d2 = agent.decide("", shop_acts, obs=shop_obs)
        assert d2.choice_id == 1

        # Mid-battle SKIP confirm popup with 'はい' must keep tracker.in_battle True & not call reset_battle()
        reset_calls = []
        agent.reset_battle = lambda: reset_calls.append(1)  # type: ignore[method-assign]
        ctrl = MockController(db=db, plan="logic")
        runner = GakumasRunner(mode="mock", controller=ctrl, agent=agent, db=db, plan="logic", scenario_id="hif_honsen", session_mode=True)
        runner.tracker.in_battle = True
        runner.scene_routing_reco = SimpleNamespace(
            recognize=lambda _fr: ScreenObservation(
                scene_type="popup",
                plan="logic",
                scenario_id="hif_honsen",
                raw_text="体力が残っています。ターンをスキップしますか？",
                options=[{"choice_id": 0, "name": "はい", "title": "はい", "bbox": {"xmin": 400, "ymin": 1300, "xmax": 600, "ymax": 1380}}],
            )
        )
        res = runner.run_loop(max_steps=1, delay_between_steps=0.0)
        assert len(res) == 1
        assert runner.tracker.in_battle is True
        assert reset_calls == []
    finally:
        db.close()



def test_stat_cap_3200_ocr_clamp_and_schedule_avoids_capped_attr():
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. OCR >3200 misread (e.g. 4171) is rejected/clamped at 3200
        reco = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        reco._last_stats = {"vocal": 3180, "dance": 1108, "visual": 3200}
        reco.ocr = SimpleNamespace(
            recognize_array=lambda *_a, **_kw: {
                "text": "H.I.F本戦まで 2日 /1000",
                "lines": [{"text": "H.I.F本戦まで 2日 /1000", "bbox": [50, 100, 300, 140]}],
            }
        )
        reco.en_ocr = SimpleNamespace(
            recognize_array=lambda *_a, **_kw: {"text": "4171", "lines": []}
        )
        frame = np.full((1600, 720, 3), 235, dtype=np.uint8)
        meta = reco._parse_week_and_stats(
            [{"text": "/1000", "bbox": [180, 500, 320, 520]}],
            frame,
        )
        assert all(v <= 3200 for v in meta["stats"].values())
        assert meta["stats"]["visual"] == 3200

        # 2. Battle config clamps stats <= 3200
        bcfg = build_live_battle_config(
            db=db,
            deck_card_ids=[920, 922, 137],
            idol_stats={"vocal": 3500, "dance": 1200, "visual": 4171},
            total_turns=9,
        )
        assert bcfg.idol_stats["vocal"] == 3200
        assert bcfg.idol_stats["visual"] == 3200

        # 3. Schedule avoids Vo=3200 capped SP/lesson when uncapped Vi/Da alternatives exist
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        obs_capped = ScreenObservation(
            scene_type="produce_schedule",
            plan="logic",
            scenario_id="hif_honsen",
            stamina=40,
            max_stamina=44,
            idol_stats={"vocal": 3200, "dance": 1200, "visual": 2800},
        )
        sched_acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="Vo.SP公開レッスン",
                bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10),
                payload={"name": "Vo.SP公開レッスン", "is_sp": True},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="Vi.公開レッスン",
                bbox=BoundingBox(xmin=10, ymin=0, xmax=20, ymax=10),
                payload={"name": "Vi.公開レッスン", "is_sp": False},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="Da.SP公開レッスン",
                bbox=BoundingBox(xmin=20, ymin=0, xmax=30, ymax=10),
                payload={"name": "Da.SP公開レッスン", "is_sp": True},
            ),
        ]
        # Between Vo.SP (capped 3200), Vi (2800), and Da.SP (1200), chooses uncapped Da.SP
        d_sp = agent.decide("", sched_acts, obs=obs_capped)
        assert d_sp.choice_id == 2
        # Without Da.SP, uncapped Vi lesson beats Vo=3200 SP
        d_non_sp = agent.decide("", sched_acts[:2], obs=obs_capped)
        assert d_non_sp.choice_id == 1
    finally:
        db.close()



def test_reward_scans_all_drinks_and_skill_cards_before_claiming():
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        agent.tracker = GameStateTracker(plan="logic", p_idol_id=163)

        # 1. Drink 3-choice picks #25 特製ハツボシエキス (slot 2) over #2 烏龍茶 (slot 1)
        drink_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="Pドリンクを選択",
        )
        drink_acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="烏龍茶",
                bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10),
                payload={"name": "候选饮料1", "title": "候选饮料 1: 《烏龍茶》", "drink_id": 2},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="スキルカード使用後、そのスキルカードがもう1度発動",
                bbox=BoundingBox(xmin=10, ymin=0, xmax=20, ymax=10),
                payload={"name": "候选饮料2", "title": "候选饮料 2: 《特製ハツボシエキス》", "drink_id": 25},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="ホットコーヒー",
                bbox=BoundingBox(xmin=20, ymin=0, xmax=30, ymax=10),
                payload={"name": "候选饮料3", "title": "候选饮料 3: 《ホットコーヒー》", "drink_id": 6},
            ),
        ]
        d_drink = agent.decide("", drink_acts, obs=drink_obs)
        assert d_drink.choice_id == 1

        # 2. Skill card 3-choice scans #1 -> #2 -> #3 and then switches back to highest-scoring slot #2 and claims!
        def _mk_card_state(active_slot: int, active_cid: int, active_name: str):
            acts = [
                LegalAction(
                    choice_id=0,
                    action_type="pick_option",
                    description=f"确认领取当前选中卡: {active_name}",
                    bbox=BoundingBox(xmin=300, ymin=1400, xmax=500, ymax=1480),
                    payload={"name": "受け取る", "title": f"确认领取当前选中卡: {active_name}", "card_id": active_cid},
                )
            ]
            for s in (0, 1, 2):
                if s != active_slot:
                    acts.append(
                        LegalAction(
                            choice_id=len(acts),
                            action_type="pick_option",
                            description=f"预览第{s + 1}张候选卡",
                            bbox=BoundingBox(xmin=100 + s * 150, ymin=1000, xmax=200 + s * 150, ymax=1100),
                            payload={"name": f"预览第{s + 1}张候选卡", "slot_idx": s},
                        )
                    )
            return acts

        card_obs = ScreenObservation(scene_type="card_reward", plan="logic", scenario_id="hif_honsen", raw_text="スキルカード獲得")
        # Step 1: slot 0 active (#98 アイドル宣言, lower than #920) -> scans slot 1
        r1 = agent.decide("", _mk_card_state(0, 98, "アイドル宣言"), obs=card_obs)
        assert r1.choice_id == 1
        # Step 2: slot 1 active (#920 ありのままのわたし+, best SSR) -> scans unseen slot 2
        r2 = agent.decide("", _mk_card_state(1, 920, "ありのままのわたし+"), obs=card_obs)
        assert r2.choice_id == 2
        # Step 3: slot 2 active (#23 眠気, weak/bad) -> all 3 scanned, switches back to slot 1 (#920)!
        r3 = agent.decide("", _mk_card_state(2, 23, "眠気"), obs=card_obs)
        assert r3.choice_id == 2  # in _mk_card_state(2), preview slot_idx=1 has choice_id=2
        # Step 4: slot 1 active (#920) with all 3 slots already scanned -> claims 受け取る!
        r4 = agent.decide("", _mk_card_state(1, 920, "ありのままのわたし+"), obs=card_obs)
        assert r4.choice_id == 0
    finally:
        db.close()



def test_customize_grid_and_menu_route_to_card_reward_and_execute_or_exit():
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        reco = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )

        # 1. Customize card-grid screen (with "あと2枚" and "(4ターン)") must be card_reward, NOT battle_turn
        grid_frame = np.full((1600, 720, 3), 235, dtype=np.uint8)
        grid_frame[1450:1505, 390:610] = (245, 130, 35)  # orange カスタマイズ button
        grid_lines = [
            {"text": "◆カスタマイズ", "bbox": [21, 40, 230, 72]},
            {"text": "びしっとキメ顔+", "bbox": [196, 112, 404, 137]},
            {"text": "好印象強化+200%(4ターン)", "bbox": [165, 193, 613, 225]},
            {"text": "カスタマイズメニュー", "bbox": [263, 392, 458, 409]},
            {"text": "カスタマイズ可能スキルカードあと2枚", "bbox": [183, 1402, 538, 1421]},
            {"text": "もどる", "bbox": [202, 1467, 280, 1492]},
            {"text": "カスタマイズ", "bbox": [410, 1462, 590, 1497]},
        ]
        reco.ocr = SimpleNamespace(
            recognize_array=lambda arr, *a, **kw: {
                "text": "",
                "lines": (
                    grid_lines
                    if arr.shape[:2] == (1600, 720)
                    else (
                        [{"text": "びしっとキメ顔+", "bbox": [10, 10, 220, 40]}]
                        if 150 <= arr.shape[0] <= 420
                        else []
                    )
                ),
            }
        )
        reco.en_ocr = SimpleNamespace(recognize_array=lambda *_a, **_kw: {"text": "", "lines": []})
        obs_grid = reco.recognize(grid_frame, plan="logic")
        assert obs_grid.scene_type == "card_reward"
        assert any(o.get("name") == "カスタマイズする" for o in obs_grid.options)

        # 2. Customize menu picker (条件変更 / 持続効果+ / 実行する / X終了 / 一覧に戻る) routes to card_reward
        menu_frame = np.full((1600, 720, 3), 220, dtype=np.uint8)
        menu_frame[1210:1290, 80:200] = (200, 90, 220)   # 条件変更 purple tile
        menu_frame[1210:1290, 290:430] = (200, 90, 220)  # 持続効果+ purple tile
        menu_frame[1455:1505, 385:590] = (170, 120, 95)  # dim disabled 実行する
        menu_lines = [
            {"text": "◆カスタマイズ", "bbox": [21, 40, 230, 72]},
            {"text": "びしっとキメ顔+", "bbox": [196, 162, 404, 188]},
            {"text": "メニューを選択してください", "bbox": [231, 577, 514, 598]},
            {"text": "カスタマイズメニュー", "bbox": [244, 1172, 458, 1191]},
            {"text": "あと1回", "bbox": [100, 1200, 177, 1218]},
            {"text": "条件変更", "bbox": [94, 1235, 189, 1258]},
            {"text": "持続効果+", "bbox": [305, 1235, 414, 1258]},
            {"text": "X終了", "bbox": [593, 1355, 696, 1387]},
            {"text": "合計あと1回", "bbox": [445, 1408, 557, 1427]},
            {"text": "一覧に戻る", "bbox": [172, 1467, 308, 1493]},
            {"text": "実行する", "bbox": [410, 1462, 576, 1497]},
        ]
        reco.ocr = SimpleNamespace(
            recognize_array=lambda arr, *a, **kw: {"text": "", "lines": menu_lines if arr.shape[:2] == (1600, 720) else []}
        )
        obs_menu = reco.recognize(menu_frame, plan="logic")
        assert obs_menu.scene_type == "card_reward"
        menu_names = [o.get("name") for o in obs_menu.options]
        assert "条件変更" in menu_names and "持続効果+" in menu_names and "終了" in menu_names

        # Agent clicks both customize tiles, then exits cleanly when 実行する never lights up orange
        menu_acts = [
            LegalAction(
                choice_id=o["choice_id"],
                action_type="pick_option",
                description=o["title"],
                bbox=BoundingBox(**o["bbox"]),
                payload=o,
            )
            for o in obs_menu.options
        ]
        c1 = agent.decide("", menu_acts, obs=obs_menu)
        assert menu_acts[c1.choice_id].payload.get("is_customize_tile") is True
        c2 = agent.decide("", menu_acts, obs=obs_menu)
        assert menu_acts[c2.choice_id].payload.get("is_customize_tile") is True and c2.choice_id != c1.choice_id
        c3 = agent.decide("", menu_acts, obs=obs_menu)
        assert menu_acts[c3.choice_id].payload.get("name") in ("終了", "一覧に戻る")

        # When 実行する is active orange, agent immediately picks カスタマイズ実行
        menu_frame[1455:1505, 385:590] = (245, 130, 35)
        obs_menu_ready = reco.recognize(menu_frame, plan="logic")
        ready_acts = [
            LegalAction(
                choice_id=o["choice_id"],
                action_type="pick_option",
                description=o["title"],
                bbox=BoundingBox(**o["bbox"]),
                payload=o,
            )
            for o in obs_menu_ready.options
        ]
        c_exec = agent.decide("", ready_acts, obs=obs_menu_ready)
        assert ready_acts[c_exec.choice_id].payload.get("name") == "カスタマイズ実行"
    finally:
        db.close()



def test_planner_buff_truncation_playable_reconciliation_and_anti_skip():
    import random
    from demo.sim.battle_simulator import HeadlessBattleSimulator
    from demo.sim.ismcts import ISMCTSAgent, ISMCTSConfig

    db = GakumasDB("gakumas_library.db")
    try:
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_episodes=32, ismcts_simulations=8),
        )
        tr = GameStateTracker(plan="logic", p_idol_id=163)
        tr.init_produce_deck(db=db, plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        pb = PromptBuilder(db)

        # Build live turn observation where #778 びしっとキメ顔+ requires GI>=10, marked playable=True even if OCR misread GI=1
        obs1 = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=7,
            turn=1,
            turns_remaining=9,
            stamina=37,
            max_stamina=44,
            genki=5,
            buffs={"goodImpressionTurns": 14, "motivation": 12},
            cards=[
                RecognizedCard(
                    card_id=171,
                    name="私がスター+",
                    playable=True,
                    cost_type="goodImpressionTurns",
                    cost_amount=2,
                    bbox=BoundingBox(xmin=100, ymin=1100, xmax=260, ymax=1350),
                ),
                RecognizedCard(
                    card_id=778,
                    name="びしっとキメ顔+",
                    playable=True,
                    cost_type="stamina",
                    cost_amount=5,
                    bbox=BoundingBox(xmin=280, ymin=1100, xmax=440, ymax=1350),
                ),
            ],
            options=[{"choice_id": 2, "name": "SKIP", "title": "SKIP"}],
        )
        _, acts1 = pb.build_prompt(obs1, tr)
        cid1, _reason1, _ = planner.decide_battle_action(obs1, acts1, tr)
        assert planner._sim is not None
        assert planner._sim.buffs["goodImpressionTurns"] == 14

        # Next turn OCR truncates GI from 14 -> 1 while #778 is still playable: planner keeps >= 10 and lifts for playable
        obs2 = obs1.model_copy(update={"turn": 2, "turns_remaining": 8, "buffs": {"goodImpressionTurns": 1, "motivation": 1}})
        _, acts2 = pb.build_prompt(obs2, tr)
        # Force mock ISMCTS returning END_TURN so anti-SKIP override is exercised cleanly
        end_aid = planner._codec.end_turn_action_id  # type: ignore[union-attr]
        planner._ismcts.select_action = lambda *_a, **_kw: (end_aid, {"best_q_score": 100.0, "top_actions": []})  # type: ignore[union-attr,method-assign]
        cid2, reason2, _ = planner.decide_battle_action(obs2, acts2, tr)
        chosen2 = next(a for a in acts2 if a.choice_id == cid2)
        assert chosen2.action_type == "play_card"
        assert "Anti-SKIP override" in reason2
        assert planner._sim.buffs["goodImpressionTurns"] >= 10

        # Negative value head prediction in ISMCTS must be clamped >= 0.0
        bcfg = build_live_battle_config(db=db, deck_card_ids=[920, 922, 137, 171], total_turns=9)
        sim = HeadlessBattleSimulator(bcfg, db=db, seed=1, record_logs=False)
        from demo.sim.rl_env import GakumasActionCodec, GakumasFeatureEncoder
        codec = GakumasActionCodec(bcfg)
        enc = GakumasFeatureEncoder(codec)
        neg_net = SimpleNamespace(
            predict_single=lambda obs_arr, mask_arr: (np.where(mask_arr, 1.0 / max(1, int(mask_arr.sum())), 0.0), -500.0),
            select_action=lambda obs_arr, mask_arr, _rng, **_kw: (int(np.flatnonzero(mask_arr)[0]), 0.0, -500.0),
        )
        agent_mcts = ISMCTSAgent(codec=codec, encoder=enc, net=neg_net, cfg=ISMCTSConfig(num_simulations=2))
        assert agent_mcts._evaluate_leaf(sim, random.Random(0)) >= 0.0
        assert agent_mcts._rollout_value(sim, random.Random(0)) >= 0.0
    finally:
        db.close()



def test_loadout_retry_empty_scan_guard_mid_battle_turn_end_popup_and_live_library_sync(tmp_path: Path):
    db = GakumasDB("gakumas_library.db")
    try:
        # 1. scan_via_controller retries long_press when modal failed to open, and never clicks << back when empty
        clicks = []
        long_presses = []
        blank_frame = np.full((1600, 720, 3), 235, dtype=np.uint8)
        ctrl_stub = SimpleNamespace(
            long_press=lambda x, y, duration_ms=950: long_presses.append((x, y, duration_ms)),
            click=lambda x, y: clicks.append((x, y)),
            screencap=lambda: blank_frame,
        )
        scanner = HifLoadoutScanner(db=db)
        scanner.ocr = SimpleNamespace(recognize_array=lambda *_a, **_kw: {"text": "アイドル選択", "lines": []})
        empty_res = scanner.scan_via_controller(ctrl_stub)  # type: ignore[arg-type]
        assert len(long_presses) == 2
        assert clicks == []
        assert empty_res.deck_card_ids == []

        # 2. GakumasRunner refuses to mark initial_loadout_captured=True on empty scan, then updates library JSON on reward add
        lib_path = tmp_path / "live_lib.json"
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
            library_out_path=lib_path,
        )
        ctrl = MockController(db=db, plan="logic")
        runner = GakumasRunner(
            mode="mock",
            controller=ctrl,
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
            loadout_scanner=SimpleNamespace(scan_via_controller=lambda *_a, **_kw: empty_res),
            loadout_library_path=lib_path,
        )
        setup_obs = ScreenObservation(
            scene_type="produce_setup",
            plan="logic",
            scenario_id="hif_honsen",
            options=[
                {"choice_id": 0, "name": "次へ", "title": "次へ", "bbox": {"xmin": 200, "ymin": 1350, "xmax": 500, "ymax": 1440}},
                {"choice_id": 1, "name": "アイドル詳細(長押し)", "title": "アイドル詳細(長押し)", "is_loadout_target": True, "bbox": {"xmin": 180, "ymin": 1100, "xmax": 300, "ymax": 1260}},
            ],
        )
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: setup_obs)
        runner.run_once()
        assert runner.tracker.initial_loadout_captured is False
        assert runner._loadout_scan_attempts == 1
        assert not lib_path.exists()

        # 3. Mid-battle "終了" popup with "ターン終了" keeps tracker.in_battle True, and mid-produce card add updates lib_path
        runner.tracker.set_initial_loadout(p_idol_id=163, plan="logic", recommended_effect="goodImpressionTurns", deck_ids=[920, 922], p_items=[], db=db)
        runner._flush_produce_library()
        runner.tracker.in_battle = True
        turn_end_popup = ScreenObservation(
            scene_type="popup",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="ターン終了しますか？",
            options=[{"choice_id": 0, "name": "終了", "title": "ターン終了", "bbox": {"xmin": 350, "ymin": 1300, "xmax": 580, "ymax": 1380}}],
        )
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: turn_end_popup)
        runner.run_once()
        assert runner.tracker.in_battle is True

        # Add #171 via card_reward 受け取る and verify library JSON immediately reflects updated deck_ids
        runner.tracker.in_battle = False
        reward_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="スキルカード獲得",
            options=[{"choice_id": 0, "name": "受け取る", "title": "確認领取: 私がスター+", "card_id": 171, "bbox": {"xmin": 300, "ymin": 1400, "xmax": 500, "ymax": 1480}}],
        )
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: reward_obs)
        runner.run_once()
        saved_lib = json.loads(lib_path.read_text(encoding="utf-8"))
        assert 171 in saved_lib["deck_ids"]
    finally:
        db.close()



def test_shop_drink_exchange_popup_confirms_without_loop_while_blocking_card_exchange():
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        agent._has_entered_produce = True
        shop_obs = ScreenObservation(
            scene_type="shop",
            plan="logic",
            scenario_id="hif_honsen",
            week=8,
            p_points=250,
            drinks=[],
        )
        drink_shop_acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="交換する《フレッシュビネガー》",
                bbox=BoundingBox(xmin=350, ymin=1300, xmax=580, ymax=1380),
                payload={"name": "交換する", "drink_id": 12, "price": 75, "title": "交換する《フレッシュビネガー》"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="終了",
                bbox=BoundingBox(xmin=580, ymin=1350, xmax=700, ymax=1420),
                payload={"name": "終了", "title": "終了"},
            ),
        ]
        # 1. Selecting shop P-drink exchange flags expected drink confirm popup
        d1 = agent.decide("", drink_shop_acts, obs=shop_obs)
        assert d1.choice_id == 0
        assert agent._shop_expecting_drink_confirm is True

        # 2. Immediate confirmation popup ("交換確認 ... この商品を交換しますか？" without word "ドリンク") confirms!
        confirm_popup_obs = ScreenObservation(
            scene_type="popup",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="交換確認 この商品を交換しますか？",
        )
        popup_acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="キャンセル",
                bbox=BoundingBox(xmin=100, ymin=1200, xmax=300, ymax=1280),
                payload={"name": "キャンセル"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="決定",
                bbox=BoundingBox(xmin=400, ymin=1200, xmax=600, ymax=1280),
                payload={"name": "決定"},
            ),
        ]
        d2 = agent.decide("", popup_acts, obs=confirm_popup_obs)
        assert d2.choice_id == 1
        assert agent._shop_expecting_drink_confirm is False

        # 3. If the same drink_id=12 reappears in shop, do not loop purchasing it infinitely -> pick 終了
        d3 = agent.decide("", drink_shop_acts, obs=shop_obs)
        assert d3.choice_id == 1

        # 4. Unexpected "交換確認" popup without prior drink_exchange still clicks キャンセル
        d4 = agent.decide("", popup_acts, obs=confirm_popup_obs)
        assert d4.choice_id == 0
    finally:
        db.close()



def test_two_step_change_reward_scans_candidates_bounds_cancel_and_anti_stall(tmp_path: Path):
    from demo.run_hif_local import build_local_hif_runner

    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        agent._has_entered_produce = True

        # 1. Page 1 ("チェンジで獲得"): must scan all 3 candidate slots as positive acquisition, NOT is_removal!
        p1_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジで獲得 スキルカードを選んでください",
        )
        p1_acts_s0 = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="受け取る",
                bbox=BoundingBox(xmin=300, ymin=1400, xmax=520, ymax=1480),
                payload={"name": "受け取る", "card_id": 137, "title": "やる気は満点+"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="预览第2张候选卡",
                bbox=BoundingBox(xmin=280, ymin=820, xmax=420, ymax=960),
                payload={"name": "预览第2张候选卡", "slot_idx": 1},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="预览第3张候选卡",
                bbox=BoundingBox(xmin=460, ymin=820, xmax=600, ymax=960),
                payload={"name": "预览第3张候选卡", "slot_idx": 2},
            ),
        ]
        d1 = agent.decide("", p1_acts_s0, obs=p1_obs)
        assert d1.choice_id == 1

        p1_acts_s1 = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="受け取る",
                bbox=BoundingBox(xmin=300, ymin=1400, xmax=520, ymax=1480),
                payload={"name": "受け取る", "card_id": 920, "title": "ありのままのわたし+"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="预览第1张候选卡",
                bbox=BoundingBox(xmin=100, ymin=820, xmax=240, ymax=960),
                payload={"name": "预览第1张候选卡", "slot_idx": 0},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="预览第3张候选卡",
                bbox=BoundingBox(xmin=460, ymin=820, xmax=600, ymax=960),
                payload={"name": "预览第3张候选卡", "slot_idx": 2},
            ),
        ]
        d2 = agent.decide("", p1_acts_s1, obs=p1_obs)
        assert d2.choice_id == 2

        p1_acts_s2 = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="受け取る",
                bbox=BoundingBox(xmin=300, ymin=1400, xmax=520, ymax=1480),
                payload={"name": "受け取る", "card_id": 299, "title": "オトメゴコロ+"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="预览第1张候选卡",
                bbox=BoundingBox(xmin=100, ymin=820, xmax=240, ymax=960),
                payload={"name": "预览第1张候选卡", "slot_idx": 0},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="预览第2张候选卡",
                bbox=BoundingBox(xmin=280, ymin=820, xmax=420, ymax=960),
                payload={"name": "预览第2张候选卡", "slot_idx": 1},
            ),
        ]
        d3 = agent.decide("", p1_acts_s2, obs=p1_obs)
        # Switch back to highest-scoring candidate (#920 at slot 1) or claim cleanly without removal logic
        assert d3.choice_id == 2
        d4 = agent.decide("", p1_acts_s1, obs=p1_obs)
        assert d4.choice_id == 0

        # 2. Standalone Page 2 ("チェンジする"): bounded cancel (at most once), then confirm best swap
        agent._came_from_change_pickup = False
        p2_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジするスキルカード",
        )
        p2_acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="チェンジする",
                bbox=BoundingBox(xmin=350, ymin=1400, xmax=580, ymax=1480),
                payload={"name": "チェンジする", "card_id": 920, "title": "ありのままのわたし+"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="预览第2张待选卡",
                bbox=BoundingBox(xmin=200, ymin=900, xmax=320, ymax=1020),
                payload={"name": "预览第2张待选卡", "slot_idx": 1},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="キャンセル",
                bbox=BoundingBox(xmin=80, ymin=1400, xmax=260, ymax=1480),
                payload={"name": "キャンセル"},
            ),
        ]
        dp2_1 = agent.decide("", p2_acts, obs=p2_obs)
        assert dp2_1.choice_id == 1  # inspects slot #2 first
        dp2_cancel = agent.decide("", p2_acts, obs=p2_obs)
        assert dp2_cancel.choice_id == 2  # cancels once when strong card would be lost
        assert agent._reward_cancel_attempts == 1
        # Re-entering Page 2 must NOT cancel a second time -> confirms swap after inspection
        dp2_re1 = agent.decide("", p2_acts, obs=p2_obs)
        assert dp2_re1.choice_id == 1
        dp2_confirm = agent.decide("", p2_acts, obs=p2_obs)
        assert dp2_confirm.choice_id == 0

        # 3. Runner reward anti-stall (>=14 steps) picks 待选卡 slot instead of キャンセル when no confirm is active
        ctrl = MockController(db=db, plan="logic")
        runner = GakumasRunner(
            mode="mock",
            controller=ctrl,
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        p2_no_confirm_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジするスキルカード",
            options=[
                {"choice_id": 0, "name": "待选卡 1", "title": "待选卡 1", "slot_idx": 0, "bbox": {"xmin": 80, "ymin": 900, "xmax": 200, "ymax": 1020}},
                {"choice_id": 1, "name": "キャンセル", "title": "キャンセル", "bbox": {"xmin": 80, "ymin": 1400, "xmax": 260, "ymax": 1480}},
            ],
        )
        runner._reward_step_count = 19  # increments to 20 on run_once()
        runner.agent = SimpleNamespace(
            decide=lambda *a, **kw: DecisionResult(choice_id=1, reason="force_cancel", source="stub"),
            export_session=lambda: {},
            load_session=lambda _s: True,
            reset_session=lambda: None,
        )
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: p2_no_confirm_obs)
        res = runner.run_once()
        assert res.action_taken is not None
        assert res.action_taken.payload.get("name") == "待选卡 1"
        assert res.decision is not None and "Reward anti-stall cap" in res.decision.reason

        # 4. build_local_hif_runner(..., initial_library=...) seeds initial_loadout_captured + deck/P-items & flushes library_out
        seed_lib = tmp_path / "seed_lib.json"
        out_lib = tmp_path / "out_lib.json"
        seed_lib.write_text(
            json.dumps(
                {
                    "p_idol_id": 163,
                    "plan": "logic",
                    "recommended_effect": "goodImpressionTurns",
                    "deck_ids": [920, 922, 127, 129, 23],
                    "p_items": [{"item_id": 509, "canonical_name": "星々を統べる輝き+", "mode": "stage"}],
                    "card_memory_hooks": {"127": {"force_initial_hand": True}},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        seeded_runner = build_local_hif_runner(
            plan="sense",
            ppo_episodes=16,
            ismcts_sims=4,
            initial_library=str(seed_lib),
            library_out=str(out_lib),
        )
        try:
            assert seeded_runner.tracker.initial_loadout_captured is True
            assert seeded_runner.tracker.p_idol_id == 163
            assert seeded_runner.tracker.plan == "logic"
            assert seeded_runner.tracker.produce_deck == [920, 922, 127, 129, 23]
            assert len(seeded_runner.tracker.equipped_p_items) == 1
            assert seeded_runner.tracker.card_memory_hooks[127]["force_initial_hand"] is True
            written = json.loads(out_lib.read_text(encoding="utf-8"))
            assert written["deck_ids"] == [920, 922, 127, 129, 23]
        finally:
            if hasattr(seeded_runner.agent, "close"):
                seeded_runner.agent.close()
            if seeded_runner.loadout_scanner is not None and hasattr(seeded_runner.loadout_scanner, "close"):
                seeded_runner.loadout_scanner.close()
            seeded_runner.db.close()
    finally:
        db.close()


def test_projected_vo_cap_headroom_and_no_odekake():
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        # 1. At Vo=2900 (week<=6), regular Vo.公開レッスン (est +280 -> 3180 > 3060 soft cap) loses to uncapped Vi.公開レッスン
        obs_2900 = ScreenObservation(
            scene_type="produce_schedule",
            plan="logic",
            scenario_id="hif_honsen",
            week=5,
            stamina=40,
            max_stamina=44,
            idol_stats={"vocal": 2900, "dance": 1100, "visual": 1800},
        )
        sched_acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="Vo.公開レッスン",
                bbox=BoundingBox(xmin=50, ymin=1000, xmax=220, ymax=1180),
                payload={"name": "Vo.公開レッスン"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="Vi.公開レッスン",
                bbox=BoundingBox(xmin=450, ymin=1000, xmax=620, ymax=1180),
                payload={"name": "Vi.公開レッスン"},
            ),
        ]
        d_2900 = agent.decide("", sched_acts, obs=obs_2900)
        assert d_2900.choice_id == 1

        # 2. At Vo=3120, even Vo.SPレッスン loses to uncapped Vi normal lesson
        obs_3120 = obs_2900.model_copy(update={"idol_stats": {"vocal": 3120, "dance": 1100, "visual": 1800}})
        sp_vs_norm = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="Vo.SP公開レッスン",
                bbox=BoundingBox(xmin=50, ymin=1000, xmax=220, ymax=1180),
                payload={"name": "Vo.SP公開レッスン", "is_sp": True},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="Vi.公開レッスン",
                bbox=BoundingBox(xmin=450, ymin=1000, xmax=620, ymax=1180),
                payload={"name": "Vi.公開レッスン"},
            ),
        ]
        d_3120 = agent.decide("", sp_vs_norm, obs=obs_3120)
        assert d_3120.choice_id == 1

        # 3. おでかけ strictly loses to 差し入れ / 相談 / 特別指導 / 休む
        obs_rest = obs_2900.model_copy(update={"idol_stats": {"vocal": 2400, "dance": 1100, "visual": 1650}})
        out_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="おでかけ (外出)", bbox=BoundingBox(xmin=50, ymin=1000, xmax=200, ymax=1150), payload={"name": "おでかけ"}),
            LegalAction(choice_id=1, action_type="pick_option", description="休む (休息)", bbox=BoundingBox(xmin=240, ymin=1000, xmax=390, ymax=1150), payload={"name": "休む"}),
            LegalAction(choice_id=2, action_type="pick_option", description="差し入れ", bbox=BoundingBox(xmin=430, ymin=1000, xmax=580, ymax=1150), payload={"name": "差し入れ"}),
        ]
        d_out = agent.decide("", out_acts, obs=obs_rest)
        assert d_out.choice_id == 2
        d_rest_vs_odekake = agent.decide("", out_acts[:2], obs=obs_rest)
        assert d_rest_vs_odekake.choice_id == 1
    finally:
        db.close()


def test_core_card_protection_mandatory_change_no_cancel_and_grid_bottom_scroll():
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        # 1. Core card #299 オトメゴコロ+ has much lower removal preference than unupgraded R (#94 キラメキ) or basic N / trouble #23
        s_core_rm = agent._score_skill_card(
            card_id=299, title="オトメゴコロ+", desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=True
        )
        s_r_rm = agent._score_skill_card(
            card_id=94, title="キラメキ", desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=True
        )
        s_n_rm = agent._score_skill_card(
            card_id=68, title="ひといき", desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=True
        )
        assert s_n_rm >= 450 and s_r_rm >= 450 and s_core_rm <= -2500

        # 2. Confirming Page 1 ("チェンジで獲得") sets _came_from_change_pickup so Page 2 never cancels in a loop
        p1_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジで獲得",
        )
        d1 = agent.decide(
            "",
            [
                LegalAction(choice_id=0, action_type="pick_option", description="次へ", bbox=BoundingBox(xmin=300, ymin=1400, xmax=500, ymax=1480), payload={"name": "次へ", "card_id": 97, "title": "みんな大好き+"}),
            ],
            obs=p1_obs,
        )
        assert d1.choice_id == 0
        assert agent._came_from_change_pickup is True

        p2_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジするスキルカード",
        )
        p2_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="チェンジする", bbox=BoundingBox(xmin=350, ymin=1400, xmax=580, ymax=1480), payload={"name": "チェンジする", "card_id": 299, "title": "オトメゴコロ+"}),
            LegalAction(choice_id=1, action_type="pick_option", description="预览第2张待选卡", bbox=BoundingBox(xmin=200, ymin=900, xmax=320, ymax=1020), payload={"name": "预览第2张待选卡", "slot_idx": 1}),
            LegalAction(choice_id=2, action_type="pick_option", description="キャンセル", bbox=BoundingBox(xmin=80, ymin=1400, xmax=260, ymax=1480), payload={"name": "キャンセル"}),
        ]
        assert agent.decide("", p2_acts, obs=p2_obs).choice_id == 1
        d2_confirm = agent.decide("", p2_acts, obs=p2_obs)
        assert d2_confirm.choice_id == 0  # confirms swap without bouncing キャンセル!

        # 3. Live runner auto-scrolls deck grid to bottom via swipe x2 before acting
        swipes = []
        dummy_frame = np.full((1600, 720, 3), 235, dtype=np.uint8)
        ctrl_stub = SimpleNamespace(
            is_connected=True,
            connect=lambda: True,
            disconnect=lambda: None,
            screencap=lambda: dummy_frame,
            click=lambda _x, _y: True,
            swipe=lambda x1, y1, x2, y2, duration_ms=650: swipes.append((x1, y1, x2, y2, duration_ms)),
        )
        runner = GakumasRunner(
            mode="adb",
            controller=ctrl_stub,  # type: ignore[arg-type]
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        grid_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジするスキルカード",
            options=[
                {"choice_id": 0, "name": "待选卡 1", "title": "待选卡 1", "slot_idx": 0, "bbox": {"xmin": 80, "ymin": 900, "xmax": 200, "ymax": 1020}},
                {"choice_id": 1, "name": "キャンセル", "title": "キャンセル", "bbox": {"xmin": 80, "ymin": 1400, "xmax": 260, "ymax": 1480}},
            ],
        )
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: grid_obs)
        runner.run_once()
        assert len(swipes) == 2
        assert runner._reward_grid_scrolled_bottom is True
    finally:
        db.close()


def test_shop_preserves_p_points_on_red_slots_and_customize_grid_clicks_card():
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. Shop with 200P held + an unaffordable 240P red item must NOT corrupt obs.p_points down to 5!
        reco = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        frame = np.full((1600, 720, 3), 240, dtype=np.uint8)
        # Draw white interior slots so shop grid scanner scans prices; paint red price text in row 0 col 0 (240P)
        frame[int(1600 * 0.485) : int(1600 * 0.510), int(720 * 0.08) : int(720 * 0.27)] = (235, 235, 235)
        frame[int(1600 * 0.490) : int(1600 * 0.505), int(720 * 0.19) : int(720 * 0.26)] = (235, 45, 45)  # red price
        shop_lines = [
            {"text": "相談", "bbox": [30, 30, 120, 70]},
            {"text": "44/44", "bbox": [340, 60, 440, 95]},
            {"text": "200", "bbox": [370, 110, 440, 145]},
            {"text": "終了", "bbox": [580, 1450, 680, 1500]},
            {"text": "強化", "bbox": [360, 1140, 440, 1185]},
            {"text": "100P", "bbox": [365, 1190, 440, 1220]},
        ]
        reco.ocr = SimpleNamespace(recognize_array=lambda arr, *a, **kw: {"text": "", "lines": shop_lines if arr.shape[:2] == (1600, 720) else []})
        reco.en_ocr = SimpleNamespace(recognize_array=lambda arr, *a, **kw: {"text": "200" if arr.shape[1] >= 250 and arr.shape[0] <= 240 else "240", "lines": []})
        obs_shop = reco.recognize(frame, plan="logic")
        assert obs_shop.scene_type == "shop"
        assert obs_shop.p_points == 200

        # 2. Customize grid without active orange カスタマイズする selects a card slot instead of clicking もどる
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        cust_grid_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            week=8,
            raw_text="カスタマイズ可能スキルカード",
        )
        grid_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="待选卡 1", bbox=BoundingBox(xmin=80, ymin=800, xmax=200, ymax=920), payload={"name": "待选卡 1", "slot_idx": 0}),
            LegalAction(choice_id=1, action_type="pick_option", description="待选卡 2", bbox=BoundingBox(xmin=240, ymin=800, xmax=360, ymax=920), payload={"name": "待选卡 2", "slot_idx": 1}),
            LegalAction(choice_id=2, action_type="pick_option", description="もどる", bbox=BoundingBox(xmin=80, ymin=1400, xmax=240, ymax=1480), payload={"name": "もどる"}),
        ]
        d_sel = agent.decide("", grid_acts, obs=cust_grid_obs)
        assert d_sel.choice_id in (0, 1)

        # 3. Once カスタマイズする lights up, agent clicks カスタマイズする
        active_grid_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="カスタマイズする", bbox=BoundingBox(xmin=320, ymin=1400, xmax=580, ymax=1480), payload={"name": "カスタマイズする"}),
            LegalAction(choice_id=1, action_type="pick_option", description="预览第2张待选卡", bbox=BoundingBox(xmin=240, ymin=800, xmax=360, ymax=920), payload={"name": "预览第2张待选卡", "slot_idx": 1}),
            LegalAction(choice_id=2, action_type="pick_option", description="もどる", bbox=BoundingBox(xmin=80, ymin=1400, xmax=240, ymax=1480), payload={"name": "もどる"}),
        ]
        d_open = agent.decide("", active_grid_acts, obs=cust_grid_obs)
        assert d_open.choice_id == 0
    finally:
        db.close()


def test_live_battle_planner_upgrade_turn1_gi_floor_mult_sync_uses_and_cache():
    import random
    from demo.sim.ismcts import ISMCTSAgent, ISMCTSConfig

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. Upgraded defaults match converged hybrid benchmark budget
        default_cfg = LiveBattlePlannerConfig()
        assert default_cfg.ppo_hidden_dim == 128
        assert default_cfg.ppo_episodes == 2000
        assert default_cfg.ismcts_simulations == 64

        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_hidden_dim=64, ppo_episodes=32, ismcts_simulations=4),
        )
        tr = GameStateTracker(plan="logic", p_idol_id=163)
        tr.set_initial_loadout(
            p_idol_id=163,
            plan="logic",
            recommended_effect="goodImpressionTurns",
            deck_ids=[920, 922, 159, 127, 129, 137, 297, 299],
            p_items=[{"item_id": 436, "canonical_name": "ポーチ（黄）", "mode": "produce"}],
            db=db,
        )
        pb = PromptBuilder(db)

        # Turn 1 has #436 + OCR truncated GI=1 -> must floor GI=6 and parse live ポーカル4887 -> ('Vo', 48.87)
        obs1 = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=7,
            turn=1,
            turns_remaining=9,
            stamina=41,
            max_stamina=44,
            genki=10,
            week_activity="H.I.F本戦ラウンド1 (試験 ポーカル4887)",
            buffs={"goodImpressionTurns": 1, "motivation": 6},
            cards=[
                RecognizedCard(card_id=159, name="アイドル魂+", playable=True, bbox=BoundingBox(xmin=80, ymin=1100, xmax=220, ymax=1350)),
                RecognizedCard(card_id=129, name="ワクワクが止まらない+", playable=True, bbox=BoundingBox(xmin=240, ymin=1100, xmax=380, ymax=1350)),
            ],
            drinks=[RecognizedDrink(drink_id=18, name="初星ホエイプロテイン", playable=True, bbox=BoundingBox(xmin=40, ymin=1420, xmax=120, ymax=1500))],
            options=[{"choice_id": 3, "name": "SKIP", "title": "SKIP"}],
        )
        _, acts1 = pb.build_prompt(obs1, tr)
        cid1, _r1, _ = planner.decide_battle_action(obs1, acts1, tr)
        assert planner._sim is not None
        assert planner._battle_cfg is not None
        assert planner._battle_cfg.initial_good_impression >= 6
        assert planner._sim.buffs["goodImpressionTurns"] >= 6
        assert planner._sim.turn_schedule[0].attribute == "Vo"
        assert abs(planner._sim.turn_schedule[0].multiplier - 48.87) < 1e-6

        # Intra-turn drink +1 use must NOT be wiped back to 1 when next frame still reports card_uses_remaining=1
        drink_act = next(a for a in acts1 if a.action_type == "use_drink")
        planner.notify_action_executed(drink_act)
        assert planner._sim.card_uses_remaining == 2
        obs1_after_drink = obs1.model_copy(update={"card_uses_remaining": 1, "drinks": []})
        _, acts1b = pb.build_prompt(obs1_after_drink, tr)
        planner.decide_battle_action(obs1_after_drink, acts1b, tr)
        assert planner._sim.card_uses_remaining >= 2

        # ISMCTS allowed_root_actions restricts selection strictly to allowed canonical IDs
        end_aid = planner._codec.end_turn_action_id  # type: ignore[union-attr]
        chosen_restricted, _ = planner._ismcts.select_action(  # type: ignore[union-attr]
            planner._sim,
            random.Random(7),
            allowed_root_actions={end_aid},
        )
        assert chosen_restricted == end_aid

        # Reset battle (simulating 再挑戦 on identical week/deck/drinks/stats) hits _trained_policy_cache!
        planner.reset_battle()
        train_calls = 0
        from demo.sim.maskable_ppo import MaskablePPOTrainer
        orig_train = MaskablePPOTrainer.train

        def _counting_train(self_tr, *a, **kw):
            nonlocal train_calls
            train_calls += 1
            return orig_train(self_tr, *a, **kw)

        MaskablePPOTrainer.train = _counting_train  # type: ignore[method-assign]
        try:
            planner.decide_battle_action(obs1, acts1, tr)
            assert train_calls == 0
        finally:
            MaskablePPOTrainer.train = orig_train  # type: ignore[method-assign]
    finally:
        db.close()


def test_loadout_detail_modal_back_button_and_recognizer_recovery():
    from PIL import Image
    import cv2
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    # 1. Synthetic 3-icon wide bottom pill -> back ('<<') at ~302/720
    wide = np.full((1600, 720, 3), 245, dtype=np.uint8)
    wide[1495:1570, 27:359] = 150
    bx_w, by_w = HifLoadoutScanner._detect_back_button_xy(wide)
    assert abs(bx_w - 302) <= 2 and abs(by_w - 1532) <= 2

    # 2. Synthetic 1-icon narrow bottom pill -> back ('<<') at ~88/720
    narrow = np.full((1600, 720, 3), 245, dtype=np.uint8)
    narrow[1500:1572, 27:143] = 150
    bx_n, by_n = HifLoadoutScanner._detect_back_button_xy(narrow)
    assert abs(bx_n - 88) <= 2 and abs(by_n - 1532) <= 2

    # 3. Real captured screenshots (if present in scratch)
    scratch = Path(r"C:\Users\josha\.gemini\antigravity\brain\ac3a127f-3564-4ed4-bf41-507d974fd6c6\scratch")
    rec = LiveScreenRecognizer(plan="logic", scenario_id="hif_honsen")
    stuck_p = scratch / "smoke_v4_stuck.png"
    if stuck_p.exists():
        fr1080 = np.array(Image.open(stuck_p))[:, :, :3]
        for label, img in [("1080p", fr1080), ("720p", cv2.resize(fr1080, (720, 1600), interpolation=cv2.INTER_AREA))]:
            obs = rec.recognize(img)
            assert obs.scene_type == "popup", f"{label} expected popup, got {obs.scene_type}"
            assert obs.options and obs.options[0]["name"] == "戻る (<<)"
            bb = obs.options[0]["bbox"]
            cx = (bb["xmin"] + bb["xmax"]) / 2.0
            assert abs(cx / img.shape[1] - 302.0 / 720.0) < 0.03, f"{label} cx ratio={cx / img.shape[1]}"

    grid_p = scratch / "lp_grid.png"
    if grid_p.exists():
        fr_grid = np.array(Image.open(grid_p))[:, :, :3]
        obs_g = rec.recognize(fr_grid)
        assert obs_g.scene_type == "popup"
        assert obs_g.options and obs_g.options[0]["name"] == "戻る (<<)"
        bb_g = obs_g.options[0]["bbox"]
        cx_g = (bb_g["xmin"] + bb_g["xmax"]) / 2.0
        assert abs(cx_g / fr_grid.shape[1] - 88.0 / 720.0) < 0.03

    # 4. Stub controller flow: smoke_v4_stuck (grid) -> setup_lp_toggle (list) -> blank exit
    toggle_p = scratch / "setup_lp_toggle.png"
    if stuck_p.exists() and toggle_p.exists():
        fr_stuck = cv2.resize(np.array(Image.open(stuck_p))[:, :, :3], (720, 1600), interpolation=cv2.INTER_AREA)
        fr_toggle = np.array(Image.open(toggle_p))[:, :, :3]
        fr_exit = np.full((1600, 720, 3), 245, dtype=np.uint8)

        class _StubScanCtrl:
            def __init__(self):
                self.clicks: list[tuple[int, int]] = []
                self.state = "grid"

            def long_press(self, x: int, y: int, duration_ms: int = 800) -> bool:
                self.state = "grid"
                return True

            def click(self, x: int, y: int) -> bool:
                self.clicks.append((x, y))
                if abs(x - 648) <= 25 and abs(y - 1312) <= 25:
                    self.state = "list"
                elif abs(x - 302) <= 25 and abs(y - 1532) <= 25:
                    self.state = "exit"
                return True

            def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> bool:
                return True

            def screencap(self) -> np.ndarray:
                if self.state == "list":
                    return fr_toggle
                if self.state == "exit":
                    return fr_exit
                return fr_stuck

        scanner = HifLoadoutScanner()
        try:
            ctrl = _StubScanCtrl()
            scan_res = scanner.scan_via_controller(ctrl, open_xy=(240, 1185), open_via_long_press=True, plan="logic", max_scrolls=3)
            assert any(abs(cx - 648) <= 5 and abs(cy - 1312) <= 5 for cx, cy in ctrl.clicks)
            assert any(abs(bx - 302) <= 5 and abs(by - 1532) <= 5 for bx, by in ctrl.clicks)
            assert scan_res.p_idol_id == 163
            assert len(scan_res.p_items) >= 3
        finally:
            scanner.close()


def test_change_grid_excludes_dark_trouble_cards_and_tracks_visited_slots():
    from PIL import Image
    import cv2
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    scratch = Path(r"C:\Users\josha\.gemini\antigravity\brain\ac3a127f-3564-4ed4-bf41-507d974fd6c6\scratch")
    change_p = scratch / "smoke_v4c_change.png"
    if change_p.exists():
        fr1080 = np.array(Image.open(change_p))[:, :, :3]
        for label, img in [("1080p", fr1080), ("720p", cv2.resize(fr1080, (720, 1600), interpolation=cv2.INTER_AREA))]:
            rec = LiveScreenRecognizer(plan="logic", scenario_id="hif_honsen")
            obs = rec.recognize(img)
            assert obs.scene_type == "card_reward", f"{label} expected card_reward, got {obs.scene_type}"
            card_slots = [
                o for o in obs.options
                if "换出候选卡" in str(o.get("name") or "") or "待选卡" in str(o.get("name") or "") or "预览第" in str(o.get("name") or "")
            ]
            assert len(card_slots) >= 4, f"{label} expected selectable card slots, got {len(card_slots)}"
            for slot in card_slots:
                bb = slot["bbox"]
                assert bb["ymax"] < int(img.shape[0] * 0.845), f"{label} leaked darkened Trouble row: {slot}"

    # Also verify LocalHybridDecisionAgent rotates through unvisited grid slots when claim_act is None
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジするスキルカードを選択してください",
        )
        acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="待换出卡 1 (暗灰基本卡)",
                bbox=BoundingBox(xmin=80, ymin=900, xmax=200, ymax=1020),
                payload={"name": "换出候选卡1", "slot_idx": 0, "title": "待换出卡 1 (暗灰)"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="待换出卡 2 (暗灰基本卡)",
                bbox=BoundingBox(xmin=220, ymin=900, xmax=340, ymax=1020),
                payload={"name": "换出候选卡2", "slot_idx": 1, "title": "待换出卡 2 (暗灰)"},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="キャンセル",
                bbox=BoundingBox(xmin=80, ymin=1400, xmax=260, ymax=1480),
                payload={"name": "キャンセル"},
            ),
        ]
        d1 = agent.decide("", acts, obs=obs)
        d2 = agent.decide("", acts, obs=obs)
        assert d1.choice_id != d2.choice_id
        assert {d1.choice_id, d2.choice_id} == {0, 1}
    finally:
        db.close()


def test_drink_scoring_without_id_and_single_drink_acquisition_banner():
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        score = agent._score_drink(
            drink_id=None,
            title="厳選初星ティー",
            desc="以降、ターン終了時、好印象+1",
            target_effect="goodImpressionTurns",
            plan="logic",
        )
        assert score > 1000

        obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="厳選初星ティー 受け取る",
        )
        acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="确认领取当前选中饮料: 厳選初星ティー",
                bbox=BoundingBox(xmin=230, ymin=1210, xmax=490, ymax=1285),
                payload={"name": "受け取る", "drink_id": 21, "title": "确认领取当前选中饮料: 厳選初星ティー"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="预览第1瓶候选饮料",
                bbox=BoundingBox(xmin=110, ymin=990, xmax=185, ymax=1120),
                payload={"name": "预览候选饮料1", "slot_idx": 0},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="预览第2瓶候选饮料",
                bbox=BoundingBox(xmin=200, ymin=990, xmax=280, ymax=1120),
                payload={"name": "预览候选饮料2", "slot_idx": 1},
            ),
            LegalAction(
                choice_id=3,
                action_type="pick_option",
                description="切换选中第3瓶候选饮料《初星ホエイプロテイン》",
                bbox=BoundingBox(xmin=300, ymin=990, xmax=380, ymax=1120),
                payload={"name": "预览候选饮料3", "slot_idx": 2, "drink_id": 18, "title": "初星ホエイプロテイン"},
            ),
            LegalAction(
                choice_id=4,
                action_type="pick_option",
                description="切换选中第4瓶候选饮料《初星スーパーソーダ》",
                bbox=BoundingBox(xmin=400, ymin=990, xmax=480, ymax=1120),
                payload={"name": "预览候选饮料4", "slot_idx": 3, "drink_id": 26, "title": "初星スーパーソーダ"},
            ),
            LegalAction(
                choice_id=5,
                action_type="pick_option",
                description="预览第5瓶候选饮料",
                bbox=BoundingBox(xmin=500, ymin=990, xmax=580, ymax=1120),
                payload={"name": "预览候选饮料5", "slot_idx": 4},
            ),
        ]
        dec = agent.decide("", acts, obs=obs)
        assert dec.choice_id == 0
    finally:
        db.close()


def test_jugyo_picks_non_highest_stat_while_lesson_picks_highest_and_events_avoid_nemuki():
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        obs = ScreenObservation(
            scene_type="produce_schedule",
            plan="logic",
            scenario_id="hif_honsen",
            week=3,
            stamina=40,
            max_stamina=44,
            idol_stats={"vocal": 2400, "dance": 1100, "visual": 1800},
        )
        # 1. 授業 (flat gain) picks non-highest stat (Vi=1800 over Vo=2400)
        jugyo_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="Vo.授業", bbox=BoundingBox(xmin=50, ymin=1000, xmax=200, ymax=1180), payload={"name": "Vo.授業"}),
            LegalAction(choice_id=1, action_type="pick_option", description="Da.授業", bbox=BoundingBox(xmin=250, ymin=1000, xmax=400, ymax=1180), payload={"name": "Da.授業"}),
            LegalAction(choice_id=2, action_type="pick_option", description="Vi.授業", bbox=BoundingBox(xmin=450, ymin=1000, xmax=600, ymax=1180), payload={"name": "Vi.授業"}),
        ]
        d_jugyo = agent.decide("", jugyo_acts, obs=obs)
        assert d_jugyo.choice_id == 2

        # 2. レッスン & 公開レッスン (% multiplier) pick highest uncapped stat (Vo=2400)
        lesson_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="Vo.レッスン", bbox=BoundingBox(xmin=50, ymin=1000, xmax=200, ymax=1180), payload={"name": "Vo.レッスン"}),
            LegalAction(choice_id=1, action_type="pick_option", description="Da.レッスン", bbox=BoundingBox(xmin=250, ymin=1000, xmax=400, ymax=1180), payload={"name": "Da.レッスン"}),
            LegalAction(choice_id=2, action_type="pick_option", description="Vi.レッスン", bbox=BoundingBox(xmin=450, ymin=1000, xmax=600, ymax=1180), payload={"name": "Vi.レッスン"}),
        ]
        d_lesson = agent.decide("", lesson_acts, obs=obs)
        assert d_lesson.choice_id == 0

        koukai_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="Vo.公開レッスン", bbox=BoundingBox(xmin=50, ymin=1000, xmax=200, ymax=1180), payload={"name": "Vo.公開レッスン"}),
            LegalAction(choice_id=1, action_type="pick_option", description="Vi.公開レッスン", bbox=BoundingBox(xmin=450, ymin=1000, xmax=600, ymax=1180), payload={"name": "Vi.公開レッスン"}),
        ]
        d_koukai = agent.decide("", koukai_acts, obs=obs)
        assert d_koukai.choice_id == 0

        # 3. 授業 event strongly avoids adding トラブル / 眠気 even if it advertises higher stats/effects
        ev_obs = ScreenObservation(
            scene_type="produce_event",
            plan="logic",
            scenario_id="hif_honsen",
            week=3,
            week_activity="授業",
            raw_text="授業",
            idol_stats={"vocal": 2400, "dance": 1100, "visual": 1800},
        )
        ev_acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="Vocal(Vo)属性强化 | ⚠️会加入『トラブル』干扰卡 | 好印象+5",
                bbox=BoundingBox(xmin=60, ymin=900, xmax=660, ymax=980),
                payload={"name": "パラメータ+80 (トラブル獲得)"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="Visual(Vi)属性强化 | 无体力消耗 (免费) (授業选项)",
                bbox=BoundingBox(xmin=60, ymin=1010, xmax=660, ymax=1090),
                payload={"name": "パラメータ+40"},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="Dance(Da)属性强化 | 消耗体力 -8 (眠気追加)",
                bbox=BoundingBox(xmin=60, ymin=1120, xmax=660, ymax=1200),
                payload={"name": "スキルカード獲得 眠気"},
            ),
        ]
        d_ev = agent.decide("", ev_acts, obs=ev_obs)
        assert d_ev.choice_id == 1
    finally:
        db.close()


def test_extra_turn_card_protection_and_avoid_seichoutsu():
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        # 1. Extra-turn cards (#170/#171 私がスター) have top pick value and ultra-low removal score
        for star_id, star_name in ((170, "私がスター"), (171, "私がスター+")):
            pick_s = agent._score_skill_card(
                card_id=star_id, title=star_name, desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=False
            )
            rem_s = agent._score_skill_card(
                card_id=star_id, title=star_name, desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=True
            )
            assert pick_s >= 4500
            assert rem_s <= -8000

        rem_299 = agent._score_skill_card(card_id=299, title="オトメゴコロ+", desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=True)
        rem_617 = agent._score_skill_card(card_id=617, title="対象、ロックオン+", desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=True)
        rem_94 = agent._score_skill_card(card_id=94, title="キラメキ", desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=True)
        rem_68 = agent._score_skill_card(card_id=68, title="ひといき", desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=True)
        rem_171 = agent._score_skill_card(card_id=171, title="私がスター+", desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=True)
        assert rem_171 < rem_299 and rem_171 < rem_617 and rem_171 < rem_94 and rem_171 < rem_68

        # 2. 成長痛 (#714/#715) is strongly penalized on pick and willingly removable ahead of core cards
        for st_id, st_name in ((714, "成長痛"), (715, "成長痛+")):
            st_pick = agent._score_skill_card(
                card_id=st_id, title=st_name, desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=False
            )
            st_rem = agent._score_skill_card(
                card_id=st_id, title=st_name, desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=True
            )
            assert st_pick <= -3000
            assert st_rem >= 650

        # Reward comparison: #715 成長痛+ loses to normal logic card #94 キラメキ
        kirameki_pick = agent._score_skill_card(
            card_id=94, title="キラメキ", desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=False
        )
        seicho_pick = agent._score_skill_card(
            card_id=715, title="成長痛+", desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=False
        )
        assert kirameki_pick > seicho_pick
    finally:
        db.close()


def test_w8_interval_shop_and_upgrade_customize_real_screenshots():
    import cv2
    from PIL import Image
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        agent._has_entered_produce = True

        # 1. W8 shop policy: zero skill cards bought, リフレッシュ only when >= price+100 and drink slots open, <= 2x
        w8_obs_rich = ScreenObservation(
            scene_type="shop",
            plan="logic",
            scenario_id="hif_honsen",
            week=8,
            p_points=250,
            drinks=[],
        )
        acts_refresh_vs_drink_upg = [
            LegalAction(choice_id=0, action_type="pick_option", description="スキルカード1 (50P)", bbox=BoundingBox(xmin=57, ymin=632, xmax=194, ymax=771), payload={"name": "スキルカード1", "shop_slot_key": "スキルカード:0", "price": 50}),
            LegalAction(choice_id=1, action_type="pick_option", description="Pドリンク3 (100P)", bbox=BoundingBox(xmin=363, ymin=819, xmax=500, ymax=956), payload={"name": "Pドリンク3", "shop_slot_key": "Pドリンク:2", "price": 100}),
            LegalAction(choice_id=2, action_type="pick_option", description="強化 (100P)", bbox=BoundingBox(xmin=57, ymin=1052, xmax=295, ymax=1129), payload={"name": "強化", "shop_slot_key": "強化", "price": 100}),
            LegalAction(choice_id=3, action_type="pick_option", description="カスタマイズ", bbox=BoundingBox(xmin=316, ymin=1052, xmax=561, ymax=1129), payload={"name": "カスタマイズ", "shop_slot_key": "カスタマイズ"}),
            LegalAction(choice_id=4, action_type="pick_option", description="リフレッシュ (10P)", bbox=BoundingBox(xmin=24, ymin=1204, xmax=207, ymax=1304), payload={"name": "リフレッシュ", "shop_slot_key": "リフレッシュ", "price": 10}),
            LegalAction(choice_id=5, action_type="pick_option", description="終了", bbox=BoundingBox(xmin=563, ymin=1208, xmax=715, ymax=1280), payload={"name": "終了"}),
        ]
        # 強化 beats Pドリンク beats カスタマイズ beats リフレッシュ beats 終了; スキルカード never chosen
        assert agent.decide("", acts_refresh_vs_drink_upg, obs=w8_obs_rich).choice_id == 2
        agent._shop_clicked_slots.clear()
        acts_no_upg = [a for a in acts_refresh_vs_drink_upg if a.choice_id != 2]
        assert agent.decide("", acts_no_upg, obs=w8_obs_rich).choice_id == 1
        agent._shop_clicked_slots.clear()
        acts_cust_only = [a for a in acts_refresh_vs_drink_upg if a.choice_id not in (1, 2)]
        assert agent.decide("", acts_cust_only, obs=w8_obs_rich).choice_id == 3

        # With <70P, カスタマイズ is skipped and 終了 beats リフレッシュ (too few P points)
        w8_obs_poor = w8_obs_rich.model_copy(update={"p_points": 60})
        assert agent.decide("", acts_cust_only, obs=w8_obs_poor).choice_id == 5

        # Earlier shop (week=4): >=110P & open drink slots & only skill cards + refresh + exit remaining -> リフレッシュ (at most twice)
        agent._shop_clicked_slots.clear()
        acts_only_refresh = [acts_refresh_vs_drink_upg[0], acts_refresh_vs_drink_upg[4], acts_refresh_vs_drink_upg[5]]
        w4_obs_rich = w8_obs_rich.model_copy(update={"week": 4})
        assert agent.decide("", acts_only_refresh, obs=w4_obs_rich).choice_id == 4
        assert agent._shop_refresh_count == 1
        assert agent.decide("", acts_only_refresh, obs=w4_obs_rich).choice_id == 4
        assert agent._shop_refresh_count == 2
        assert agent.decide("", acts_only_refresh, obs=w4_obs_rich).choice_id == 5

        # Final W8 shop: continues refreshing (> 2) to spend remaining P-points
        agent._shop_refresh_count = 0
        agent._shop_clicked_slots.clear()
        assert agent.decide("", acts_only_refresh, obs=w8_obs_rich).choice_id == 4
        assert agent.decide("", acts_only_refresh, obs=w8_obs_rich).choice_id == 4
        assert agent.decide("", acts_only_refresh, obs=w8_obs_rich).choice_id == 4

        # 2. Live runner does NOT bottom-scroll 強化 or カスタマイズ grids
        swipes = []
        dummy_frame = np.full((1600, 720, 3), 235, dtype=np.uint8)
        ctrl_stub = SimpleNamespace(
            is_connected=True,
            connect=lambda: True,
            disconnect=lambda: None,
            screencap=lambda: dummy_frame,
            click=lambda _x, _y: True,
            swipe=lambda x1, y1, x2, y2, duration_ms=650: swipes.append((x1, y1, x2, y2, duration_ms)),
        )
        runner = GakumasRunner(
            mode="adb",
            controller=ctrl_stub,  # type: ignore[arg-type]
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        for grid_raw in ("強化\nスキルカードが選択されていません", "◆カスタマイズ\nカスタマイズ可能スキルカードあと2枚"):
            grid_obs = ScreenObservation(
                scene_type="card_reward",
                plan="logic",
                scenario_id="hif_honsen",
                raw_text=grid_raw,
                options=[
                    {"choice_id": 0, "name": "待选卡1", "title": "待选卡 1", "slot_idx": 0, "bbox": {"xmin": 79, "ymin": 700, "xmax": 198, "ymax": 822}},
                    {"choice_id": 1, "name": "キャンセル", "title": "キャンセル", "bbox": {"xmin": 138, "ymin": 1453, "xmax": 344, "ymax": 1507}},
                ],
            )
            runner._reward_grid_scrolled_bottom = False
            runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr, _o=grid_obs: _o)
            runner.run_once()
        assert swipes == []

        # 3. Real captured W8 screenshots verification (when present in scratch)
        scratch = Path(r"C:\Users\josha\.gemini\antigravity\brain\ac3a127f-3564-4ed4-bf41-507d974fd6c6\scratch")
        if (scratch / "smoke_v4e_w8_main.png").exists():
            reco = LiveScreenRecognizer(db=db, scenario_id="hif_honsen", plan="logic")

            def _rec(fn: str) -> ScreenObservation:
                reco._shop_visited_slots.clear()
                reco._reward_visited_slots.clear()
                img = cv2.cvtColor(cv2.imread(str(scratch / fn)), cv2.COLOR_BGR2RGB)
                return reco.recognize(img, plan="logic")

            # w8_main
            obs_main = _rec("smoke_v4e_w8_main.png")
            assert obs_main.scene_type == "shop"
            assert obs_main.p_points == 580
            main_by_name = {o["name"]: o for o in obs_main.options}
            assert "スキルカード5" in main_by_name and "スキルカード6" in main_by_name
            assert "Pドリンク1" not in main_by_name and "Pドリンク2" not in main_by_name
            assert main_by_name["Pドリンク3"]["price"] == 100
            assert main_by_name["Pドリンク4"]["price"] == 75
            assert main_by_name["強化"]["price"] == 100 and main_by_name["強化"]["bbox"]["ymin"] < 1090
            assert "カスタマイズ" in main_by_name
            assert main_by_name["リフレッシュ"]["price"] == 10

            # w8_now
            obs_now = _rec("smoke_v4e_w8_now.png")
            now_names = {o["name"] for o in obs_now.options}
            assert "Pドリンク3" not in now_names
            assert "Pドリンク4" in now_names

            # w8_cust_sel_ok (both drinks sold out + 強化 あと0枚 omitted)
            obs_sel_ok = _rec("smoke_v4e_w8_cust_sel_ok.png")
            sel_ok_names = {o["name"] for o in obs_sel_ok.options}
            assert "Pドリンク3" not in sel_ok_names and "Pドリンク4" not in sel_ok_names
            assert "強化" not in sel_ok_names
            assert "カスタマイズ" in sel_ok_names and "リフレッシュ" in sel_ok_names

            # w8_refreshed
            obs_ref = _rec("smoke_v4e_w8_refreshed.png")
            assert obs_ref.p_points == 155
            ref_by_name = {o["name"]: o for o in obs_ref.options}
            assert ref_by_name["Pドリンク3"]["price"] == 75
            assert ref_by_name["Pドリンク4"]["price"] == 100
            assert ref_by_name["強化"]["price"] == 125

            # w8_upg_top
            obs_upg_top = _rec("smoke_v4e_w8_upg_top.png")
            assert obs_upg_top.scene_type == "card_reward"
            upg_slots = [o for o in obs_upg_top.options if "待选卡" in str(o.get("name") or "")]
            assert len(upg_slots) == 1
            assert 680 <= upg_slots[0]["bbox"]["ymin"] <= 720 and 800 <= upg_slots[0]["bbox"]["ymax"] <= 845

            # w8_upg_sel_kirameki
            obs_upg_sel = _rec("smoke_v4e_w8_upg_sel_kirameki.png")
            assert obs_upg_sel.scene_type == "card_reward"
            upg_conf = [o for o in obs_upg_sel.options if o.get("name") == "強化する"]
            assert len(upg_conf) == 1
            assert upg_conf[0].get("card_id") == 94 or "キラメキ" in str(upg_conf[0].get("title") or "")

            # w8_customize
            obs_cust = _rec("smoke_v4e_w8_customize.png")
            assert obs_cust.scene_type == "card_reward"
            cust_slots = [o for o in obs_cust.options if "待选卡" in str(o.get("name") or "")]
            assert len(cust_slots) >= 4
            assert all(s["bbox"]["ymax"] <= 1135 for s in cust_slots)

            # w8_customize_perk & perk2
            obs_p1 = _rec("smoke_v4e_w8_customize_perk.png")
            assert obs_p1.scene_type == "card_reward"
            p1_names = [o["name"] for o in obs_p1.options]
            assert "条件変更" in p1_names and "元気追加" in p1_names
            p1_acts = [
                LegalAction(choice_id=o["choice_id"], action_type="pick_option", description=o["title"], bbox=BoundingBox(**o["bbox"]), payload=o)
                for o in obs_p1.options
            ]
            agent._customize_clicked_tiles.clear()
            d_p1 = agent.decide("", p1_acts, obs=obs_p1)
            assert p1_acts[d_p1.choice_id].payload.get("name") == "条件変更"

            obs_p2 = _rec("smoke_v4e_w8_customize_perk2.png")
            assert obs_p2.scene_type == "card_reward"
            p2_names = [o["name"] for o in obs_p2.options]
            assert "好印象+" in p2_names and "好印象追加" in p2_names
    finally:
        db.close()


def test_round2_drink_gating_empty_hand_score_outlier_pitem429_and_hud_ocr(tmp_path: Path):
    import cv2
    from demo.run_hif_local import build_local_hif_runner
    from demo.sim.live_battle_planner import _parse_live_turn_multiplier
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_hidden_dim=64, ppo_episodes=24, ismcts_simulations=6),
        )
        tr = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tr.set_initial_loadout(
            p_idol_id=163,
            plan="logic",
            recommended_effect="goodImpressionTurns",
            deck_ids=[920, 922, 159, 127, 129, 137, 297, 299, 427, 94],
            p_items=[{"item_id": 509, "canonical_name": "星々を統べる輝き+", "mode": "stage"}],
            db=db,
        )
        tr.idol_stats = {"vocal": 2961, "dance": 1108, "visual": 2123}
        pb = PromptBuilder(db)

        drinks_r2 = [
            RecognizedDrink(drink_id=25, name="特製ハツボシエキス", playable=True, bbox=BoundingBox(xmin=27, ymin=1488, xmax=93, ymax=1554)),
            RecognizedDrink(drink_id=13, name="パワフル漢方ドリンク", playable=True, bbox=BoundingBox(xmin=114, ymin=1488, xmax=180, ymax=1554)),
        ]
        # 1a. Start Round 2 at turns_remaining=12 then step to empty playable hand at turns_remaining=10:
        # #25 must be gated out, sim.hand cleared, and #13 or END_TURN picked
        obs_empty = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=9,
            turn=1,
            turns_remaining=12,
            stamina=22,
            max_stamina=44,
            genki=17,
            score=6580,
            buffs={"goodImpressionTurns": 18, "motivation": 14},
            cards=[],
            drinks=drinks_r2,
            options=[{"choice_id": 0, "name": "SKIP", "title": "SKIP"}],
        )
        _, acts_empty = pb.build_prompt(obs_empty, tr)
        cid_empty, _r_empty, _ = planner.decide_battle_action(obs_empty, acts_empty, tr)
        chosen_empty = next(a for a in acts_empty if a.choice_id == cid_empty)
        assert not (chosen_empty.action_type == "use_drink" and (chosen_empty.payload or {}).get("drink_id") == 25)
        assert planner._sim is not None and planner._sim.hand == []
        # P-item #429 H.I.F応援棒（黄） automatically modeled in Round 2 (week>=8 / >=11T)
        assert planner._battle_cfg is not None
        assert any(it.item_id == 429 for it in planner._battle_cfg.p_items)
        # Also verify mid-battle resume at week=9, turns_remaining=10 includes #429
        assert any(
            it.item_id == 429
            for it in build_live_battle_config(
                db=db,
                deck_card_ids=[920, 922, 159],
                equipped_p_items=[{"item_id": 509, "canonical_name": "星々を統べる輝き+", "mode": "stage"}],
                total_turns=10,
                week=9,
            ).p_items
        )
        # Real tracker.idol_stats used even when obs.idol_stats was default low
        assert planner._battle_cfg.idol_stats["vocal"] == 2961

        # 1b. Hand containing ONLY a Mental card (#920): #25 still gated out
        obs_mental = obs_empty.model_copy(
            update={
                "cards": [
                    RecognizedCard(card_id=920, name="ありのままのわたし+", playable=True, card_type="mental", bbox=BoundingBox(xmin=200, ymin=1100, xmax=360, ymax=1350)),
                ]
            }
        )
        _, acts_mental = pb.build_prompt(obs_mental, tr)
        cid_m, _r_m, _ = planner.decide_battle_action(obs_mental, acts_mental, tr)
        chosen_m = next(a for a in acts_mental if a.choice_id == cid_m)
        assert not (chosen_m.action_type == "use_drink" and (chosen_m.payload or {}).get("drink_id") == 25)

        # 1c. Hand with playable non-L Active card (#427): #25 IS permitted in allowed_root
        obs_active = obs_empty.model_copy(
            update={
                "cards": [
                    RecognizedCard(card_id=427, name="キセキの魔法+", playable=True, card_type="active", bbox=BoundingBox(xmin=200, ymin=1100, xmax=360, ymax=1350)),
                ]
            }
        )
        _, acts_active = pb.build_prompt(obs_active, tr)
        seen_roots: list[Optional[set[int]]] = []
        orig_sel = planner._ismcts.select_action  # type: ignore[union-attr]

        def _cap_root(sim_arg, rng_arg, allowed_root_actions=None):
            seen_roots.append(set(allowed_root_actions) if allowed_root_actions is not None else None)
            return orig_sel(sim_arg, rng_arg, allowed_root_actions=allowed_root_actions)

        planner._ismcts.select_action = _cap_root  # type: ignore[union-attr,method-assign]
        try:
            planner.decide_battle_action(obs_active, acts_active, tr)
        finally:
            planner._ismcts.select_action = orig_sel  # type: ignore[union-attr,method-assign]
        d25_aid = planner._codec.drink_id_to_action[25]  # type: ignore[union-attr]
        assert seen_roots and seen_roots[0] is not None and d25_aid in seen_roots[0]

        # 2. Score OCR 10x spike rejection: 62,401 -> ignore 842,471 spike -> accept 87,109
        planner._sim.score = 62_401
        planner._last_trusted_obs_score = 62_401
        obs_ok1 = obs_active.model_copy(update={"turns_remaining": 9, "score": 62_401})
        planner.decide_battle_action(obs_ok1, acts_active, tr)
        obs_spike = obs_active.model_copy(update={"turns_remaining": 8, "score": 842_471})
        planner.decide_battle_action(obs_spike, acts_active, tr)
        assert planner._sim.score < 200_000
        obs_ok2 = obs_active.model_copy(update={"turns_remaining": 7, "score": 87_109})
        planner.decide_battle_action(obs_ok2, acts_active, tr)
        assert planner._sim.score >= 87_109

        # 3. idol_stats persistence in export_produce_library & restoration in build_local_hif_runner, plus 強化する deck update
        lib_p = tmp_path / "r2_lib.json"
        lib_out_p = tmp_path / "r2_lib_out.json"
        exported = tr.export_produce_library(db=db)
        assert exported["idol_stats"]["vocal"] == 2961
        lib_p.write_text(json.dumps(exported, ensure_ascii=False), encoding="utf-8")
        r_seeded = build_local_hif_runner(
            plan="logic",
            ppo_episodes=16,
            ismcts_sims=4,
            initial_library=str(lib_p),
            library_out=str(lib_out_p),
        )
        try:
            assert r_seeded.tracker.idol_stats["vocal"] == 2961
            dummy_r2_fr = np.full((1600, 720, 3), 235, dtype=np.uint8)
            r_seeded.controller = SimpleNamespace(
                is_connected=True,
                connect=lambda: True,
                disconnect=lambda: None,
                screencap=lambda: dummy_r2_fr,
                click=lambda _x, _y: True,
                swipe=lambda *_a, **_kw: None,
            )  # type: ignore[assignment]
            r_seeded.idle_blank_tap_seconds = 0.0
            upg_obs = ScreenObservation(
                scene_type="card_reward",
                plan="logic",
                scenario_id="hif_honsen",
                raw_text="強化確認",
                options=[{"choice_id": 0, "name": "強化する", "title": "確認強化: キラメキ", "card_id": 94, "bbox": {"xmin": 379, "ymin": 1445, "xmax": 623, "ymax": 1515}}],
            )
            r_seeded.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: upg_obs)
            r_seeded.run_once()
            assert 95 in r_seeded.tracker.produce_deck and 94 not in r_seeded.tracker.produce_deck
        finally:
            r_seeded.stop()
            if hasattr(r_seeded.agent, "close"):
                r_seeded.agent.close()
            if r_seeded.loadout_scanner is not None and hasattr(r_seeded.loadout_scanner, "close"):
                r_seeded.loadout_scanner.close()
            r_seeded.db.close()

        # 4. Offline recognition verification on real Round 2 debug screenshots (when present)
        r2_dir = Path(r"C:\Users\josha\.gemini\antigravity\brain\ac3a127f-3564-4ed4-bf41-507d974fd6c6\scratch\smoke_v4_debug")
        if (r2_dir / "battle_W7_rem12_step005.png").exists():
            reco = LiveScreenRecognizer(db=db, scenario_id="hif_honsen", plan="logic")

            def _rec_r2(fn: str) -> ScreenObservation:
                img = cv2.cvtColor(cv2.imread(str(r2_dir / fn)), cv2.COLOR_BGR2RGB)
                return reco.recognize(img, plan="logic")

            o12 = _rec_r2("battle_W7_rem12_step005.png")
            assert o12.week == 9
            assert o12.turns_remaining == 12
            assert o12.buffs.get("goodImpressionTurns") == 12
            assert o12.buffs.get("motivation") == 6
            assert _parse_live_turn_multiplier(o12) == ("Vo", 45.36)
            assert [d.drink_id for d in o12.drinks] == [25, 16, 13]

            o7 = _rec_r2("battle_W7_rem7_step026.png")
            assert o7.score == 85100
            assert _parse_live_turn_multiplier(o7) == ("Da", 20.71)

            o5 = _rec_r2("battle_W7_rem5_step030.png")
            assert o5.score == 158239
            assert _parse_live_turn_multiplier(o5) == ("Vi", 33.14)

            o4 = _rec_r2("battle_W7_rem4_step033.png")
            assert o4.buffs.get("goodImpressionTurns") == 194
            assert o4.buffs.get("motivation") == 11
    finally:
        db.close()


def test_scrolled_change_grid_dynamic_rows_and_antistall_recovery():
    from types import SimpleNamespace
    import cv2
    from demo.maa_bridge.runner import GakumasRunner
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. Real scrolled 1080x2400 deck-grid verification (smoke_v5_stuck.png)
        scratch = Path(r"C:\Users\josha\.gemini\antigravity\brain\ac3a127f-3564-4ed4-bf41-507d974fd6c6\scratch")
        stuck_p = scratch / "smoke_v5_stuck.png"
        if stuck_p.exists():
            img = cv2.cvtColor(cv2.imread(str(stuck_p)), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            reco = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
            obs = reco.recognize(img, plan="logic")
            assert obs.scene_type == "card_reward"
            card_slots = [
                o for o in obs.options
                if "换出候选卡" in str(o.get("name") or "") or "待选卡" in str(o.get("name") or "")
            ]
            assert len(card_slots) >= 4
            valid_bands = [(1064, 1236), (1285, 1455), (1504, 1676), (1725, 1895)]
            for slot in card_slots:
                bb = slot["bbox"]
                cy = (bb["ymin"] + bb["ymax"]) // 2
                assert any(y1 <= cy <= y2 for y1, y2 in valid_bands), f"Center {cy} landed outside real card bands: {slot}"
                assert bb["ymax"] < 1940, f"Selectable slot leaked into dimmed 眠気 row: {slot}"
                assert "暗灰" not in str(slot.get("title") or ""), f"Bright Row 5 SSR/SR card falsely tagged 暗灰: {slot}"

            dark_slots = reco._scan_deck_grid_slots(img, w, h, allow_trouble_dark=True, bottom_first=True)
            assert any(y1 >= 1940 and y2 <= 2172 for _x1, y1, _x2, y2, _sel, _gray in dark_slots)

        # 2. Agent no-selection preview loop recovery (cancel once when allowed; rotate otherwise)
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_hidden_dim=64, ppo_episodes=16, ismcts_simulations=4),
        )
        obs_chg = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジするスキルカードを選択してください",
        )
        acts_chg = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="待换出卡 1 (底部暗灰基本卡区域)",
                bbox=BoundingBox(xmin=119, ymin=1725, xmax=297, ymax=1895),
                payload={"name": "换出候选卡1", "slot_idx": 0, "title": "待换出卡 1 (暗灰)"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="待换出卡 2 (底部暗灰基本卡区域)",
                bbox=BoundingBox(xmin=335, ymin=1725, xmax=513, ymax=1895),
                payload={"name": "换出候选卡2", "slot_idx": 1, "title": "待换出卡 2 (暗灰)"},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="キャンセル",
                bbox=BoundingBox(xmin=137, ymin=2169, xmax=497, ymax=2248),
                payload={"name": "キャンセル"},
            ),
        ]
        # Visit both unvisited dark-gray slots -> next call sees all slots failed & cancels once
        d1 = agent.decide("", acts_chg, obs=obs_chg)
        d2 = agent.decide("", acts_chg, obs=obs_chg)
        assert {d1.choice_id, d2.choice_id} == {0, 1}
        d3 = agent.decide("", acts_chg, obs=obs_chg)
        assert d3.choice_id == 2
        # Once _reward_cancel_attempts >= 1, exhausted slots must rotate rather than stick on slot 0
        agent._reward_visited_grid_slots = {0, 1}
        d4 = agent.decide("", acts_chg, obs=obs_chg)
        d5 = agent.decide("", acts_chg, obs=obs_chg)
        assert {d4.choice_id, d5.choice_id} == {0, 1}

        # 3. Runner Reward anti-stall cap: rotates preview slots at steps 20..21 and forces キャンセル at step 24
        dummy_frame = np.full((2400, 1080, 3), 235, dtype=np.uint8)
        ctrl_stub = SimpleNamespace(
            is_connected=True,
            connect=lambda: True,
            disconnect=lambda: None,
            screencap=lambda: dummy_frame,
            click=lambda _x, _y: True,
            swipe=lambda _x1, _y1, _x2, _y2, duration_ms=650: None,
        )
        runner = GakumasRunner(
            mode="adb",
            controller=ctrl_stub,  # type: ignore[arg-type]
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        stall_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジするスキルカードを選択してください\nスキルカードが選択されていません",
            options=[
                {"choice_id": 0, "name": "换出候选卡1", "slot_idx": 0, "title": "待选卡 1", "bbox": {"xmin": 119, "ymin": 1725, "xmax": 297, "ymax": 1895}},
                {"choice_id": 1, "name": "换出候选卡2", "slot_idx": 1, "title": "待选卡 2", "bbox": {"xmin": 335, "ymin": 1725, "xmax": 513, "ymax": 1895}},
                {"choice_id": 2, "name": "キャンセル", "title": "キャンセル", "bbox": {"xmin": 137, "ymin": 2169, "xmax": 497, "ymax": 2248}},
            ],
        )
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: stall_obs)

        runner._reward_grid_scrolled_bottom = True
        runner._reward_step_count = 19  # becomes 20 inside run_once
        r20 = runner.run_once()
        assert r20.action_taken is not None and r20.action_taken.payload.get("name") == "换出候选卡1"

        runner._reward_grid_scrolled_bottom = True
        runner._reward_step_count = 20  # becomes 21 inside run_once
        r21 = runner.run_once()
        assert r21.action_taken is not None and r21.action_taken.payload.get("name") == "换出候选卡2"

        runner._reward_grid_scrolled_bottom = True
        runner._reward_step_count = 23  # becomes 24 inside run_once
        r24 = runner.run_once()
        assert r24.action_taken is not None and r24.action_taken.payload.get("name") == "キャンセル"
    finally:
        db.close()


def test_free_action_dominance_and_memory_exile_hooks():
    import random
    from demo.sim.battle_simulator import CardInstance, HeadlessBattleSimulator
    from demo.sim.ismcts import ISMCTSAgent, ISMCTSConfig
    from demo.sim.maskable_ppo import MaskablePPOConfig, MaskablePPOTrainer
    from demo.sim.rl_env import GakumasBattleEnv

    db = GakumasDB("gakumas_library.db")
    scanner = HifLoadoutScanner(db=db)
    try:
        # 1. LoadoutScanner must NOT flag #427 キセキの魔法+ as exiles_trouble_on_play,
        #    MUST flag genuine trouble-exile text, and MUST flag memory "+1 card use" on #137
        rng = np.random.default_rng(19)
        doc = np.full((2200, 720, 3), 255, dtype=np.uint8)
        for box_y in (726, 980, 1230, 1480):
            doc[box_y : box_y + 115, 55:170] = rng.integers(30, 140, size=(115, 115, 3), dtype=np.uint8)
            doc[box_y + 10 : box_y + 105, 200:600] = rng.integers(180, 240, size=(95, 400, 3), dtype=np.uint8)
        fr0 = doc[:1600].copy()
        fr1 = doc[420:2020].copy()

        def _fake_ja_ocr(arr: np.ndarray, *a, **kw):
            ah = arr.shape[0]
            if 220 <= ah <= 300:
                return {
                    "text": "SSR ガラクタロード 秦谷 美鈴",
                    "lines": [
                        {"text": "SSR ガラクタロード", "bbox": [180, 20, 460, 50]},
                        {"text": "秦谷 美鈴", "bbox": [180, 55, 360, 85]},
                    ],
                }
            if ah >= 900 and np.array_equal(arr, fr0[int(1600 * 0.254) : int(1600 * 0.858), :]):
                return {
                    "text": "",
                    "lines": [
                        {"text": "Pアイテム", "bbox": [24, 270, 140, 295]},
                        {"text": "星々を統べる輝き+", "bbox": [200, 325, 480, 355]},
                        {"text": "直接効果で好印象が4回増加時", "bbox": [200, 370, 560, 395]},
                        {"text": "スキルカード", "bbox": [24, 525, 160, 550]},
                        {"text": "キセキの魔法+", "bbox": [200, 580, 460, 610]},
                        {"text": "除外以外にあるトラブルカードが2枚以上の場合、好印象の500%分パラメータ上昇", "bbox": [200, 620, 660, 650]},
                    ],
                }
            if ah >= 900:
                return {
                    "text": "",
                    "lines": [
                        {"text": "ワクワクが止まらない+", "bbox": [200, 410, 480, 440]},
                        {"text": "トラブルカードを除外", "bbox": [200, 450, 520, 480]},
                        {"text": "やる気は満点+", "bbox": [200, 660, 450, 690]},
                        {"text": "スキルカード使用数追加+1", "bbox": [200, 700, 520, 730]},
                    ],
                }
            return {"text": "", "lines": []}

        def _fake_en_ocr(_arr: np.ndarray, *a, **kw):
            return {"text": "2600", "lines": []}

        scanner.ocr = SimpleNamespace(recognize_array=_fake_ja_ocr)
        scanner.reco.en_ocr = SimpleNamespace(recognize_array=_fake_en_ocr)
        res = scanner.parse_detail_frames([fr0, fr1], plan="logic")
        assert 427 in res.deck_card_ids
        assert not (res.card_memory_hooks.get(427) or {}).get("exiles_trouble_on_play")
        assert res.card_memory_hooks[129]["exiles_trouble_on_play"] is True
        assert res.card_memory_hooks[137]["free_extra_use_first_play"] is True

        # 2. Build config with duplicate #137 and dirty #427 hook: #427 must be cleaned,
        #    and #137 must fire extra-use ONCE stage-wide per base_id across duplicate copies
        cfg = build_live_battle_config(
            db=db,
            deck_card_ids=[920, 922, 127, 129, 137, 137, 159, 171, 173, 97, 427, 23, 23],
            equipped_p_items=[{"item_id": 509, "canonical_name": "星々を統べる輝き+", "mode": "stage"}],
            total_turns=9,
            buffs={"goodImpressionTurns": 12, "motivation": 6},
            card_memory_hooks={
                137: {"free_extra_use_first_play": True},
                129: {"force_initial_hand": True, "exiles_trouble_on_play": True},
                427: {"exiles_trouble_on_play": True},
            },
            scenario_id="hif_honsen",
        )
        c427_cfgs = [c for c in cfg.deck if c.card_id == 427]
        assert c427_cfgs and not c427_cfgs[0].exiles_trouble_on_play
        c137_cfgs = [c for c in cfg.deck if c.card_id == 137]
        assert len(c137_cfgs) == 2 and all(c.free_extra_use_first_play for c in c137_cfgs)

        sim = HeadlessBattleSimulator(cfg, db=db, seed=101, record_logs=False)
        sim.turn = 2
        sim.turns_remaining = 8
        sim.card_uses_remaining = 1
        sim.stamina = 30
        sim.buffs["goodImpressionTurns"] = 6
        sim.buffs["motivation"] = 6

        c129_obj = next(c for c in (sim.hand + sim.draw_pile) if c.cfg.card_id == 129)
        c137_a = next(c for c in (sim.hand + sim.draw_pile) if c.cfg.card_id == 137)
        c137_b = next(c for c in (sim.hand + sim.draw_pile) if c.cfg.card_id == 137 and c.uid != c137_a.uid)
        c159 = next(c for c in (sim.hand + sim.draw_pile) if c.cfg.card_id == 159)
        c171 = next(c for c in (sim.hand + sim.draw_pile) if c.cfg.card_id == 171)
        c173 = next(c for c in (sim.hand + sim.draw_pile) if c.cfg.card_id == 173)
        c97 = next(c for c in (sim.hand + sim.draw_pile) if c.cfg.card_id == 97)
        c127 = next(c for c in (sim.hand + sim.draw_pile) if c.cfg.card_id == 127)
        sim.hand = [c137_a, c137_b, c159, c171, c173, c97, c127]

        assert sim.is_safe_dominating_free_card(c159) is True
        assert sim.is_safe_dominating_free_card(c171) is True
        assert sim.is_safe_dominating_free_card(c137_a) is True
        # Motivation-costing free-action cards (#97) must NOT be forced without #129 engine or sufficient reserve
        assert sim.is_safe_dominating_free_card(c97) is False
        # #173 has conditional bonus goodImpressionTurns>=10 (unsatisfied at GI=6), so must NOT be forced early
        assert sim.is_safe_dominating_free_card(c173) is False

        # Test #97 dominance on a clone with #129 engine active so `sim` keeps turn==2 and c137_a at index 0
        sim_mot = sim.clone()
        c129_m = next(c for c in sim_mot._all_instances.values() if c.cfg.card_id == 129)
        c97_m = next(c for c in sim_mot.hand if c.cfg.card_id == 97)
        for z in (sim_mot.draw_pile, sim_mot.discard_pile):
            if c129_m in z:
                z.remove(c129_m)
        if c129_m not in sim_mot.hand:
            sim_mot.hand.append(c129_m)
        sim_mot.play_card(sim_mot.hand.index(c129_m))
        sim_mot.card_uses_remaining = 1
        sim_mot.turns_remaining = 8
        sim_mot.buffs["motivation"] = 10
        assert sim_mot.has_motivation_engine() is True
        assert sim_mot.is_safe_dominating_free_card(c97_m) is True
        # Active card (#97) must NOT be forced while #25 double_card_effect_cards > 0
        sim_mot.double_card_effect_cards = 1
        assert sim_mot.is_safe_dominating_free_card(c97_m) is False
        sim_mot.double_card_effect_cards = 0
        # At endgame (turns_remaining <= 2), c97 can still be dumped safely even at low reserve
        sim_mot.turns_remaining = 2
        sim_mot.buffs["motivation"] = 2
        assert sim_mot.is_safe_dominating_free_card(c97_m) is True

        # Playing first #137 keeps card_uses_remaining==1 (same turn) AND marks base_id 136 used for second copy
        sim.play_card(0)
        assert sim.turn == 2 and sim.card_uses_remaining == 1
        assert sim.has_unused_memory_extra_use(c137_b) is False
        assert sim.is_safe_dominating_free_card(c137_b) is False

        # 3. Planner / ISMCTS on T1 [#127, #129, #159] MUST choose #159 (or a drink), NEVER end turn with #127/#129
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_hidden_dim=64, ppo_episodes=64, ismcts_simulations=16),
        )
        tr = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tr.set_initial_loadout(
            p_idol_id=163,
            plan="logic",
            recommended_effect="goodImpressionTurns",
            deck_ids=[920, 922, 127, 129, 137, 159, 171, 299, 23, 23],
            p_items=[{"item_id": 509, "canonical_name": "星々を統べる輝き+", "mode": "stage"}],
            card_memory_hooks={127: {"force_initial_hand": True}, 129: {"force_initial_hand": True, "exiles_trouble_on_play": True}},
            db=db,
        )
        obs_t1 = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=7,
            turn=1,
            turns_remaining=9,
            stamina=44,
            max_stamina=44,
            genki=0,
            score=0,
            buffs={"goodImpressionTurns": 3, "motivation": 3},
            cards=[
                RecognizedCard(card_id=127, name="ファンシーチャーム+", playable=True, card_type="mental", bbox=BoundingBox(xmin=100, ymin=1100, xmax=260, ymax=1350)),
                RecognizedCard(card_id=129, name="ワクワクが止まらない+", playable=True, card_type="mental", bbox=BoundingBox(xmin=280, ymin=1100, xmax=440, ymax=1350)),
                RecognizedCard(card_id=159, name="アイドル魂+", playable=True, card_type="mental", bbox=BoundingBox(xmin=460, ymin=1100, xmax=620, ymax=1350)),
            ],
            drinks=[],
            options=[{"choice_id": 0, "name": "SKIP", "title": "SKIP"}],
        )
        _, acts_t1 = PromptBuilder(db).build_prompt(obs_t1, tr)
        cid_t1, _reason_t1, _ = planner.decide_battle_action(obs_t1, acts_t1, tr)
        chosen_t1 = next(a for a in acts_t1 if a.choice_id == cid_t1)
        assert chosen_t1.action_type == "play_card" and (chosen_t1.payload or {}).get("card_id") == 159

        # 4. ISMCTS prior boosts memory-exile card when troubles are present (#129 gets non-zero exploration/visits)
        env = GakumasBattleEnv(cfg, db=db)
        mcts = ISMCTSAgent(env.codec, env.encoder, net=None, cfg=ISMCTSConfig(num_simulations=24))
        s_ex = HeadlessBattleSimulator(cfg, db=db, seed=33)
        c129_ex = next(c for c in (s_ex.hand + s_ex.draw_pile) if c.cfg.card_id == 129)
        c127_ex = next(c for c in (s_ex.hand + s_ex.draw_pile) if c.cfg.card_id == 127)
        s_ex.hand = [c127_ex, c129_ex]
        _, st_ex = mcts.select_action(s_ex, random.Random(33))
        vis_map = {nm: vis for nm, vis, _q in (st_ex.get("top_actions") or [])}
        assert any("#129" in nm and vis > 0 for nm, vis in vis_map.items())
    finally:
        scanner.close()
        db.close()


def test_change_swap_strategy_stable_grid_and_post_produce_stop():
    import cv2
    from demo.core.idol_guides import get_p_idol_guide
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. _scan_deck_grid_slots keeps spatial order invariant regardless of selected cell
        scratch = Path(r"C:\Users\josha\.gemini\antigravity\brain\ac3a127f-3564-4ed4-bf41-507d974fd6c6\scratch")
        for fn, bottom_first in [("smoke_v4d_selected.png", True), ("smoke_v5_stuck.png", True)]:
            p = scratch / fn
            if p.exists():
                img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                reco = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
                slots = reco._scan_deck_grid_slots(img, w, h, bottom_first=bottom_first)
                ys_xs = [((-y1, x1) if bottom_first else (y1, x1)) for x1, y1, _x2, _y2, _sel, _g in slots]
                assert ys_xs == sorted(ys_xs), f"{fn} slots were reordered instead of staying spatially stable"

        # 2. _score_skill_card removal ranking strictly prefers filler Active R/SR over core Mental / multiplier engines
        tr = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tr.set_initial_loadout(
            p_idol_id=163,
            plan="logic",
            recommended_effect="goodImpressionTurns",
            deck_ids=[920, 922, 521, 617, 171, 137, 137, 98, 94, 44],
            p_items=[],
            card_memory_hooks={137: {"free_extra_use_first_play": True}},
            db=db,
        )
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_hidden_dim=64, ppo_episodes=16, ismcts_simulations=4),
        )
        agent.bind_tracker(tr)
        guide = get_p_idol_guide(163)

        def _rem(cid: int) -> int:
            r = db.get_card_by_id(cid)
            assert r is not None
            return agent._score_skill_card(
                card_id=cid,
                title=r["name"],
                desc=r.get("description") or "",
                target_effect="goodImpressionTurns",
                guide=guide,
                plan="logic",
                for_removal=True,
            )

        assert _rem(44) == 1100
        assert _rem(94) == 900
        assert _rem(96) == 900
        assert _rem(97) == 550
        assert _rem(98) < 0
        assert _rem(44) > _rem(94) >= _rem(96) > _rem(97) > _rem(98)
        assert _rem(44) > _rem(94) > _rem(137) > _rem(521)
        assert _rem(521) <= -5500 and _rem(617) <= -5500 and _rem(171) == -8000

        # 3. Multi-step チェンジ scan across 6 grid slots picks #44 手拍子 and NEVER confirms #521 or #617
        grid_cards = [
            (521, "夢色リップ+"),
            (617, "対象、ロックオン+"),
            (171, "私がスター+"),
            (920, "ありのままのわたし+"),
            (44, "手拍子"),
            (137, "やる気は満点+"),
        ]
        selected_idx = 0
        agent._came_from_change_pickup = True
        agent._clear_reward_scan_state()
        confirmed_cid = None
        for _ in range(12):
            cid_cur, title_cur = grid_cards[selected_idx]
            opts = [
                {
                    "choice_id": 0,
                    "name": "チェンジする",
                    "title": f"确认换出/删除/自定义当前选中卡: {title_cur}",
                    "card_id": cid_cur,
                    "slot_idx": selected_idx,
                    "bbox": {"xmin": 370, "ymin": 1440, "xmax": 630, "ymax": 1520},
                }
            ]
            for si in range(len(grid_cards)):
                if si == selected_idx:
                    continue
                opts.append({
                    "choice_id": len(opts),
                    "name": f"预览第{si + 1}张待选卡",
                    "title": f"切换选中卡库第{si + 1}张卡",
                    "slot_idx": si,
                    "bbox": {"xmin": 100 + si * 90, "ymin": 700, "xmax": 180 + si * 90, "ymax": 820},
                })
            obs_g = ScreenObservation(
                scene_type="card_reward",
                plan="logic",
                scenario_id="hif_honsen",
                raw_text="チェンジするスキルカードを選択してください",
                options=opts,
            )
            acts_g = [
                LegalAction(
                    choice_id=o["choice_id"],
                    action_type="pick_option",
                    description=o["title"],
                    bbox=BoundingBox(**o["bbox"]),
                    payload=o,
                )
                for o in opts
            ]
            dec_g = agent.decide("", acts_g, obs=obs_g, p_idol_id=163)
            chosen_g = acts_g[dec_g.choice_id]
            if (chosen_g.payload or {}).get("name") == "チェンジする":
                confirmed_cid = cid_cur
                break
            selected_idx = int((chosen_g.payload or {})["slot_idx"])
        assert confirmed_cid == 44

        # 4. Runner stays at top on H.I.F チェンジ (0 N cards) and stops cleanly on post-produce produce_setup
        swipes = []
        tr_no_n = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tr_no_n.set_initial_loadout(
            p_idol_id=163,
            plan="logic",
            recommended_effect="goodImpressionTurns",
            deck_ids=[920, 922, 521, 617, 44],
            p_items=[],
            db=db,
        )
        dummy_fr = np.full((1600, 720, 3), 235, dtype=np.uint8)
        ctrl_stub = SimpleNamespace(
            is_connected=True,
            connect=lambda: True,
            disconnect=lambda: None,
            screencap=lambda: dummy_fr,
            click=lambda _x, _y: True,
            swipe=lambda x1, y1, x2, y2, duration_ms=650: swipes.append((x1, y1, x2, y2, duration_ms)),
        )
        runner = GakumasRunner(
            mode="adb",
            controller=ctrl_stub,  # type: ignore[arg-type]
            agent=agent,
            tracker=tr_no_n,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        agent._clear_reward_scan_state()
        seq = iter([
            ScreenObservation(
                scene_type="card_reward",
                plan="logic",
                scenario_id="hif_honsen",
                raw_text="チェンジするスキルカードを選択してください",
                options=[
                    {"choice_id": 0, "name": "チェンジする", "title": "确认换出: 手拍子", "card_id": 44, "slot_idx": 0, "bbox": {"xmin": 370, "ymin": 1440, "xmax": 630, "ymax": 1520}},
                    {"choice_id": 1, "name": "预览第2张待选卡", "title": "待选卡 2", "slot_idx": 1, "bbox": {"xmin": 310, "ymin": 700, "xmax": 460, "ymax": 820}},
                ],
            ),
            ScreenObservation(
                scene_type="produce_setup",
                plan="logic",
                scenario_id="hif_honsen",
                options=[
                    {"choice_id": 0, "name": "本戦", "title": "本戦", "bbox": {"xmin": 100, "ymin": 900, "xmax": 600, "ymax": 1100}},
                ],
            ),
        ])
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: next(seq))
        res_loop = runner.run_loop(max_steps=5, delay_between_steps=0.0, stop_on_home_return=True)
        assert swipes == []
        assert [r.scene_type for r in res_loop] == ["card_reward", "produce_setup"]
        assert res_loop[1].click_coords is None
    finally:
        db.close()


def test_setup_screen_classification_confirm_stats_and_detail_fallback_scan(tmp_path: Path):
    from PIL import Image
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer
    from demo.vision.loadout_scanner import LoadoutScanResult

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. Synthetic 1.アイドル選択 lacking top-left pill ("アイドル選択") must NOT fall back to popup
        reco_syn = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        syn_lines = [
            {"text": "True End条件：達成", "bbox": [430, 540, 690, 568]},
            {"text": "Pアイドル設定", "bbox": [30, 810, 120, 870]},
            {"text": "選抜試験メモリー", "bbox": [460, 715, 680, 745]},
            {"text": "プラン ALL", "bbox": [330, 1045, 455, 1080]},
            {"text": "育成情報", "bbox": [55, 1390, 195, 1428]},
            {"text": "次へ", "bbox": [325, 1388, 415, 1435]},
            {"text": "選抜試験メモリー", "bbox": [520, 1392, 680, 1428]},
            {"text": "特訓段階", "bbox": [495, 1045, 610, 1080]},
            {"text": "ガラクタロード", "bbox": [480, 855, 660, 885]},
            {"text": "秦谷 美鈴", "bbox": [470, 640, 690, 685]},
            {"text": "親愛度 37/37", "bbox": [560, 910, 695, 955]},
        ]
        syn_frame = np.full((1600, 720, 3), 235, dtype=np.uint8)
        reco_syn.ocr = SimpleNamespace(recognize_array=lambda *_a, **_kw: {"text": " ".join(l["text"] for l in syn_lines), "lines": syn_lines})
        obs_syn = reco_syn.recognize(syn_frame)
        assert obs_syn.scene_type == "produce_setup"
        assert obs_syn.week_activity == "1.アイドル選択"
        assert any(o.get("name") == "次へ" for o in obs_syn.options)

        # 2. Scratch setup_2.png & setup_3b.png: accurate classification + unpoisoned right-aligned 2.開始確認 stats
        scratch = Path(r"C:\Users\josha\.gemini\antigravity\brain\ac3a127f-3564-4ed4-bf41-507d974fd6c6\scratch")
        s2_p = scratch / "setup_2.png"
        s3b_p = scratch / "setup_3b.png"
        if s2_p.exists() and s3b_p.exists():
            reco_real = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
            fr2 = np.array(Image.open(s2_p))[:, :, :3]
            fr3b = np.array(Image.open(s3b_p))[:, :, :3]
            obs2 = reco_real.recognize(fr2)
            assert obs2.scene_type == "produce_setup"
            assert "アイドル選択" in obs2.week_activity
            obs3b = reco_real.recognize(fr3b)
            assert obs3b.scene_type == "produce_setup"
            assert "開始確認" in obs3b.week_activity
            assert obs3b.idol_stats == {"vocal": 2393, "dance": 1108, "visual": 1644}
            assert reco_real._last_stats == {"vocal": 130, "dance": 100, "visual": 105}
            assert any(o.get("is_loadout_target") and not o.get("open_via_long_press") and o.get("name") == "詳細" for o in obs3b.options)

        # 3. HifLoadoutScanner must NOT treat "選抜試験メモリー" / "育成情報" on 1.アイドル選択 as modal_opened
        clicks_sc = []
        swipes_sc = []
        lp_sc = []
        ctrl_scan = SimpleNamespace(
            long_press=lambda x, y, duration_ms=950: lp_sc.append((x, y, duration_ms)),
            click=lambda x, y: clicks_sc.append((x, y)),
            swipe=lambda *a, **kw: swipes_sc.append(a),
            screencap=lambda: syn_frame,
        )
        scanner = HifLoadoutScanner(db=db)
        try:
            scanner.ocr = SimpleNamespace(
                recognize_array=lambda *_a, **_kw: {"text": "選抜試験メモリー 育成情報 プラン ALL 次へ", "lines": []}
            )
            res_stuck = scanner.scan_via_controller(ctrl_scan, open_via_long_press=True, plan="logic")  # type: ignore[arg-type]
            assert len(lp_sc) == 2
            assert res_stuck.screenshots_scanned == 0
            assert res_stuck.deck_card_ids == []
            assert clicks_sc == []
            assert swipes_sc == []
        finally:
            scanner.close()

        # 4. Runner: failed long-press on 1.アイドル選択 keeps budget for 2.開始確認 詳細 (even on 2nd detail try)
        lib_path = tmp_path / "detail_fallback_lib.json"
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
            library_out_path=lib_path,
        )
        ctrl = MockController(db=db, plan="logic")
        scan_calls = []

        def _fake_scan(_ctrl, *, open_xy=(0, 0), open_via_long_press=True, plan="logic", **_kw):
            scan_calls.append((open_xy, open_via_long_press))
            if open_via_long_press or len(scan_calls) == 2:
                return LoadoutScanResult(plan=plan, screenshots_scanned=0)
            return LoadoutScanResult(
                p_idol_id=163,
                idol_name="秦谷 美鈴",
                title="ガラクタロード",
                plan="logic",
                recommended_effect="goodImpressionTurns",
                deck_card_ids=[920, 922, 521, 617, 171],
                p_items=[{"item_id": 509, "canonical_name": "星々を統べる輝き+", "mode": "stage"}],
                card_memory_hooks={127: {"force_initial_hand": True}},
                screenshots_scanned=4,
            )

        obs_lp = ScreenObservation(
            scene_type="produce_setup",
            plan="logic",
            scenario_id="hif_honsen",
            week_activity="1.アイドル選択",
            options=[
                {"choice_id": 0, "name": "次へ", "title": "次へ", "bbox": {"xmin": 240, "ymin": 1368, "xmax": 480, "ymax": 1452}},
                {"choice_id": 1, "name": "アイドル詳細(長押し)", "title": "アイドル詳細(長押し)", "is_loadout_target": True, "open_via_long_press": True, "bbox": {"xmin": 170, "ymin": 1105, "xmax": 310, "ymax": 1265}},
            ],
        )
        obs_detail = ScreenObservation(
            scene_type="produce_setup",
            plan="logic",
            scenario_id="hif_honsen",
            week_activity="2.開始確認",
            options=[
                {"choice_id": 0, "name": "プロデュース開始", "title": "プロデュース開始", "bbox": {"xmin": 255, "ymin": 1358, "xmax": 575, "ymax": 1462}},
                {"choice_id": 1, "name": "詳細", "title": "詳細", "is_loadout_target": True, "open_via_long_press": False, "bbox": {"xmin": 450, "ymin": 872, "xmax": 566, "ymax": 924}},
            ],
        )
        runner = GakumasRunner(
            mode="mock",
            controller=ctrl,
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
            loadout_scanner=SimpleNamespace(scan_via_controller=_fake_scan),
            loadout_library_path=lib_path,
        )
        cur_obs = {"val": obs_lp}
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: cur_obs["val"])
        step1 = runner.run_once()
        assert step1.action_taken is not None and step1.action_taken.payload.get("name") == "次へ"
        assert runner.tracker.initial_loadout_captured is False
        assert runner._loadout_lp_attempts == 1
        assert runner._loadout_detail_attempts == 0

        cur_obs["val"] = obs_detail
        step2 = runner.run_once()
        assert step2.action_taken is not None and step2.action_taken.payload.get("name") == "プロデュース開始"
        assert runner._loadout_detail_attempts == 2
        assert len(scan_calls) == 3
        assert scan_calls[0][1] is True and scan_calls[1][1] is False and scan_calls[2][1] is False
        assert runner.tracker.initial_loadout_captured is True
        assert runner.tracker.p_idol_id == 163
        assert runner.tracker.produce_deck == [920, 922, 521, 617, 171]
        assert lib_path.exists()
    finally:
        db.close()


def test_motivation_cost_model_pbrs_and_live_buff_alignment():
    from demo.sim.battle_simulator import HeadlessBattleSimulator
    from demo.sim.live_battle_planner import LiveBattlePlanner
    from demo.sim.rl_env import GakumasBattleEnv, GakumasFeatureEncoder

    db = GakumasDB("gakumas_library.db")
    try:
        assert GakumasFeatureEncoder.SCALAR_DIMS == 68

        # 1. Explicit live T1 buffs (3/3) must NOT be inflated to 6/6 by 花ロボ(黄) (#436), while buffs=None falls back to 6/6
        p_items = [{"item_id": 436, "canonical_name": "花ロボ(黄)", "mode": "produce"}]
        deck = [920, 922, 127, 129, 137, 159, 171, 299, 97, 305, 521, 23, 23]
        cfg_live = build_live_battle_config(
            db=db,
            deck_card_ids=deck,
            equipped_p_items=p_items,
            buffs={"goodImpressionTurns": 3, "motivation": 3},
            total_turns=12,
            scenario_id="hif_honsen",
        )
        assert cfg_live.initial_good_impression == 3
        assert cfg_live.initial_motivation == 3

        cfg_fallback = build_live_battle_config(
            db=db,
            deck_card_ids=deck,
            equipped_p_items=p_items,
            buffs=None,
            total_turns=12,
            scenario_id="hif_honsen",
        )
        assert cfg_fallback.initial_good_impression == 6
        assert cfg_fallback.initial_motivation == 6

        # 2. GakumasBattleEnv PBRS invariant: sum(mot_shaping) over full episode == -init_potential (0 terminal potential)
        env = GakumasBattleEnv(cfg_live, db=db, motivation_cost_weight=0.035)
        obs0, mask0 = env.reset(seed=17)
        assert len(obs0) == env.obs_dim
        init_pot = env._prev_mot_potential
        assert env.sim is not None

        # Draw-pile order invariance check on GakumasFeatureEncoder with new 4 Motivation dims
        sim_clone = env.sim.clone()
        sim_clone.draw_pile = list(reversed(sim_clone.draw_pile))
        np.testing.assert_allclose(obs0, env.encoder.encode(sim_clone), atol=1e-7)

        rng = np.random.default_rng(17)
        done = False
        shaping_sum = 0.0
        steps = 0
        while not done and steps < 200:
            legal = np.flatnonzero(mask0)
            aid = int(rng.choice(legal))
            obs0, mask0, _rew, done, info = env.step(aid)
            shaping_sum += float(info["mot_shaping"])
            steps += 1
        assert done is True
        assert env._motivation_potential(env.sim) == 0.0
        assert abs(shaping_sum + init_pot) < 1e-9

        # 3. _sync_sim_to_observation must trust low HUD motivation & never inflate when visual_playable is not True
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_hidden_dim=32, ppo_episodes=16, ismcts_simulations=4),
        )
        obs_low_mot = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=9,
            turn=12,
            turns_remaining=1,
            stamina=18,
            max_stamina=44,
            score=300000,
            buffs={"goodImpressionTurns": 150, "motivation": 1},
            cards=[
                RecognizedCard(
                    card_id=299,
                    name="オトメゴコロ+",
                    upgraded=True,
                    playable=True,
                    visual_playable=False,
                    bbox=BoundingBox(xmin=340, ymin=1180, xmax=520, ymax=1450),
                ),
                RecognizedCard(
                    card_id=137,
                    name="やる気は満点+",
                    upgraded=True,
                    playable=True,
                    visual_playable=True,
                    bbox=BoundingBox(xmin=530, ymin=1180, xmax=700, ymax=1450),
                ),
            ],
            drinks=[],
            options=[{"choice_id": 0, "name": "SKIP", "title": "SKIP"}],
        )
        tr = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tr.set_initial_loadout(p_idol_id=163, plan="logic", recommended_effect="goodImpressionTurns", deck_ids=deck, p_items=p_items, db=db)
        planner._train_for_battle(obs_low_mot, tr)
        assert planner._sim is not None
        planner._sim.buffs["motivation"] = 10
        pb = PromptBuilder(db)
        acts_low = pb.generate_legal_actions_and_update_obs(obs_low_mot, tr)
        planner._sync_sim_to_observation(obs_low_mot, acts_low)
        assert int(planner._sim.buffs.get("motivation", 0)) == 1
    finally:
        db.close()


def test_home_sparse_ocr_recovery_and_guards():
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        frame = np.full((1600, 720, 3), 210, dtype=np.uint8)
        # Sparse full-frame OCR returns 1 noisy line, while cropped bands recover コンテスト / プロテヨース / PLv
        def _fake_ja_ocr(arr, *a, **kw):
            h_a, w_a = arr.shape[:2]
            if (h_a, w_a) == (1600, 720):
                return {"text": "/)っⅸ冫", "lines": [{"text": "/)っⅸ冫", "bbox": [100, 300, 180, 330]}]}
            # pb_up (2x of 0.085*1600 x 0.56*720 = 272 x 806)
            if 240 <= h_a <= 300 and 760 <= w_a <= 840:
                return {"text": "プロテヨース", "lines": [{"text": "プロテヨース", "bbox": [180, 90, 620, 170]}]}
            # bn_up (2x of 0.12*1600 x 720 = 384 x 1440)
            if 340 <= h_a <= 420 and w_a >= 1400:
                return {"text": "コンテスト", "lines": [{"text": "コンテスト", "bbox": [820, 180, 1020, 230]}]}
            return {"text": "", "lines": []}

        def _fake_en_ocr(arr, *a, **kw):
            h_a, w_a = arr.shape[:2]
            # plv_up (2x of 0.09*1600 x 0.20*720 = 288 x 288)
            if 240 <= h_a <= 320 and 240 <= w_a <= 320:
                return {"text": "PLv", "lines": [{"text": "PLv", "bbox": [30, 30, 140, 120]}]}
            return {"text": "", "lines": []}

        rec.ocr = SimpleNamespace(recognize_array=_fake_ja_ocr)
        rec.en_ocr = SimpleNamespace(recognize_array=_fake_en_ocr)

        obs = rec.recognize(frame)
        assert obs.scene_type == "home"
        assert obs.options and obs.options[0]["name"] == "プロデュース"

        # _find_event_choice_pills must ignore Home labels and Home-only HUD must block produce_event fallthrough
        home_pill_lines = [
            {"text": "お仕事", "bbox": [92, 1090, 150, 1115]},
            {"text": "キャン十ン中", "bbox": [201, 1240, 318, 1259]},
            {"text": "キャンペーン中", "bbox": [201, 1240, 318, 1259]},
        ]
        assert rec._find_event_choice_pills(frame, home_pill_lines, 720, 1600) == []

        # Even if >=2 pill-like lines exist around y=0.55..0.80, Home-only HUD (e.g. PLv) prevents produce_event
        rec.ocr = SimpleNamespace(
            recognize_array=lambda arr, *a, **kw: {
                "text": "PLv\n謎の選択肢A\n謎の選択肢B",
                "lines": [
                    {"text": "PLv", "bbox": [35, 30, 110, 80]},
                    {"text": "謎の選択肢A", "bbox": [140, 950, 360, 990]},
                    {"text": "謎の選択肢B", "bbox": [140, 1060, 360, 1100]},
                ],
            }
            if arr.shape[:2] == (1600, 720)
            else {"text": "", "lines": []}
        )
        obs_guarded = rec.recognize(frame)
        assert obs_guarded.scene_type != "produce_event"
    finally:
        db.close()


def test_ocr_bottom_orange_button_rejects_small_icon_and_accepts_wide_pill():
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        rec.ocr = SimpleNamespace(recognize_array=lambda _arr, *a, **kw: {"text": "", "lines": []})

        # Small 43x35 orange Home icon (338..381 -> ox1..ox2 = 342..377, width=35) + disjoint orange glint on same row
        icon_frame = np.full((1600, 720, 3), 240, dtype=np.uint8)
        icon_frame[1461:1496, 338:381] = (245, 130, 30)
        icon_frame[1461:1496, 610:625] = (245, 130, 30)
        assert rec._ocr_bottom_orange_button(icon_frame, 720, 1600) is None

        # Wide bottom orange primary pill (220x60) must still be detected as 決定
        pill_frame = np.full((1600, 720, 3), 240, dtype=np.uint8)
        pill_frame[1430:1490, 250:470] = (245, 130, 30)
        btn = rec._ocr_bottom_orange_button(pill_frame, 720, 1600)
        assert btn is not None
        assert btn["name"] == "決定"
    finally:
        db.close()


def test_runner_single_w1_produce_event_before_setup_does_not_latch_home_stop():
    db = GakumasDB("gakumas_library.db")
    try:
        # 1. Single spurious W1 produce_event before any produce_setup/produce_schedule must NOT stop on Home
        agent1 = LocalHybridDecisionAgent(db=db)
        ctrl1 = MockController(db=db, plan="logic")
        runner1 = GakumasRunner(
            mode="mock",
            controller=ctrl1,
            agent=agent1,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        seq1 = iter([
            ScreenObservation(
                scene_type="produce_event",
                plan="logic",
                scenario_id="hif_honsen",
                week=1,
                options=[{"choice_id": 0, "name": "偽選択肢", "bbox": {"xmin": 100, "ymin": 1000, "xmax": 400, "ymax": 1060}}],
            ),
            ScreenObservation(
                scene_type="home",
                plan="logic",
                scenario_id="hif_honsen",
                options=[{"choice_id": 0, "name": "プロデュース", "title": "プロデュース", "bbox": {"xmin": 210, "ymin": 1272, "xmax": 510, "ymax": 1368}}],
            ),
            ScreenObservation(
                scene_type="produce_setup",
                plan="logic",
                scenario_id="hif_honsen",
                options=[{"choice_id": 0, "name": "次へ", "title": "次へ", "bbox": {"xmin": 240, "ymin": 1400, "xmax": 480, "ymax": 1470}}],
            ),
        ])
        runner1.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: next(seq1))
        res1 = runner1.run_loop(max_steps=3, delay_between_steps=0.0, stop_on_home_return=True)
        assert [r.scene_type for r in res1] == ["produce_event", "home", "produce_setup"]
        assert res1[1].click_coords is not None
        assert runner1._has_entered_produce is False

        # 2. Genuine produce_setup -> produce_event -> home DOES stop cleanly on home without re-clicking
        agent2 = LocalHybridDecisionAgent(db=db)
        ctrl2 = MockController(db=db, plan="logic")
        runner2 = GakumasRunner(
            mode="mock",
            controller=ctrl2,
            agent=agent2,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        seq2 = iter([
            ScreenObservation(
                scene_type="produce_setup",
                plan="logic",
                scenario_id="hif_honsen",
                options=[{"choice_id": 0, "name": "プロデュース開始", "title": "プロデュース開始", "bbox": {"xmin": 240, "ymin": 1400, "xmax": 480, "ymax": 1470}}],
            ),
            ScreenObservation(
                scene_type="produce_event",
                plan="logic",
                scenario_id="hif_honsen",
                week=1,
                options=[{"choice_id": 0, "name": "選択肢A", "bbox": {"xmin": 100, "ymin": 1000, "xmax": 400, "ymax": 1060}}],
            ),
            ScreenObservation(
                scene_type="home",
                plan="logic",
                scenario_id="hif_honsen",
                options=[{"choice_id": 0, "name": "プロデュース", "title": "プロデュース", "bbox": {"xmin": 210, "ymin": 1272, "xmax": 510, "ymax": 1368}}],
            ),
        ])
        runner2.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: next(seq2))
        res2 = runner2.run_loop(max_steps=10, delay_between_steps=0.0, stop_on_home_return=True)
        assert [r.scene_type for r in res2] == ["produce_setup", "produce_event", "home"]
        assert res2[2].click_coords is None
    finally:
        db.close()


def test_confirm_setup_true_end_condition_not_terminal_and_garbled_setup_recognized():
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. LocalHybridDecisionAgent: skip frame with ◆TrueEnd条件:達成 / 獲得量上昇 must NOT set _has_entered_produce
        agent = LocalHybridDecisionAgent(db=db)
        skip_obs = ScreenObservation(
            scene_type="skip",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="◆TrueEnd条件:達成\n獲得量上昇\nアイテム",
            options=[{"choice_id": 0, "name": "次へ", "bbox": {"xmin": 240, "ymin": 1360, "xmax": 480, "ymax": 1440}}],
        )
        pb = PromptBuilder(db)
        skip_acts = pb.generate_legal_actions_and_update_obs(skip_obs)
        agent.decide("", skip_acts, obs=skip_obs)
        assert agent._has_entered_produce is False

        setup_obs = ScreenObservation(
            scene_type="produce_setup",
            plan="logic",
            scenario_id="hif_honsen",
            week_activity="2.開始確認",
            options=[{"choice_id": 0, "name": "プロデュース開始", "title": "プロデュース開始", "bbox": {"xmin": 255, "ymin": 1358, "xmax": 575, "ymax": 1462}}],
        )
        setup_acts = pb.generate_legal_actions_and_update_obs(setup_obs)
        d_setup = agent.decide("", setup_acts, obs=setup_obs)
        assert "プロデュース開始" in setup_acts[d_setup.choice_id].description
        assert "holding without re-entry" not in d_setup.reason

        # Genuine post-produce TrueEnd / 最終評価 skip frame MUST still set _has_entered_produce
        end_skip_obs = ScreenObservation(
            scene_type="skip",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="最終評価 SS\nTrueEnd\nタップして次へ",
            options=[{"choice_id": 0, "name": "次へ", "bbox": {"xmin": 240, "ymin": 1360, "xmax": 480, "ymax": 1440}}],
        )
        end_acts = pb.generate_legal_actions_and_update_obs(end_skip_obs)
        agent.decide("", end_acts, obs=end_skip_obs)
        assert agent._has_entered_produce is True

        # 2. LiveScreenRecognizer: garbled 2.開始確認 missing 開始確認 & splitting ース開始 + 獲得量上昇 -> produce_setup
        rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        frame = np.full((1600, 720, 3), 230, dtype=np.uint8)
        garbled_lines = [
            {"text": "◆TrueEnd条件:達成", "bbox": [430, 540, 690, 568]},
            {"text": "獲得量上昇", "bbox": [410, 1250, 520, 1300]},
            {"text": "プロデュ", "bbox": [260, 1370, 360, 1410]},
            {"text": "ース開始を", "bbox": [360, 1370, 520, 1410]},
        ]
        rec.ocr = SimpleNamespace(
            recognize_array=lambda arr, *a, **kw: {"text": "\n".join(l["text"] for l in garbled_lines), "lines": garbled_lines}
            if arr.shape[:2] == (1600, 720)
            else {"text": "", "lines": []}
        )
        rec.en_ocr = SimpleNamespace(recognize_array=lambda _arr, *a, **kw: {"text": "", "lines": []})
        obs_garbled = rec.recognize(frame)
        assert obs_garbled.scene_type == "produce_setup"
        assert obs_garbled.week_activity == "2.開始確認"
        assert any(o.get("name") == "プロデュース開始" for o in obs_garbled.options)
    finally:
        db.close()


def test_change_deck_grid_clipped_ocr_and_chenshi_orange_button():
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        frame = np.full((1600, 720, 3), 40, dtype=np.uint8)
        # Upper preview modal (y=0.155..0.345 -> y=248..552)
        frame[240:555, 40:680] = (235, 235, 238)
        # 4x2 colorful cards in deck grid rows (row 0 at y=670..800, row 1 at y=820..950)
        for ry in (670, 820):
            for cx_s, cx_e in ((79, 198), (223, 342), (367, 486), (511, 630)):
                frame[ry : ry + 130, cx_s:cx_e] = (75, 165, 225)
        # Orange selection bracket around bottom-left card (79..198, 820..950)
        frame[816:954, 74:80] = (250, 125, 25)
        frame[816:954, 198:204] = (250, 125, 25)
        # Bottom orange confirm button (y=1444..1515, x=378..623)
        frame[1444:1515, 378:623] = (248, 130, 30)

        full_lines = [
            {"text": "チェンジ", "bbox": [67, 43, 174, 68]},
            {"text": "カスタマイズ確認", "bbox": [46, 441, 171, 455]},
            {"text": "ェンジするスキルカードを選択してください", "bbox": [136, 582, 610, 605]},
            {"text": "Xキャンセル", "bbox": [117, 1466, 309, 1494]},
        ]

        def _fake_ocr(arr, *a, **kw):
            h_a, w_a = arr.shape[:2]
            if (h_a, w_a) == (1600, 720):
                return {"text": "\n".join(l["text"] for l in full_lines), "lines": full_lines}
            # pv_patch: 0.155..0.345 x 0.26..0.92 -> ~304 x 475
            if 280 <= h_a <= 330 and 450 <= w_a <= 510:
                return {"text": "夢色リップ+\n好印象1.1倍", "lines": [{"text": "夢色リップ+", "bbox": [10, 10, 180, 40]}]}
            # bottom orange button crop (2x upscaled + 15px pad)
            if w_a > 300 and h_a < 220:
                return {"text": "←チェンシ", "lines": [{"text": "←チェンシ", "bbox": [30, 20, 260, 80]}]}
            return {"text": "", "lines": []}

        rec.ocr = SimpleNamespace(recognize_array=_fake_ocr)
        rec.en_ocr = SimpleNamespace(recognize_array=lambda _arr, *a, **kw: {"text": "", "lines": []})

        btn = rec._ocr_bottom_orange_button(frame, 720, 1600)
        assert btn is not None
        assert btn["name"] == "チェンジ"

        obs = rec.recognize(frame)
        assert obs.scene_type == "card_reward"
        names = [o.get("name") for o in obs.options]
        assert "チェンジする" in names
        assert any(o.get("name") == "チェンジする" and o.get("card_id") == 521 for o in obs.options)
        assert any(str(n).startswith("预览第") and "待选卡" in str(n) for n in names)
        assert not any(str(n).startswith("候选卡") for n in names)
        assert "キャンセル" in names
    finally:
        db.close()


def test_m_card_preference_unique_duplicate_guard_and_change_protection():
    from demo.core.idol_guides import get_p_idol_guide

    db = GakumasDB("gakumas_library.db")
    try:
        tr = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tr.set_initial_loadout(
            p_idol_id=163,
            plan="logic",
            recommended_effect="goodImpressionTurns",
            deck_ids=[920, 922, 127, 129, 137, 299, 521, 99, 96, 94],
            p_items=[],
            db=db,
        )
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_hidden_dim=64, ppo_episodes=16, ismcts_simulations=4),
        )
        agent.bind_tracker(tr)
        guide = get_p_idol_guide(163)

        def _pick(cid: int) -> int:
            r = db.get_card_by_id(cid)
            assert r is not None
            return agent._score_skill_card(
                card_id=cid,
                title=r["name"],
                desc=r.get("description") or "",
                target_effect="goodImpressionTurns",
                guide=guide,
                plan="logic",
                for_removal=False,
            )

        def _rem(cid: int) -> int:
            r = db.get_card_by_id(cid)
            assert r is not None
            return agent._score_skill_card(
                card_id=cid,
                title=r["name"],
                desc=r.get("description") or "",
                target_effect="goodImpressionTurns",
                guide=guide,
                plan="logic",
                for_removal=True,
            )

        # 1. Picking new rewards strongly prefers Mental cards over non-elite Active cards
        for m_id in (180, 138, 130):
            assert _pick(m_id) > _pick(96)
            assert _pick(m_id) > _pick(94)
            assert _pick(m_id) > _pick(44)
        # Elite Actives (#778, #305, #922) still stay above filler Actives (#44, #94, #96)
        assert _pick(778) > _pick(94)
        assert _pick(305) > _pick(94)
        # 美鈴 (p_idol_id=163) strictly bans #152/#153 届いて！ and #426/#427 キセキの魔法
        assert _pick(152) == -6000
        assert _pick(427) == -6000
        assert _rem(152) == 2500
        assert _rem(427) == 2500

        # 2. Swapping (for_removal=True) removes non-elite Active cards strictly before ANY Mental card
        # Even if OCR drops '+' (#98 instead of #99), M-cards remain strongly protected
        for a_id in (44, 94, 96, 97):
            assert _rem(a_id) > 500
            for m_id in (98, 99, 137, 299, 521):
                assert _rem(m_id) < 0
                assert _rem(a_id) > _rem(m_id)

        # 3. 重複不可 duplicate protection: already owning #127 strongly penalizes #126 pick & blocks add_card_to_deck
        assert _pick(126) < _pick(138)
        deck_len_before = len(tr.produce_deck)
        assert tr.add_card_to_deck(126, db=db) is False
        assert len(tr.produce_deck) == deck_len_before
        # Non-unique card #137 (is_unique=0) may still have a second copy added
        assert tr.add_card_to_deck(137, db=db) is True
        assert tr.produce_deck.count(137) == 2
    finally:
        db.close()


def test_relaxed_motivation_dominance_for_mental_engine_and_endgame():
    from demo.sim.battle_simulator import CardInstance, HeadlessBattleSimulator, build_last_run_hif_preset

    db = GakumasDB("gakumas_library.db")
    try:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        sim = HeadlessBattleSimulator(cfg, db=db, seed=101, record_logs=False)
        sim.turns_remaining = 8
        sim.card_uses_remaining = 1
        sim.stamina = 30
        sim.buffs["goodImpressionTurns"] = 12
        sim.buffs["motivation"] = 6

        def _take_card(cid: int) -> CardInstance:
            inst = next(
                c for c in (sim.hand + sim.draw_pile + sim.discard_pile + sim.exhaust_pile)
                if c.cfg.card_id == cid
            )
            for z in (sim.hand, sim.draw_pile, sim.discard_pile, sim.exhaust_pile):
                if inst in z:
                    z.remove(inst)
            return inst

        c129 = _take_card(129)
        c299 = _take_card(299)
        c305 = _take_card(305)
        c173 = _take_card(173)
        c920 = _take_card(920)
        sim.hand = [c299, c305, c173, c920]

        # 1. Without #129 engine at Mot=6, T_rem=8: #299/#173 (rem_mot=3 >= 1) and #305 (rem_mot=2 >= 1) dominate!
        assert sim.has_motivation_engine() is False
        assert sim.is_safe_dominating_free_card(c299) is True
        assert sim.is_safe_dominating_free_card(c173) is True
        assert sim.is_safe_dominating_free_card(c305) is True

        # At Mot=4 without #129, #305 would leave rem_mot=0 < 1 -> waits; #299 leaves rem_mot=1 -> still True!
        sim.buffs["motivation"] = 4
        assert sim.is_safe_dominating_free_card(c299) is True
        assert sim.is_safe_dominating_free_card(c305) is False

        # 2. Once #129 engine is active, rem_mot == 0 is allowed at T_rem=8 for #299 and #305
        sim.hand.append(c129)
        sim.play_card(sim.hand.index(c129))
        sim.card_uses_remaining = 1
        sim.turns_remaining = 8
        assert sim.has_motivation_engine() is True
        sim.buffs["motivation"] = 3
        assert sim.is_safe_dominating_free_card(c299) is True
        sim.buffs["motivation"] = 4
        assert sim.is_safe_dominating_free_card(c305) is True

        # Double active drink #25 still blocks #305 from firing automatically before an intended target
        sim.double_card_effect_cards = 1
        assert sim.is_safe_dominating_free_card(c305) is False
        sim.double_card_effect_cards = 0

        # 3. Endgame (turns_remaining <= 4) lifts even repeatable Active #97 motivation floor
        cfg_97 = cfg
        cfg_97_inst_sim = HeadlessBattleSimulator(cfg_97, db=db, seed=102, record_logs=False)
        from demo.sim.battle_simulator import _make_card_config
        c97_inst = CardInstance(uid=9991, cfg=_make_card_config(db, 97))
        cfg_97_inst_sim.hand = [c97_inst]
        cfg_97_inst_sim.card_uses_remaining = 1
        cfg_97_inst_sim.turns_remaining = 4
        cfg_97_inst_sim.buffs["motivation"] = 2
        assert cfg_97_inst_sim.is_safe_dominating_free_card(c97_inst) is True
    finally:
        db.close()


def test_w3_unswitchable_reward_slot_breaks_loop_and_w8_shop_skips_upgraded_deck():
    db = GakumasDB("gakumas_library.db")
    try:
        # 1. W3 3-choice card_reward with stuck cur_slot=1 must break out within <= 6 steps instead of looping forever
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_hidden_dim=64, ppo_episodes=16, ismcts_simulations=4),
        )
        obs_w3 = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="スキルカード選択\n受け取る",
        )
        acts_w3 = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="预览第1张候选卡",
                bbox=BoundingBox(xmin=100, ymin=1050, xmax=230, ymax=1220),
                payload={"name": "预览第1张候选卡", "slot_idx": 0},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="预览第3张候选卡",
                bbox=BoundingBox(xmin=490, ymin=1050, xmax=620, ymax=1220),
                payload={"name": "预览第3张候选卡", "slot_idx": 2},
            ),
            LegalAction(
                choice_id=2,
                action_type="pick_option",
                description="受け取る (励まし)",
                bbox=BoundingBox(xmin=240, ymin=1400, xmax=480, ymax=1470),
                payload={"name": "受け取る", "card_id": 68, "title": "励まし"},
            ),
        ]
        chosen_names = []
        for _ in range(10):
            dec = agent.decide("", acts_w3, obs=obs_w3, p_idol_id=163)
            nm = str((acts_w3[dec.choice_id].payload or {}).get("name") or "")
            chosen_names.append(nm)
            if nm == "受け取る":
                break
        assert chosen_names[-1] == "受け取る"
        assert chosen_names.count("预览第1张候选卡") <= 2
        assert chosen_names.count("预览第3张候选卡") <= 2

        # 2a. W8 shop with a fully-upgraded H.I.F deck must NOT click 強化
        tr_upg = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tr_upg.set_initial_loadout(
            p_idol_id=163,
            plan="logic",
            recommended_effect="goodImpressionTurns",
            deck_ids=[920, 922, 127, 129, 299],
            p_items=[],
            db=db,
        )
        agent.bind_tracker(tr_upg)
        w8_obs = ScreenObservation(
            scene_type="shop",
            plan="logic",
            scenario_id="hif_honsen",
            week=8,
            p_points=250,
            drinks=[],
        )
        shop_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="強化 (100P)", bbox=BoundingBox(xmin=57, ymin=1052, xmax=295, ymax=1129), payload={"name": "強化", "shop_slot_key": "強化", "price": 100}),
            LegalAction(choice_id=1, action_type="pick_option", description="カスタマイズ", bbox=BoundingBox(xmin=316, ymin=1052, xmax=561, ymax=1129), payload={"name": "カスタマイズ", "shop_slot_key": "カスタマイズ"}),
            LegalAction(choice_id=2, action_type="pick_option", description="終了", bbox=BoundingBox(xmin=563, ymin=1208, xmax=715, ymax=1280), payload={"name": "終了"}),
        ]
        assert agent.decide("", shop_acts, obs=w8_obs).choice_id == 1

        # 2b. Even without a bound tracker, if clicking 強化 stays in shop with unchanged p_points,
        # _shop_noop_features blocks 強化 permanently across subsequent P-point changes (drink purchase)
        agent_no_tr = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_hidden_dim=64, ppo_episodes=16, ismcts_simulations=4),
        )
        d1 = agent_no_tr.decide("", shop_acts, obs=w8_obs)
        assert d1.choice_id == 0
        # Still in shop at same p_points=250 -> 強化 marked no-op, picks カスタマイズ
        d2 = agent_no_tr.decide("", shop_acts, obs=w8_obs)
        assert d2.choice_id == 1
        # Simulate P-point change after buying a drink -> _shop_clicked_slots clears, yet 強化 stays blocked!
        w8_obs_spent = w8_obs.model_copy(update={"p_points": 150})
        d3 = agent_no_tr.decide("", [shop_acts[0], shop_acts[2]], obs=w8_obs_spent)
        assert d3.choice_id == 2
    finally:
        db.close()


def test_mid_battle_skip_4frame_debounce_and_watchdog_pause_during_battle_decide():
    import time as _time

    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_hidden_dim=64, ppo_episodes=16, ismcts_simulations=4),
        )
        reset_calls = []
        orig_reset = agent.reset_battle

        def _spy_reset():
            reset_calls.append(1)
            orig_reset()

        agent.reset_battle = _spy_reset  # type: ignore[method-assign]

        dummy_fr = np.full((1600, 720, 3), 235, dtype=np.uint8)
        ctrl_stub = SimpleNamespace(
            is_connected=True,
            connect=lambda: True,
            disconnect=lambda: None,
            screencap=lambda: dummy_fr,
            click=lambda _x, _y: True,
            swipe=lambda *_a, **_kw: None,
        )
        runner = GakumasRunner(
            mode="adb",
            controller=ctrl_stub,  # type: ignore[arg-type]
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
            idle_blank_tap_seconds=30.0,
        )
        runner.tracker.in_battle = True

        transient_skip = ScreenObservation(
            scene_type="skip",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="演出中",
            options=[{"choice_id": 0, "name": "次へ", "bbox": {"xmin": 240, "ymin": 1360, "xmax": 480, "ymax": 1440}}],
        )
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: transient_skip)

        # Frames 1..4 stay in_battle without calling reset_battle; frame 5 resets battle cleanly
        for _ in range(4):
            runner.run_once()
            assert runner.tracker.in_battle is True
            assert len(reset_calls) == 0
        runner.run_once()
        assert runner.tracker.in_battle is False
        assert len(reset_calls) == 1

        # Idle watchdog must NOT fire while _in_battle_decision is True during battle_turn decide()
        watchdog_checks_during_decide = []
        battle_obs = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=7,
            turn=1,
            turns_remaining=9,
            stamina=44,
            max_stamina=44,
            cards=[
                RecognizedCard(card_id=159, name="アイドル魂+", playable=True, card_type="mental", bbox=BoundingBox(xmin=460, ymin=1100, xmax=620, ymax=1350)),
            ],
            options=[{"choice_id": 0, "name": "SKIP", "title": "SKIP"}],
        )

        def _slow_battle_decide(*_a, **_kw):
            assert runner._in_battle_decision is True
            watchdog_checks_during_decide.append(runner.check_idle_watchdog(now=_time.monotonic() + 200.0))
            return DecisionResult(choice_id=0, reason="test", source="local_hybrid")

        agent.decide = _slow_battle_decide  # type: ignore[method-assign]
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: battle_obs)
        runner.start()
        runner._last_progress_ts = _time.monotonic() - 120.0
        runner._last_blank_tap_ts = _time.monotonic() - 120.0
        assert runner.check_idle_watchdog(now=_time.monotonic() + 200.0) is True
        assert runner.watchdog_tap_count == 1

        runner._last_progress_ts = _time.monotonic() - 120.0
        runner._last_blank_tap_ts = _time.monotonic() - 120.0
        runner.run_once()
        assert watchdog_checks_during_decide == [False]
        assert runner.watchdog_tap_count == 1
        assert runner._in_battle_decision is False
    finally:
        if "runner" in locals():
            runner.stop()
        db.close()


def test_exam_stamina_bar_fallback_and_turn1_no_false_skip():
    """Stamina bar estimator, stamina_ocr_zero recovery, and LiveBattlePlanner must not false-SKIP Turn 1."""
    import cv2
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    # 1. Synthetic RGB exam HUD green stamina bar (width 720, height 1600) at y=[0.114..0.123], x=[0.685..0.916]
    # y=[182..196], x=[493..659] (width 166)
    frame = np.zeros((1600, 720, 3), dtype=np.uint8)
    # Dark bar track across full bar span (S<60, V<130)
    frame[182:196, 493:659] = (30, 30, 30)
    # Left 50% bright green fill in RGB (R=30, G=220, B=90 -> HSV H~75 in [5..95], S>80, V>140)
    frame[182:196, 493:576] = (30, 220, 90)
    est_half = LiveScreenRecognizer._estimate_exam_stamina_from_bar(frame, 44)
    assert est_half is not None and 20 <= est_half <= 24

    # Full green bar -> 44 / 44
    frame_full = np.zeros((1600, 720, 3), dtype=np.uint8)
    frame_full[182:196, 493:659] = (30, 220, 90)
    est_full = LiveScreenRecognizer._estimate_exam_stamina_from_bar(frame_full, 44)
    assert est_full is not None and 42 <= est_full <= 44

    # 2. If real debug frames from W7 exist, verify RGB bar estimator reads 44/44 on Turn 1 and ~15 on rem4
    step190 = Path("debug_frames_live2/battle_W7_rem9_step190.png")
    if step190.exists():
        bgr190 = cv2.imread(str(step190))
        assert bgr190 is not None
        rgb190 = cv2.cvtColor(bgr190, cv2.COLOR_BGR2RGB)
        est190 = LiveScreenRecognizer._estimate_exam_stamina_from_bar(rgb190, 44)
        assert est190 is not None and est190 >= 42
    step215 = Path("debug_frames_live2/battle_W7_rem4_step215.png")
    if step215.exists():
        bgr215 = cv2.imread(str(step215))
        assert bgr215 is not None
        rgb215 = cv2.cvtColor(bgr215, cv2.COLOR_BGR2RGB)
        est215 = LiveScreenRecognizer._estimate_exam_stamina_from_bar(rgb215, 44)
        assert est215 is not None and 12 <= est215 <= 18

    # 3. PromptBuilder + LiveBattlePlanner recover when OCR stamina==0 but card visual_playable is True
    db = GakumasDB("gakumas_library.db")
    try:
        pb = PromptBuilder(db)
        obs = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=7,
            turn=1,
            turns_remaining=9,
            stamina=0,
            max_stamina=44,
            cards=[
                RecognizedCard(
                    card_id=159,
                    name="アイドル魂+",
                    playable=False,
                    visual_playable=True,
                    card_type="mental",
                    bbox=BoundingBox(xmin=300, ymin=1100, xmax=460, ymax=1350),
                ),
            ],
            options=[],
        )
        actions = pb.generate_legal_actions_and_update_obs(obs)
        assert obs.cards[0].playable is True
        play_acts = [a for a in actions if a.action_type == "play_card"]
        assert len(play_acts) == 1

        tracker = GameStateTracker(plan="logic")
        tracker.set_initial_loadout(
            p_idol_id=163,
            plan="logic",
            recommended_effect="goodImpressionTurns",
            deck_ids=[159, 171, 173, 181, 139, 129, 299, 920],
            p_items=[],
            db=db,
        )
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        try:
            cid, _reason, _stats = planner.decide_battle_action(obs, actions, tracker=tracker)
            chosen = next(a for a in actions if a.choice_id == cid)
            assert chosen.action_type == "play_card"
            assert chosen.payload.get("card_id") == 159
        finally:
            planner.close()
    finally:
        db.close()


def test_card173_phantom_exhaustion_recovery_unupgraded_codec_and_motivation_sync():
    """#173 must not downgrade to #172 or lock in used_limit_cards, and must play even when OCR reads low motivation."""
    from demo.sim.rl_env import GakumasActionCodec

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. Tracker phantom-exhaustion recovery + anti-downgrade (#173 stays 173 even if hand OCRs #172)
        tracker = GameStateTracker(plan="logic")
        tracker.set_initial_loadout(
            p_idol_id=163,
            plan="logic",
            recommended_effect="goodImpressionTurns",
            deck_ids=[173, 159, 171, 181, 139],
            p_items=[],
            db=db,
        )
        _b = BoundingBox(xmin=0, ymin=0, xmax=100, ymax=100)
        tracker.used_limit_cards.add(173)
        obs_172 = [
            RecognizedCard(card_id=172, name="星屑センセーション", playable=True, bbox=_b),
            RecognizedCard(card_id=159, name="アイドル魂+", playable=True, bbox=_b),
        ]
        tracker.sync_observed_hand(obs_172, db=db)
        assert 173 in tracker.produce_deck
        assert 172 not in tracker.produce_deck
        assert 173 not in tracker.used_limit_cards

        tracker.used_limit_cards.add(173)
        tracker._restore_limit_card_if_still_in_hand(
            [RecognizedCard(card_id=173, name="星屑センセーション+", playable=True, bbox=_b)],
            db=db,
        )
        assert 173 not in tracker.used_limit_cards

        # 2. GakumasActionCodec maps unupgraded base_id #172 -> canonical #173 slot
        battle_cfg = build_live_battle_config(
            db=db,
            deck_card_ids=[173, 159, 171, 181, 139],
            p_idol_id=163,
            plan="logic",
            max_stamina=44,
            total_turns=9,
        )
        codec = GakumasActionCodec(battle_cfg, db)
        assert 173 in codec.card_id_to_fallback_action
        assert 172 in codec.card_id_to_fallback_action
        assert codec.card_id_to_fallback_action[172] == codec.card_id_to_fallback_action[173]

        # 3. LiveBattlePlanner plays #173 (not held when turns_remaining<=2, and visual_playable overrides OCR mot=1)
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        )
        try:
            pb = PromptBuilder(db)
            obs = ScreenObservation(
                scene_type="battle_turn",
                plan="logic",
                scenario_id="hif_honsen",
                week=7,
                turn=9,
                turns_remaining=2,
                stamina=15,
                max_stamina=44,
                buffs={"motivation": 1, "good_impression_turns": 6},
                cards=[
                    RecognizedCard(
                        card_id=172,
                        name="星屑センセーション",
                        playable=True,
                        visual_playable=True,
                        card_type="mental",
                        bbox=BoundingBox(xmin=300, ymin=1100, xmax=460, ymax=1350),
                    ),
                ],
                options=[],
            )
            actions = pb.generate_legal_actions_and_update_obs(obs)
            cid, _reason, _stats = planner.decide_battle_action(obs, actions, tracker=tracker)
            chosen = next(a for a in actions if a.choice_id == cid)
            assert chosen.action_type == "play_card"
            assert chosen.payload.get("card_id") == 172
        finally:
            planner.close()
    finally:
        db.close()


def test_forbidden_cards_152_427_and_same_card_change_swap():
    """p_idol_id=163 bans #152/#427 and preserves clean deck via same-card swap (X -> X) in チェンジ."""
    _b = BoundingBox(xmin=0, ymin=0, xmax=100, ymax=100)
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(db)
        tracker = GameStateTracker(plan="logic")
        clean_deck = [171, 99, 181, 139, 299, 23]
        tracker.set_initial_loadout(
            p_idol_id=163,
            plan="logic",
            recommended_effect="goodImpressionTurns",
            deck_ids=list(clean_deck),
            p_items=[],
            db=db,
        )
        agent.bind_tracker(tracker)

        # 1. Reward skip guard: if all offered cards are forbidden (#152 / #427), skip/decline rather than claim
        obs_bad_pick = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="スキルカード選択\n届いて！+\nコスト0",
            options=[],
        )
        acts_bad_pick = [
            LegalAction(choice_id=0, action_type="pick_option", description="接受 #153", bbox=_b, payload={"name": "受け取る", "card_id": 153, "title": "届いて！+"}),
            LegalAction(choice_id=1, action_type="pick_option", description="预览卡 2", bbox=_b, payload={"name": "预览候选卡2", "slot_idx": 1}),
            LegalAction(choice_id=2, action_type="pick_option", description="跳过", bbox=_b, payload={"name": "スキップ"}),
        ]
        # Scan slot 2 (#427) first, then switch/confirm -> must choose スキップ (choice_id=2)
        d1 = agent.decide("reward", acts_bad_pick, obs=obs_bad_pick, p_idol_id=163)
        assert d1.choice_id == 1
        acts_bad_pick2 = [
            LegalAction(choice_id=0, action_type="pick_option", description="接受 #427", bbox=_b, payload={"name": "受け取る", "card_id": 427, "title": "キセキの魔法+"}),
            LegalAction(choice_id=1, action_type="pick_option", description="预览卡 1", bbox=_b, payload={"name": "预览候选卡1", "slot_idx": 0}),
            LegalAction(choice_id=2, action_type="pick_option", description="跳过", bbox=_b, payload={"name": "スキップ"}),
        ]
        d2 = agent.decide("reward", acts_bad_pick2, obs=obs_bad_pick, p_idol_id=163)
        assert d2.choice_id == 2

        # 2. Same-card swap in チェンジ: Page 1 picks owned #99 (beats filler #44 and forbidden #152)
        obs_chg1 = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジで獲得するスキルカードを選択\nアイドル宣言+",
            options=[],
        )
        acts_chg1 = [
            LegalAction(choice_id=0, action_type="pick_option", description="次へ #99", bbox=_b, payload={"name": "次へ", "card_id": 99, "title": "アイドル宣言+"}),
            LegalAction(choice_id=1, action_type="pick_option", description="预览卡 2", bbox=_b, payload={"name": "预览候选卡2", "slot_idx": 1}),
            LegalAction(choice_id=2, action_type="pick_option", description="预览卡 3", bbox=_b, payload={"name": "预览候选卡3", "slot_idx": 2}),
        ]
        r1 = agent.decide("chg1", acts_chg1, obs=obs_chg1, p_idol_id=163)
        assert r1.choice_id == 1
        acts_chg1_b = [
            LegalAction(choice_id=0, action_type="pick_option", description="次へ #44", bbox=_b, payload={"name": "次へ", "card_id": 44, "title": "表現の基本"}),
            LegalAction(choice_id=1, action_type="pick_option", description="预览卡 1", bbox=_b, payload={"name": "预览候选卡1", "slot_idx": 0}),
            LegalAction(choice_id=2, action_type="pick_option", description="预览卡 3", bbox=_b, payload={"name": "预览候选卡3", "slot_idx": 2}),
        ]
        r2 = agent.decide("chg1", acts_chg1_b, obs=obs_chg1, p_idol_id=163)
        assert r2.choice_id == 2
        acts_chg1_c = [
            LegalAction(choice_id=0, action_type="pick_option", description="次へ #152", bbox=_b, payload={"name": "次へ", "card_id": 152, "title": "届いて！"}),
            LegalAction(choice_id=1, action_type="pick_option", description="预览卡 1", bbox=_b, payload={"name": "预览候选卡1", "slot_idx": 0}),
            LegalAction(choice_id=2, action_type="pick_option", description="预览卡 2", bbox=_b, payload={"name": "预览候选卡2", "slot_idx": 1}),
        ]
        r3 = agent.decide("chg1", acts_chg1_c, obs=obs_chg1, p_idol_id=163)
        # Must switch back to slot 1 (#99) rather than claiming #152!
        assert r3.choice_id == 1
        r4 = agent.decide("chg1", acts_chg1, obs=obs_chg1, p_idol_id=163)
        assert r4.choice_id == 0
        assert agent._pending_change_gain_base_id == 98

        # Page 2 (`チェンジする`): starts on #171 (slot 0), scans #181 (slot 1), finds #99 (slot 2) and swaps #99 -> #99!
        obs_chg2 = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジするスキルカードを選択",
            options=[],
        )
        acts_chg2_s0 = [
            LegalAction(choice_id=0, action_type="pick_option", description="チェンジ #171", bbox=_b, payload={"name": "チェンジする", "card_id": 171, "slot_idx": 0, "title": "私がスター+"}),
            LegalAction(choice_id=1, action_type="pick_option", description="待选卡 2", bbox=_b, payload={"name": "待选卡2", "slot_idx": 1}),
            LegalAction(choice_id=2, action_type="pick_option", description="待选卡 3", bbox=_b, payload={"name": "待选卡3", "slot_idx": 2}),
        ]
        p2_a = agent.decide("chg2", acts_chg2_s0, obs=obs_chg2, p_idol_id=163)
        assert p2_a.choice_id in (1, 2)
        next_slot = 1 if p2_a.choice_id == 1 else 2
        if next_slot == 1:
            acts_chg2_s1 = [
                LegalAction(choice_id=0, action_type="pick_option", description="チェンジ #181", bbox=_b, payload={"name": "チェンジする", "card_id": 181, "slot_idx": 1, "title": "届ける心+"}),
                LegalAction(choice_id=1, action_type="pick_option", description="待选卡 1", bbox=_b, payload={"name": "待选卡1", "slot_idx": 0}),
                LegalAction(choice_id=2, action_type="pick_option", description="待选卡 3", bbox=_b, payload={"name": "待选卡3", "slot_idx": 2}),
            ]
            p2_b = agent.decide("chg2", acts_chg2_s1, obs=obs_chg2, p_idol_id=163)
            assert p2_b.choice_id == 2
        acts_chg2_s2 = [
            LegalAction(choice_id=0, action_type="pick_option", description="チェンジ #99", bbox=_b, payload={"name": "チェンジする", "card_id": 99, "slot_idx": 2, "title": "アイドル宣言+"}),
            LegalAction(choice_id=1, action_type="pick_option", description="待选卡 1", bbox=_b, payload={"name": "待选卡1", "slot_idx": 0}),
            LegalAction(choice_id=2, action_type="pick_option", description="待选卡 2", bbox=_b, payload={"name": "待选卡2", "slot_idx": 1}),
        ]
        p2_confirm = agent.decide("chg2", acts_chg2_s2, obs=obs_chg2, p_idol_id=163)
        assert p2_confirm.choice_id == 0
        assert agent._pending_change_gain_base_id is None

        # 3. Shop urgency boost when #427 is in deck + _shop_noop_features preserved across transient produce_event
        tracker.produce_deck = [171, 99, 427]
        agent._shop_noop_features.add("強化")
        agent.decide("event", [LegalAction(choice_id=0, action_type="pick_option", description="ok", bbox=_b, payload={"name": "決定"})], obs=ScreenObservation(scene_type="produce_event", plan="logic"))
        assert "強化" in agent._shop_noop_features
        obs_shop = ScreenObservation(scene_type="shop", plan="logic", scenario_id="hif_honsen", p_points=120)
        shop_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="セレクトチェンジ", bbox=_b, payload={"name": "セレクトチェンジ", "shop_slot_key": "セレクトチェンジ", "price": 50}),
            LegalAction(choice_id=1, action_type="pick_option", description="強化", bbox=_b, payload={"name": "強化", "shop_slot_key": "強化", "price": 40}),
            LegalAction(choice_id=2, action_type="pick_option", description="終了", bbox=_b, payload={"name": "終了", "shop_slot_key": "終了"}),
        ]
        shop_dec = agent.decide("shop", shop_acts, obs=obs_shop, p_idol_id=163)
        assert shop_dec.choice_id == 0

        # 4. Runner _pending_change_pickup_cid + チェンジする swaps base_id cleanly even if unupgraded OCR'd
        ctrl = MockController(db=db, plan="logic")
        runner = GakumasRunner(
            mode="mock",
            controller=ctrl,
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        runner.tracker.produce_deck = [171, 99, 181, 139]
        p1_chg_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジで獲得するスキルカードを選択",
            options=[
                {"choice_id": 0, "name": "次へ", "card_id": 99, "title": "アイドル宣言+", "bbox": {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100}},
            ],
        )
        runner.agent = SimpleNamespace(
            decide=lambda *a, **kw: DecisionResult(choice_id=0, reason="p1", source="stub"),
            export_session=lambda: {},
            load_session=lambda _s: True,
            reset_session=lambda: None,
        )
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: p1_chg_obs)
        runner.run_once()
        assert runner._pending_change_pickup_cid == 99

        # Page 2 swaps out base_id 98 (matching 99 in deck) and adds 99 -> net deck composition preserved!
        p2_chg_obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            raw_text="チェンジするスキルカードを選択",
            options=[
                {"choice_id": 0, "name": "チェンジする", "card_id": 98, "title": "アイドル宣言", "bbox": {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100}},
            ],
        )
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: p2_chg_obs)
        runner.run_once()
        assert runner._pending_change_pickup_cid is None
        assert sorted(runner.tracker.produce_deck) == sorted([171, 99, 181, 139])
    finally:
        db.close()


def test_buff_toast_occlusion_detection_and_runner_settle_recapture():
    """Floating buff toast over left-HUD numbers must flag buff_hud_occluded, drop corrupted digits, and trigger recapture."""
    import cv2
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    # 1. Line-based floating buff toast detector
    assert LiveScreenRecognizer._has_left_buff_toast_in_lines(
        [{"text": "好印象", "bbox": [120, 210, 250, 255]}], w=720, h=1600
    ) is True
    assert LiveScreenRecognizer._has_left_buff_toast_in_lines(
        [{"text": "やる気+4", "bbox": [95, 240, 240, 285]}], w=720, h=1600
    ) is True
    assert LiveScreenRecognizer._has_left_buff_toast_in_lines(
        [
            {"text": "306", "bbox": [85, 195, 160, 225]},
            {"text": "14", "bbox": [85, 255, 140, 285]},
            {"text": "1回", "bbox": [85, 315, 140, 345]},
        ],
        w=720,
        h=1600,
    ) is False

    # 2. Live debug frames: step296 (occluded by 好印象 toast) vs step277 (clean 49 / 12)
    db = GakumasDB("gakumas_library.db")
    try:
        reco = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        step296 = Path("debug_frames_live2/battle_W9_rem1_step296.png")
        if step296.exists():
            bgr296 = cv2.imread(str(step296))
            assert bgr296 is not None
            obs296 = reco.recognize(cv2.cvtColor(bgr296, cv2.COLOR_BGR2RGB), plan="logic")
            assert obs296.buff_hud_occluded is True
            assert "goodImpressionTurns" not in obs296.buffs
            assert obs296.buffs.get("motivation") == 14

        step277 = Path("debug_frames_live2/battle_W9_rem8_step277.png")
        if step277.exists():
            bgr277 = cv2.imread(str(step277))
            assert bgr277 is not None
            obs277 = reco.recognize(cv2.cvtColor(bgr277, cv2.COLOR_BGR2RGB), plan="logic")
            assert obs277.buff_hud_occluded is False
            assert obs277.buffs.get("goodImpressionTurns") == 49
            assert obs277.buffs.get("motivation") == 12

        # 3. Runner re-captures when buff_hud_occluded is True and forwards clean observation to agent.decide
        dummy_frame = np.zeros((1600, 720, 3), dtype=np.uint8)
        screencap_calls = {"n": 0}
        seen_buffs = []

        class _StubLiveCtrl:
            is_connected = True

            def connect(self):
                return True

            def screencap(self):
                screencap_calls["n"] += 1
                return dummy_frame

            def click(self, _x, _y):
                return True

        obs_occluded = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=9,
            turn=12,
            turns_remaining=1,
            stamina=20,
            max_stamina=44,
            buffs={"motivation": 14},
            buff_hud_occluded=True,
            cards=[
                RecognizedCard(
                    card_id=159,
                    name="アイドル魂+",
                    playable=True,
                    card_type="mental",
                    bbox=BoundingBox(xmin=200, ymin=1100, xmax=350, ymax=1350),
                )
            ],
            options=[{"choice_id": 0, "name": "SKIP", "title": "SKIP"}],
        )
        obs_clean = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=9,
            turn=12,
            turns_remaining=1,
            stamina=20,
            max_stamina=44,
            buffs={"goodImpressionTurns": 306, "motivation": 14},
            buff_hud_occluded=False,
            cards=[
                RecognizedCard(
                    card_id=159,
                    name="アイドル魂+",
                    playable=True,
                    card_type="mental",
                    bbox=BoundingBox(xmin=200, ymin=1100, xmax=350, ymax=1350),
                )
            ],
            options=[{"choice_id": 0, "name": "SKIP", "title": "SKIP"}],
        )
        recog_calls = {"n": 0}

        def _fake_recognize(_fr):
            recog_calls["n"] += 1
            return obs_occluded if recog_calls["n"] == 1 else obs_clean

        def _capture_decide(*_args, **kwargs):
            o = kwargs.get("obs")
            seen_buffs.append(dict(o.buffs))
            return DecisionResult(choice_id=0, reason="test_clean_buffs", source="local_hybrid")

        agent = LocalHybridDecisionAgent(db)
        agent.decide = _capture_decide  # type: ignore[method-assign]
        runner = GakumasRunner(
            mode="adb",
            controller=_StubLiveCtrl(),
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        runner._battle_toast_settle_sleep = 0.0
        runner.scene_routing_reco = SimpleNamespace(recognize=_fake_recognize)
        runner.run_once()
        assert screencap_calls["n"] == 2
        assert seen_buffs == [{"goodImpressionTurns": 306, "motivation": 14}]
    finally:
        db.close()


def test_shop_delete_submodal_routes_to_card_reward_and_recognizes_confirm(monkeypatch):
    """Shop full-screen 削除 grid must route to card_reward even when preview contains 'ターン終了時'."""
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        frame = np.full((1600, 720, 3), 170, dtype=np.uint8)
        # Paint bright orange confirm pill at bottom-right (y=1445..1515, x=400..610)
        frame[1445:1515, 400:610] = (245, 130, 35)
        # Darken trailing row5 col3 slot (y=1138..1257, x=512..630)
        frame[1138:1257, 512:630] = (110, 110, 110)

        rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        full_lines = [
            {"text": "削除", "bbox": [66, 43, 119, 69]},
            {"text": "虹色ドリーマー+", "bbox": [195, 111, 415, 137]},
            {"text": "以降、ターン終了時、好印象+3", "bbox": [195, 192, 600, 224]},
            {"text": "Xキャンセル", "bbox": [117, 1466, 309, 1494]},
            {"text": "削除", "bbox": [497, 1467, 551, 1493]},
        ]
        pv_lines = [
            {"text": "虹色ドリーマー+", "bbox": [20, 20, 220, 50]},
            {"text": "以降、ターン終了時、好印象+3", "bbox": [20, 60, 360, 90]},
        ]

        def _fake_ocr(arr):
            if arr.shape[0] < 500:
                return {"text": "虹色ドリーマー+", "lines": pv_lines}
            return {"text": "削除 虹色ドリーマー+ キャンセル", "lines": full_lines}

        monkeypatch.setattr(rec.ocr, "recognize_array", _fake_ocr)
        monkeypatch.setattr(rec.en_ocr, "recognize_array", lambda _arr: {"text": "削除", "lines": [{"text": "削除", "bbox": [50, 10, 120, 40]}]})
        obs = rec.recognize(frame, plan="logic")
        assert obs.scene_type == "card_reward"
        conf = next((o for o in obs.options if o.get("name") == "削除する"), None)
        assert conf is not None
        assert conf.get("card_id") == 181
        assert conf["bbox"]["ymin"] >= int(1600 * 0.74)
    finally:
        db.close()


def test_exam_7digit_score_sparkle_recovery_and_rank_result_score(monkeypatch):
    """Exam score OCR must handle 7-digit shrunken digits, sparkle-damaged step153, and 現在順位 bonus sums."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")

        # 1. Debug frames from live3 (W7 rem=1 and sparkle-covered W9 rem=1)
        f_w7 = Path("debug_frames_live3/battle_W7_rem1_step062.png")
        if f_w7.exists():
            rgb7 = cv2.cvtColor(cv2.imread(str(f_w7)), cv2.COLOR_BGR2RGB)
            assert rec.recognize(rgb7, plan="logic").score == 771647
        f_w9 = Path("debug_frames_live3/battle_W9_rem1_step153.png")
        if f_w9.exists():
            rgb9 = cv2.cvtColor(cv2.imread(str(f_w9)), cv2.COLOR_BGR2RGB)
            assert rec.recognize(rgb9, plan="logic").score == 891502

        try:
            font_17 = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 21)
            font_12 = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 13)
            font_big = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 34)
            font_cyan = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 24)
        except Exception:
            font_17 = ImageFont.load_default()
            font_12 = ImageFont.load_default()
            font_big = ImageFont.load_default()
            font_cyan = ImageFont.load_default()

        # 2. Synthetic 720x1600 exam HUD frame with 7-digit (~17px tall) score 1682934 in 1st box
        exam_frame = np.full((1600, 720, 3), 55, dtype=np.uint8)
        pil_e = Image.fromarray(exam_frame)
        dr_e = ImageDraw.Draw(pil_e)
        # Black outline + white 7-digit text vertically centered inside y=136..156
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx or dy:
                    dr_e.text((226 + dx, 136 + dy), "1682934", fill=(10, 10, 10), font=font_17)
                    dr_e.text((385 + dx, 142 + dy), "611388", fill=(10, 10, 10), font=font_12)
        dr_e.text((226, 136), "1682934", fill=(255, 255, 255), font=font_17)
        dr_e.text((385, 142), "611388", fill=(255, 255, 255), font=font_12)
        exam_rgb = np.array(pil_e)

        obs_exam = rec._parse_battle_screen(
            exam_rgb,
            lines=[
                {"text": "残りターン", "bbox": [14, 11, 110, 30]},
                {"text": "1st", "bbox": [230, 16, 300, 56]},
            ],
            plan="logic",
            meta={
                "week": 9,
                "total_weeks": 13,
                "week_label": "最終試験",
                "stamina": 24,
                "max_stamina": 44,
                "p_points": 125,
                "stats": {"vocal": 3000, "dance": 1200, "visual": 2000},
            },
            turn_box_txt="残りターン 1",
        )
        assert obs_exam.score == 1682934

        # 3. Synthetic 720x1600 現在順位 result screen: 1,682,934 (white) + 1,944,735 (cyan) -> 3,627,669
        def _render_rank_frame(with_cyan: bool) -> np.ndarray:
            arr = np.full((1600, 720, 3), 24, dtype=np.uint8)
            pil_r = Image.fromarray(arr)
            dr_r = ImageDraw.Draw(pil_r)
            dr_r.text((210, 344), "1,682,934", fill=(255, 255, 255), font=font_big)
            if with_cyan:
                dr_r.text((395, 352), "+1,944,735", fill=(110, 220, 255), font=font_cyan)
            return np.array(pil_r)

        rank_lines = [
            {"text": "現在順位", "bbox": [40, 224, 145, 250]},
            {"text": "秦谷美鈴", "bbox": [210, 290, 350, 320]},
        ]
        monkeypatch.setattr(rec.ocr, "recognize_array", lambda _arr: {"text": "現在順位 秦谷美鈴", "lines": rank_lines})
        obs_rank_both = rec.recognize(_render_rank_frame(with_cyan=True), plan="logic")
        assert obs_rank_both.score == 3627669
        assert "3,627,669" in obs_rank_both.week_activity

        obs_rank_single = rec.recognize(_render_rank_frame(with_cyan=False), plan="logic")
        assert obs_rank_single.score == 1682934

        # 4. Synthetic cyan-only raw_vals case still triggers white_vals to recover base score
        arr_split = np.full((1600, 720, 3), 24, dtype=np.uint8)
        pil_sp = Image.fromarray(arr_split)
        dr_sp = ImageDraw.Draw(pil_sp)
        dr_sp.text((210, 344), "486,580", fill=(255, 255, 255), font=font_big)
        dr_sp.text((395, 352), "+1,095,642", fill=(110, 220, 255), font=font_cyan)
        sp_frame = np.array(pil_sp)

        orig_en = rec.en_ocr.recognize_array
        def _cyan_biased_en(img_arr):
            if img_arr.shape[0] > 200 or img_arr.shape[1] > 900:
                # Simulate raw/inv missing white score and only catching cyan +1,095,642
                if float(img_arr.mean()) < 100.0 or int((img_arr < 80).sum()) > 10:
                    return {"text": "+1,095,642", "lines": []}
            return orig_en(img_arr)

        monkeypatch.setattr(rec.en_ocr, "recognize_array", _cyan_biased_en)
        # Note: above uses real en_ocr on native-scale white pad=40, which reads "486,580"
        obs_split = rec.recognize(sp_frame, plan="logic")
        assert obs_split.score == 1582222
        assert "486,580+1,095,642=1,582,222" in obs_split.week_activity
    finally:
        db.close()


def test_live4_scoreboard_left_buff_accuracy_and_final_shop_refresh_and_reroll():
    """Verify live4 frame accuracy (no 10x core spike, 3-digit GI, moved MOT), final shop refresh, and early reroll."""
    import cv2
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer
    from demo.maa_bridge.runner import GakumasRunner
    from demo.core.tracker import GameStateTracker

    db = GakumasDB("gakumas_library.db")
    try:
        rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        live4_cases = [
            ("debug_frames_live4/battle_W7_rem4_step200.png", 66159, 108, (17,), False),
            ("debug_frames_live4/battle_W7_rem2_step204.png", 167352, 174, (13, 15), False),
            ("debug_frames_live4/battle_W7_rem1_step207.png", 234370, 194, (8,), False),
            ("debug_frames_live4/battle_W9_rem6_step259.png", 89441, None, (1,), True),
            ("debug_frames_live4/battle_W9_rem1_step270.png", 457492, 282, (6,), False),
        ]
        for rel_p, exp_sc, exp_gi, exp_mots, exp_occ in live4_cases:
            p = Path(rel_p)
            if not p.exists():
                continue
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            obs = rec.recognize(rgb, plan="logic")
            assert obs.score == exp_sc, f"{rel_p}: expected score={exp_sc}, got {obs.score}"
            assert obs.buffs.get("goodImpressionTurns") == exp_gi, f"{rel_p}: expected GI={exp_gi}, got {obs.buffs}"
            assert obs.buffs.get("motivation") in exp_mots, f"{rel_p}: expected MOT in {exp_mots}, got {obs.buffs}"
            assert obs.buff_hud_occluded is exp_occ, f"{rel_p}: expected occ={exp_occ}, got {obs.buff_hud_occluded}"

        # 2. Final shop (Week 8): spend down P-points via リフレッシュ even with 4/4 drinks, stop when low
        agent = LocalHybridDecisionAgent(db=db)
        tracker = GameStateTracker()
        tracker.p_idol_id = 163
        tracker.recommended_effect = "goodImpressionTurns"
        agent.bind_tracker(tracker)

        db_box = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)
        full_drinks = [
            RecognizedDrink(drink_id=10, name="烏龍茶", description="", bbox=db_box),
            RecognizedDrink(drink_id=11, name="ホットコーヒー", description="", bbox=db_box),
            RecognizedDrink(drink_id=12, name="初星水", description="", bbox=db_box),
            RecognizedDrink(drink_id=13, name="ミックススムージー", description="", bbox=db_box),
        ]

        def _shop_actions():
            return [
                LegalAction(
                    choice_id=0,
                    action_type="pick_option",
                    description="リフレッシュ (50P)",
                    bbox=db_box,
                    payload={"name": "リフレッシュ", "shop_slot_key": "リフレッシュ", "price": 50},
                ),
                LegalAction(
                    choice_id=1,
                    action_type="pick_option",
                    description="終了",
                    bbox=db_box,
                    payload={"name": "終了"},
                ),
            ]

        obs_hi = ScreenObservation(
            scene_type="shop",
            plan="logic",
            scenario_id="hif_honsen",
            week=8,
            total_weeks=9,
            p_points=250,
            drinks=full_drinks,
            options=[],
        )
        dec_refresh = agent.decide("", _shop_actions(), obs=obs_hi)
        assert dec_refresh.choice_id == 0

        obs_lo = obs_hi.model_copy(update={"p_points": 60})
        dec_exit = agent.decide("", _shop_actions(), obs=obs_lo)
        assert dec_exit.choice_id == 1

        # Also verify GakumasRunner clears _shop_visited_slots and deducts price on リフレッシュ
        dummy_shop_frame = np.full((1600, 720, 3), 235, dtype=np.uint8)
        ctrl_stub = SimpleNamespace(
            is_connected=True,
            connect=lambda: True,
            disconnect=lambda: None,
            screencap=lambda: dummy_shop_frame,
            click=lambda _x, _y: True,
            swipe=lambda x1, y1, x2, y2, duration_ms=650: None,
        )
        runner2 = GakumasRunner(
            mode="adb",
            controller=ctrl_stub,  # type: ignore[arg-type]
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        agent.reset_session()
        runner2.tracker.p_points = 250
        rec._shop_visited_slots.update({"Pドリンク:0", "カスタマイズ"})
        rec._last_p_points = 250
        obs_mock = obs_hi.model_copy(update={
            "options": [{
                "choice_id": 0,
                "name": "リフレッシュ",
                "title": "リフレッシュ (50P)",
                "shop_slot_key": "リフレッシュ",
                "price": 50,
                "bbox": {"xmin": 24, "ymin": 1204, "xmax": 207, "ymax": 1304},
            }]
        })
        runner2.scene_routing_reco = SimpleNamespace(recognizer=rec, recognize=lambda _fr, **_kw: obs_mock)
        runner2.run_once()
        assert len(rec._shop_visited_slots) == 0
        assert runner2.tracker.p_points == 200
        assert rec._last_p_points == 200

        # 3. Early card_reward / チェンジで獲得 rerolls non-exhaust M-card (#68 励まし) via 再抽選,
        #    and accepts exhaustible M-card (#180 虹色ドリーマー) directly without rerolling.
        agent.reset_session()
        obs_chg = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            week=3,
            total_weeks=9,
            raw_text="チェンジで獲得\n再抽選",
            options=[],
        )
        acts_non_ex = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="励まし",
                bbox=db_box,
                payload={"name": "受け取る", "card_id": 68, "title": "励まし"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="再抽選 (免费重抽候选卡)",
                bbox=db_box,
                payload={"name": "再抽選", "title": "再抽選 (免费重抽候选卡)"},
            ),
        ]
        dec_reroll = agent.decide("", acts_non_ex, obs=obs_chg)
        assert dec_reroll.choice_id == 1

        acts_ex = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="虹色ドリーマー",
                bbox=db_box,
                payload={"name": "受け取る", "card_id": 180, "title": "虹色ドリーマー"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="再抽選 (免费重抽候选卡)",
                bbox=db_box,
                payload={"name": "再抽選", "title": "再抽選 (免费重抽候选卡)"},
            ),
        ]
        dec_keep = agent.decide("", acts_ex, obs=obs_chg)
        assert dec_keep.choice_id == 0
    finally:
        db.close()


def test_loadout_scanner_preserves_compact_adjacent_cards_and_split_spans():
    """LoadoutScanner must split merged tall icon spans, allow compact <120px spacing, resolve #171 by body text, and survive 1 stale swipe."""
    db = GakumasDB("gakumas_library.db")
    scanner = HifLoadoutScanner(db=db)
    try:
        # 1. Compact adjacent cards #175 and #171 (~95px apart) + garbled #171 title recovered via body text
        rng = np.random.default_rng(31)
        doc = np.full((1600, 720, 3), 255, dtype=np.uint8)
        # Put non-white icon boxes at y=520..605 (P-item, rel 114), y=750..830 (#175, rel 344), y=845..925 (#171, rel 439 -> 95px spacing!)
        for by in (520, 750, 845):
            doc[by : by + 80, 55:170] = rng.integers(30, 140, size=(80, 115, 3), dtype=np.uint8)
            doc[by + 8 : by + 72, 200:600] = rng.integers(180, 240, size=(64, 400, 3), dtype=np.uint8)

        def _fake_ja_compact(arr: np.ndarray, *a, **kw):
            ah = arr.shape[0]
            if 220 <= ah <= 300:
                return {
                    "text": "SSR ガラクタロード 秦谷 美鈴",
                    "lines": [
                        {"text": "SSR ガラクタロード", "bbox": [180, 20, 460, 50]},
                        {"text": "秦谷 美鈴", "bbox": [180, 55, 360, 85]},
                    ],
                }
            if ah >= 900:
                return {
                    "text": "",
                    "lines": [
                        {"text": "Pアイテム", "bbox": [24, 90, 140, 110]},
                        {"text": "星々を統べる輝き+", "bbox": [200, 118, 480, 145]},
                        {"text": "直接効果で好印象が4回増加時", "bbox": [200, 150, 560, 178]},
                        {"text": "スキルカード", "bbox": [24, 305, 160, 330]},
                        {"text": "ノートの端の決意+", "bbox": [200, 346, 480, 374]},
                        {"text": "好印象+7 / レッスン中1回", "bbox": [200, 378, 520, 404]},
                        {"text": "不明タイトル+", "bbox": [200, 441, 420, 469]},
                        {"text": "好印象消費1 / ターン追加+1 / スキルカード使用数追加+1 / スキルカードを引く", "bbox": [200, 472, 620, 500]},
                    ],
                }
            return {"text": "", "lines": []}

        scanner.ocr = SimpleNamespace(recognize_array=_fake_ja_compact)
        scanner.reco.en_ocr = SimpleNamespace(recognize_array=lambda _arr, *a, **kw: {"text": "2500", "lines": []})
        res_compact = scanner.parse_detail_frames([doc], plan="logic")
        assert 175 in res_compact.deck_card_ids
        assert 171 in res_compact.deck_card_ids

        # 2. _find_left_icon_spans splits a tall (>145px) merged icon span cleanly
        vp = np.full((600, 720, 3), 255, dtype=np.uint8)
        vp[100:195, int(720 * 0.08) : int(720 * 0.22)] = 80
        vp[195:203, int(720 * 0.08) : int(720 * 0.13)] = 215  # 36% of icon band (<232) -> bridges 8px gap
        vp[203:298, int(720 * 0.08) : int(720 * 0.22)] = 80
        split_spans = scanner._find_left_icon_spans(vp, 720)
        assert len(split_spans) == 2

        # 3. scan_via_controller survives 1 stale swipe and only stops after 2 consecutive stagnant frames
        rng2 = np.random.default_rng(99)
        doc2 = np.full((2600, 720, 3), 255, dtype=np.uint8)
        for by in (700, 1200, 1700):
            doc2[by : by + 95, 55:170] = rng2.integers(20, 130, size=(95, 115, 3), dtype=np.uint8)
            doc2[by + 10 : by + 85, 200:580] = rng2.integers(140, 220, size=(75, 380, 3), dtype=np.uint8)
        p0 = doc2[:1600].copy()
        p1 = doc2[420:2020].copy()
        p2 = doc2[840:2440].copy()
        seq = [p0, p0, p1, p1, p2, p2, p2]  # initial frame0 + [p0, p1(diff), p1(stale=1 -> continue!), p2(diff), p2(stale=1), p2(stale=2 -> break)]
        state_idx = {"i": 0, "swipes": 0}

        class _ScanCtrl:
            def long_press(self, *_a, **_kw):
                return True
            def click(self, *_a, **_kw):
                return True
            def swipe(self, *_a, **_kw):
                state_idx["swipes"] += 1
                return True
            def screencap(self):
                idx = min(state_idx["i"], len(seq) - 1)
                state_idx["i"] += 1
                return seq[idx]

        def _fake_ja_ctrl(arr: np.ndarray, *a, **kw):
            ah = arr.shape[0]
            if ah > 600 and ah < 800:
                return {"text": "好印象 基本情報", "lines": []}
            if 220 <= ah <= 300:
                return {"text": "秦谷 美鈴", "lines": [{"text": "秦谷 美鈴", "bbox": [180, 55, 360, 85]}]}
            if ah <= 200:
                return {"text": "OK", "lines": []}
            # vp captures
            if np.array_equal(arr, p2[int(1600 * 0.254) : int(1600 * 0.858), :]):
                return {
                    "text": "",
                    "lines": [{"text": "私がスター+", "bbox": [200, 456, 440, 486]}],
                }
            return {
                "text": "",
                "lines": [
                    {"text": "スキルカード", "bbox": [24, 250, 160, 275]},
                    {"text": "ワクワクが止まらない+", "bbox": [200, 300, 480, 330]},
                ],
            }

        scanner.ocr = SimpleNamespace(recognize_array=_fake_ja_ctrl)
        orig_sleep = __import__("time").sleep
        __import__("time").sleep = lambda *_a: None
        try:
            res_ctrl = scanner.scan_via_controller(_ScanCtrl(), open_via_long_press=True, plan="logic")
        finally:
            __import__("time").sleep = orig_sleep
        assert state_idx["swipes"] >= 4
        assert 171 in res_ctrl.deck_card_ids
    finally:
        db.close()


def test_battle_planner_unmapped_card_fallback_and_star_slot():
    """ActionCodec pre-registers #170/#171 + resolves base_id fallback, and LiveBattlePlanner never drops unmapped playable cards."""
    from demo.sim.battle_simulator import _make_card_config
    from demo.sim.live_battle_planner import LiveBattlePlanner, LiveBattlePlannerConfig, build_live_battle_config
    from demo.sim.rl_env import GakumasActionCodec

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. ActionCodec built from logic deck WITHOUT #171 still registers #170/#171, and maps #172 -> #173 base_id
        cfg = build_live_battle_config(
            db=db,
            deck_card_ids=[920, 922, 127, 129, 137, 173],
            equipped_p_items=[],
            plan="logic",
            total_turns=9,
        )
        codec = GakumasActionCodec(cfg, db=db)
        assert codec.card_slot_index(_make_card_config(db, 171)) >= 0
        assert codec.card_slot_index(_make_card_config(db, 170)) >= 0
        assert codec.card_slot_index(_make_card_config(db, 172)) == codec.card_slot_index(_make_card_config(db, 173)) >= 0

        # 2. Planner Tier 1 + Tier 2 fallback when codec lacks unmapped cards
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_episodes=4, ismcts_simulations=4),
        )
        tr = GameStateTracker()
        tr.p_idol_id = 163
        tr.recommended_effect = "goodImpressionTurns"
        tr.produce_deck = [920, 922, 127, 129, 137, 175, 297, 711, 159, 173, 778, 617, 181, 299, 23]

        box = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)
        obs_star = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=7,
            turn=3,
            turns_remaining=7,
            stamina=36,
            max_stamina=44,
            score=12000,
            buffs={"goodImpressionTurns": 12, "motivation": 6},
            cards=[
                RecognizedCard(card_id=171, name="私がスター+", upgraded=True, playable=True, visual_playable=True, bbox=box),
                RecognizedCard(card_id=137, name="やる気は満点+", upgraded=True, playable=True, visual_playable=True, bbox=box),
            ],
        )
        acts_star = [
            LegalAction(choice_id=0, action_type="play_card", description="私がスター+", bbox=box, payload={"card_id": 171, "name": "私がスター+"}),
            LegalAction(choice_id=1, action_type="play_card", description="やる気は満点+", bbox=box, payload={"card_id": 137, "name": "やる気は満点+"}),
            LegalAction(choice_id=2, action_type="end_turn", description="ターン終了", bbox=box, payload={"name": "SKIP"}),
        ]
        # Train planner once, then strip 171/170 from codec to force Tier 1 unmapped path
        planner._train_for_battle(obs_star, tr)
        assert planner._codec is not None
        for k_del in [k for k in list(planner._codec.card_key_to_action) if k[0] in (170, 171)]:
            planner._codec.card_key_to_action.pop(k_del, None)
        planner._codec.card_id_to_fallback_action.pop(170, None)
        planner._codec.card_id_to_fallback_action.pop(171, None)

        cid_tier1, reason_tier1, _ = planner.decide_battle_action(obs_star, acts_star, tracker=tr)
        assert cid_tier1 == 0
        assert "unmapped extra-turn priority" in reason_tier1

        # Tier 2: #521 夢色リップ+ was NOT in initial deck or pre-registered; unmapped fallback evaluates it over 眠気/END_TURN
        obs_yume = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=7,
            turn=4,
            turns_remaining=6,
            stamina=32,
            max_stamina=44,
            score=25000,
            buffs={"goodImpressionTurns": 24, "motivation": 8},
            cards=[
                RecognizedCard(card_id=521, name="夢色リップ+", upgraded=True, playable=True, visual_playable=True, bbox=box),
                RecognizedCard(card_id=23, name="眠気", upgraded=False, playable=True, visual_playable=True, bbox=box),
            ],
        )
        acts_yume = [
            LegalAction(choice_id=0, action_type="play_card", description="夢色リップ+", bbox=box, payload={"card_id": 521, "name": "夢色リップ+"}),
            LegalAction(choice_id=1, action_type="play_card", description="眠気", bbox=box, payload={"card_id": 23, "name": "眠気"}),
            LegalAction(choice_id=2, action_type="end_turn", description="ターン終了", bbox=box, payload={"name": "SKIP"}),
        ]
        cid_tier2, reason_tier2, _ = planner.decide_battle_action(obs_yume, acts_yume, tracker=tr)
        assert cid_tier2 == 0, f"Expected Tier 2 unmapped #521 over 眠気/END_TURN, got choice_id={cid_tier2} ({reason_tier2})"
        assert "Unmapped fallback simQ=" in reason_tier2
    finally:
        db.close()


def test_shop_p_points_truncation_guard_and_settle_recapture(monkeypatch):
    """_extract_header_meta rejects truncated P-points (140->1, 255->25) while allowing real lows (105->5, 100->0), and GakumasRunner settles shop 140->1->130."""
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        frame = np.full((1600, 720, 3), 240, dtype=np.uint8)
        hp_lines = [{"text": "Pポイント 36/44", "bbox": [250, 32, 330, 58]}]
        cur_val = {"v": 140}
        monkeypatch.setattr(rec, "_ocr_bw_banner_number", lambda *_a, **_kw: cur_val["v"])
        monkeypatch.setattr(rec, "_ocr_crop_upscaled", lambda *_a, **_kw: "")

        for raw_v, exp_v in ((140, 140), (1, 140), (255, 255), (25, 255), (105, 105), (5, 5), (100, 100), (0, 0)):
            cur_val["v"] = raw_v
            assert rec._parse_week_and_stats(hp_lines, frame)["p_points"] == exp_v

        # Runner settle & re-capture on abrupt shop P-point drop (140 -> 1 -> 130)
        box = BoundingBox(xmin=100, ymin=1100, xmax=300, ymax=1200)
        exit_box = BoundingBox(xmin=520, ymin=1320, xmax=680, ymax=1390)
        clicks = []
        screencaps = {"n": 0}

        class _StubShopCtrl:
            def __init__(self):
                self.is_connected = True

            def connect(self):
                return True

            def screencap(self):
                screencaps["n"] += 1
                return frame

            def click(self, x, y):
                clicks.append((x, y))
                return True

        obs_drop = ScreenObservation(
            scene_type="shop",
            plan="logic",
            scenario_id="hif_honsen",
            week=8,
            stamina=36,
            max_stamina=44,
            p_points=1,
            options=[
                {"choice_id": 0, "name": "リフレッシュ", "title": "リフレッシュ (10P)", "shop_slot_key": "リフレッシュ", "price": 10, "bbox": box.model_dump()},
                {"choice_id": 1, "name": "終了", "title": "終了", "bbox": exit_box.model_dump()},
            ],
        )
        obs_settled = ScreenObservation(
            scene_type="shop",
            plan="logic",
            scenario_id="hif_honsen",
            week=8,
            stamina=36,
            max_stamina=44,
            p_points=130,
            options=[
                {"choice_id": 0, "name": "リフレッシュ", "title": "リフレッシュ (10P)", "shop_slot_key": "リフレッシュ", "price": 10, "bbox": box.model_dump()},
                {"choice_id": 1, "name": "終了", "title": "終了", "bbox": exit_box.model_dump()},
            ],
        )
        inner_reco = SimpleNamespace(_last_p_points=1)
        recog_calls = {"n": 0}

        def _fake_shop_recognize(_fr):
            recog_calls["n"] += 1
            if recog_calls["n"] == 1:
                inner_reco._last_p_points = 1
                return obs_drop
            assert inner_reco._last_p_points == 140
            inner_reco._last_p_points = 130
            return obs_settled

        agent = LocalHybridDecisionAgent(db=db, planner_cfg=LiveBattlePlannerConfig(ppo_episodes=4, ismcts_simulations=4))
        runner = GakumasRunner(
            mode="adb",
            controller=_StubShopCtrl(),
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        runner._shop_settle_sleep = 0.0
        runner._last_stable_shop_p_points = 140
        runner.scene_routing_reco = SimpleNamespace(recognize=_fake_shop_recognize, recognizer=inner_reco)
        monkeypatch.setattr("time.sleep", lambda *_a: None)

        res = runner.run_once()
        assert screencaps["n"] == 2
        assert res.observation is not None and res.observation.p_points == 130
        assert res.action_taken is not None and res.action_taken.payload.get("name") == "リフレッシュ"
    finally:
        db.close()


def test_aojiru_drink19_banned_and_auto_discarded_across_scenes(monkeypatch):
    """#19 初星スペシャル青汁 is never chosen/bought/used, auto-discarded via 捨てる everywhere, and filtered in battle."""
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(db=db, planner_cfg=LiveBattlePlannerConfig(ppo_episodes=4, ismcts_simulations=4))
        box = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)
        tray_box = BoundingBox(xmin=180, ymin=1490, xmax=240, ymax=1550)

        assert agent._score_drink(drink_id=19, title="初星スペシャル青汁", desc="", target_effect="goodImpressionTurns", plan="logic") <= -100_000
        assert agent._score_drink(drink_id=None, title="初星スペシャル青汁", desc="ランダムな強化済みスキルカード（SSR）を、手札に生成", target_effect="goodImpressionTurns", plan="logic") <= -100_000

        # 1. Reward: never pick 青汁 when alternatives or 受け取らない exist
        obs_rw = ScreenObservation(scene_type="card_reward", plan="logic", scenario_id="hif_honsen", week=3, raw_text="Pドリンクを選択")
        acts_rw = [
            LegalAction(choice_id=0, action_type="pick_option", description="受け取る 《初星スペシャル青汁》", bbox=box, payload={"name": "受け取る", "drink_id": 19, "title": "初星スペシャル青汁"}),
            LegalAction(choice_id=1, action_type="pick_option", description="预览候选饮料1", bbox=box, payload={"name": "预览候选饮料1", "slot_idx": 0, "drink_id": 19, "title": "初星スペシャル青汁"}),
            LegalAction(choice_id=2, action_type="pick_option", description="预览候选饮料2", bbox=box, payload={"name": "预览候选饮料2", "slot_idx": 1, "drink_id": 19, "title": "初星スペシャル青汁"}),
            LegalAction(choice_id=3, action_type="pick_option", description="受け取らない", bbox=box, payload={"name": "受け取らない", "title": "受け取らない"}),
        ]
        dec_rw = agent.decide("", acts_rw, obs=obs_rw)
        assert dec_rw.choice_id == 3

        # 2. Held 青汁 in bottom tray triggers 破棄候補_19 on schedule -> single tap in runner + 捨てる in popup
        clicks = []

        class _StubSchedCtrl:
            is_connected = True

            def connect(self):
                return True

            def click(self, x, y):
                clicks.append((x, y))
                return True

        aojiru_drink = RecognizedDrink(drink_id=19, name="初星スペシャル青汁", playable=True, bbox=tray_box, dsl_summary="ランダムな強化済みスキルカード（SSR）を、手札に生成")
        obs_sched = ScreenObservation(
            scene_type="produce_schedule",
            plan="logic",
            scenario_id="hif_honsen",
            week=4,
            drinks=[aojiru_drink],
            options=[
                {"choice_id": 0, "name": "Vo.SPレッスン", "title": "Vo.SPレッスン", "is_sp": True, "bbox": box.model_dump()},
                {"choice_id": 1, "drink_id": 19, "name": "破棄候補_19", "title": "点击饮料栏旧饮料2《初星スペシャル青汁》打开详情删除(捨てる/破棄)", "bbox": tray_box.model_dump()},
            ],
        )
        runner = GakumasRunner(
            mode="adb",
            controller=_StubSchedCtrl(),
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: obs_sched)
        runner._ctrl_lock = __import__("threading").Lock()
        _StubSchedCtrl.screencap = lambda self: np.zeros((1600, 720, 3), dtype=np.uint8)
        monkeypatch.setattr("time.sleep", lambda *_a: None)
        res_sched = runner.run_once()
        assert res_sched.action_taken is not None and res_sched.action_taken.payload.get("name") == "破棄候補_19"
        assert len(clicks) == 1  # single tap only, never double-tapped

        obs_popup_del = ScreenObservation(scene_type="popup", plan="logic", raw_text="Pドリンク詳細 初星スペシャル青汁 捨てる 閉じる")
        acts_popup_del = [
            LegalAction(choice_id=0, action_type="pick_option", description="捨てる (确认删除该旧P饮料)", bbox=box, payload={"name": "捨てる", "title": "捨てる"}),
            LegalAction(choice_id=1, action_type="pick_option", description="閉じる", bbox=box, payload={"name": "閉じる", "title": "閉じる"}),
        ]
        assert agent.decide("", acts_popup_del, obs=obs_popup_del).choice_id == 0

        obs_popup_conf = ScreenObservation(scene_type="popup", plan="logic", raw_text="このPドリンクを捨てますか？ 初星スペシャル青汁 決定 キャンセル")
        acts_popup_conf = [
            LegalAction(choice_id=0, action_type="pick_option", description="決定", bbox=box, payload={"name": "決定", "title": "決定"}),
            LegalAction(choice_id=1, action_type="pick_option", description="キャンセル", bbox=box, payload={"name": "キャンセル", "title": "キャンセル"}),
        ]
        assert agent.decide("", acts_popup_conf, obs=obs_popup_conf).choice_id == 0

        # If a popup ever offers 使う/交換する for 青汁 without 捨てる, refuse via 閉じる
        obs_popup_use = ScreenObservation(scene_type="popup", plan="logic", raw_text="Pドリンク詳細 初星スペシャル青汁 使う 閉じる")
        acts_popup_use = [
            LegalAction(choice_id=0, action_type="pick_option", description="使う", bbox=box, payload={"name": "使う", "title": "使う"}),
            LegalAction(choice_id=1, action_type="pick_option", description="閉じる", bbox=box, payload={"name": "閉じる", "title": "閉じる"}),
        ]
        assert agent.decide("", acts_popup_use, obs=obs_popup_use).choice_id == 1

        # 3. Shop: filters #19 preview/exchange and discards held #19 first
        obs_shop = ScreenObservation(
            scene_type="shop",
            plan="logic",
            scenario_id="hif_honsen",
            week=8,
            p_points=200,
            drinks=[],
        )
        acts_shop = [
            LegalAction(choice_id=0, action_type="pick_option", description="交換する 初星スペシャル青汁 (-100P)", bbox=box, payload={"name": "交換する", "drink_id": 19, "price": 100, "title": "交換する《初星スペシャル青汁》"}),
            LegalAction(choice_id=1, action_type="pick_option", description="切换预览 Pドリンク #1 《初星スペシャル青汁》 (100P)", bbox=box, payload={"name": "Pドリンク1", "shop_slot_key": "Pドリンク:0", "drink_id": 19, "price": 100, "title": "切换预览 Pドリンク #1 《初星スペシャル青汁》 (100P)"}),
            LegalAction(choice_id=2, action_type="pick_option", description="終了", bbox=box, payload={"name": "終了", "title": "終了"}),
        ]
        assert agent.decide("", acts_shop, obs=obs_shop).choice_id == 2

        acts_shop_discard = [
            LegalAction(choice_id=0, action_type="pick_option", description="破棄候補_19 初星スペシャル青汁", bbox=tray_box, payload={"name": "破棄候補_19", "drink_id": 19}),
            LegalAction(choice_id=1, action_type="pick_option", description="終了", bbox=box, payload={"name": "終了"}),
        ]
        obs_shop_held = ScreenObservation(scene_type="shop", plan="logic", scenario_id="hif_honsen", week=8, p_points=200, drinks=[aojiru_drink])
        assert agent.decide("", acts_shop_discard, obs=obs_shop_held).choice_id == 0

        # Recognizer helper exposes 破棄候補_19 cleanly
        opts: list = [{"choice_id": 0, "name": "終了", "bbox": box.model_dump()}]
        LiveScreenRecognizer._append_banned_drink_discards(opts, [aojiru_drink])
        assert any(o.get("name") == "破棄候補_19" for o in opts)

        # 4. Battle planner: never trains/syncs/maps/uses #19
        tr = GameStateTracker(plan="logic")
        tr.p_idol_id = 163
        tr.recommended_effect = "goodImpressionTurns"
        tr.produce_deck = [920, 922, 127, 129, 137, 175, 297, 711, 159, 173, 778, 617, 171]
        planner = LiveBattlePlanner(db=db, cfg=LiveBattlePlannerConfig(ppo_episodes=4, ismcts_simulations=4))
        obs_bat = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=7,
            turn=1,
            turns_remaining=9,
            stamina=36,
            max_stamina=44,
            drinks=[aojiru_drink],
            cards=[RecognizedCard(card_id=127, name="ファンシーチャーム+", upgraded=True, playable=True, visual_playable=True, bbox=box)],
        )
        acts_bat = [
            LegalAction(choice_id=0, action_type="use_drink", description="使用饮料: 初星スペシャル青汁", bbox=tray_box, payload={"drink_id": 19, "name": "初星スペシャル青汁"}),
            LegalAction(choice_id=1, action_type="play_card", description="ファンシーチャーム+", bbox=box, payload={"card_id": 127, "name": "ファンシーチャーム+"}),
            LegalAction(choice_id=2, action_type="end_turn", description="ターン終了", bbox=box, payload={"name": "SKIP"}),
        ]
        chosen_bat, _reason, _dbg = planner.decide_battle_action(obs_bat, acts_bat, tracker=tr)
        assert chosen_bat != 0
        assert all(d.drink_id != 19 for d in planner._sim.available_drinks)
        assert 19 not in planner._codec.drink_id_to_action
    finally:
        db.close()


def test_hatsuboshi_drink25_strict_active_only_gating_and_followup():
    """#25 特製ハツボシエキス is blocked before free Mental draws or when post-drink cost makes Active unaffordable, and commits to playing an Active card once armed."""
    from demo.sim.battle_simulator import CardInstance, HeadlessBattleSimulator, _make_card_config, _make_drink_config
    from demo.sim.ismcts import ISMCTSAgent, ISMCTSConfig
    from demo.sim.maskable_ppo import PolicyValueNet
    from demo.sim.rl_env import GakumasActionCodec, GakumasBattleEnv, GakumasFeatureEncoder

    db = GakumasDB("gakumas_library.db")
    try:
        box = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)
        deck_ids = [920, 922, 137, 99, 520, 305, 778, 97, 175, 299]
        cfg = build_live_battle_config(
            db=db,
            deck_card_ids=deck_ids,
            equipped_p_items=[],
            drinks=[
                RecognizedDrink(drink_id=25, name="特製ハツボシエキス", playable=True, bbox=box),
                RecognizedDrink(drink_id=5, name="ルイボスティー", playable=True, bbox=box),
            ],
            idol_stats={"vocal": 2800, "dance": 1200, "visual": 2100},
            stamina=30,
            max_stamina=44,
            total_turns=12,
            week=9,
        )
        codec = GakumasActionCodec(cfg, db=db)
        d25_aid = codec.drink_id_to_action[25]
        d5_aid = codec.drink_id_to_action[5]

        def _put_hand(s_obj, card_ids):
            out = []
            for cid_h in card_ids:
                uid_h = s_obj.next_uid
                s_obj.next_uid += 1
                inst_h = CardInstance(uid=uid_h, cfg=_make_card_config(db, cid_h))
                s_obj._all_instances[uid_h] = inst_h
                out.append(inst_h)
            s_obj.hand = out
            return out

        # 1. Free Mental (#99 アイドル宣言+) + Active (#922 認めてくれるひと+) -> #25 gated OUT until #99 plays
        sim = HeadlessBattleSimulator(cfg, db=db, seed=101, record_logs=False)
        sim.draw_pile = []
        sim.discard_pile = []
        sim.exhaust_pile = []
        sim.buffs["goodImpressionTurns"] = 20
        sim.buffs["motivation"] = 10
        sim.genki = 20
        c_99, c_922, c_520 = _put_hand(sim, [99, 922, 520])
        dr25 = sim.available_drinks[next(i for i, d in enumerate(sim.available_drinks) if d.drink_id == 25)]
        assert not sim.can_use_active_retrigger_drink_now(dr25)
        mask_free, _ = codec.legal_mask_and_map(sim, prune_dominated=True)
        assert not mask_free[d25_aid]
        # Base unpruned bijection mask stays unaffected
        mask_raw, _ = codec.legal_mask_and_map(sim, prune_dominated=False)
        assert mask_raw[d25_aid]

        # 2. Low stamina + zero genki where #25 (-2 fixed HP + doubleCostTurns) makes #922 (6 -> 12) unaffordable -> #25 blocked
        sim_low = HeadlessBattleSimulator(cfg, db=db, seed=102, record_logs=False)
        sim_low.draw_pile = []
        sim_low.discard_pile = []
        sim_low.stamina = 8
        sim_low.genki = 0
        sim_low.buffs["nullifyDebuff"] = 0
        sim_low.buffs["doubleCostTurns"] = 0
        _put_hand(sim_low, [922])
        dr25_low = sim_low.available_drinks[next(i for i, d in enumerate(sim_low.available_drinks) if d.drink_id == 25)]
        assert not sim_low.can_use_active_retrigger_drink_now(dr25_low)

        # 3. With Active #305 + Mental #520 (no free Mental cards) -> #25 permitted; after drinking #25, must play Active (#305/#922), not #520 or END_TURN
        sim_act = HeadlessBattleSimulator(cfg, db=db, seed=103, record_logs=False)
        sim_act.draw_pile = []
        sim_act.discard_pile = []
        sim_act.buffs["goodImpressionTurns"] = 20
        sim_act.buffs["motivation"] = 10
        sim_act.genki = 20
        c_305, c_520_b = _put_hand(sim_act, [305, 520])
        idx_25 = next(i for i, d in enumerate(sim_act.available_drinks) if d.drink_id == 25)
        assert sim_act.can_use_active_retrigger_drink_now(sim_act.available_drinks[idx_25])

        # Arm #25 now and verify follow-up restriction across codec, ISMCTS, and is_safe_dominating_free_card
        sim_act.step("drink", idx_25)
        assert sim_act.has_armed_active_retrigger(same_turn_only=True)
        # Double #25 trigger stacking also blocked
        sim_act.available_drinks.append(_make_drink_config(db, 25))
        assert not sim_act.can_use_active_retrigger_drink_now(sim_act.available_drinks[-1])

        mask_armed, map_armed = codec.legal_mask_and_map(sim_act, prune_dominated=True)
        c520_aid = codec.card_slot_index(c_520_b.cfg)
        c305_aid = codec.card_slot_index(c_305.cfg)
        assert mask_armed[c305_aid]
        assert mask_armed[d5_aid]
        assert not mask_armed[c520_aid]
        assert not mask_armed[codec.end_turn_action_id]
        assert not mask_armed[d25_aid]

        # When #25 is armed via trigger (even before double_card_effect_cards > 0), #97 free Active does not auto-prune #305/#778
        c_97, c_778, c_305_b = _put_hand(sim_act, [97, 778, 305])
        assert not sim_act.is_safe_dominating_free_card(c_97)

        enc = GakumasFeatureEncoder(codec)
        net = PolicyValueNet(enc.obs_dim, codec.num_actions, hidden_dim=32, seed=5)
        mcts = ISMCTSAgent(codec, enc, net, ISMCTSConfig(num_simulations=4))
        best_aid, _ = mcts.select_action(sim_act, __import__("random").Random(7))
        assert best_aid in (
            codec.card_slot_index(c_97.cfg),
            codec.card_slot_index(c_778.cfg),
            codec.card_slot_index(c_305_b.cfg),
            d5_aid,
        )

        # 4. LiveBattlePlanner integration: free Mental #99 + Active #922 blocks #25 at root; armed #25 commits to Active #305/#922
        planner = LiveBattlePlanner(db=db, cfg=LiveBattlePlannerConfig(ppo_episodes=4, ismcts_simulations=4))
        tr = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tr.produce_deck = deck_ids
        tr.idol_stats = {"vocal": 2800, "dance": 1200, "visual": 2100}
        box = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)
        drinks_obs = [
            RecognizedDrink(drink_id=25, name="特製ハツボシエキス", playable=True, bbox=box),
        ]
        obs_pre = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=9,
            turn=2,
            turns_remaining=11,
            stamina=30,
            max_stamina=44,
            genki=20,
            score=11200,
            buffs={"goodImpressionTurns": 20, "motivation": 10},
            drinks=drinks_obs,
            cards=[
                RecognizedCard(card_id=99, name="アイドル宣言+", upgraded=True, playable=True, visual_playable=True, bbox=box),
                RecognizedCard(card_id=922, name="認めてくれるひと+", upgraded=True, playable=True, visual_playable=True, bbox=box),
            ],
        )
        acts_pre = [
            LegalAction(choice_id=0, action_type="use_drink", description="使用饮料: 特製ハツボシエキス", bbox=box, payload={"drink_id": 25, "name": "特製ハツボシエキス"}),
            LegalAction(choice_id=1, action_type="play_card", description="アイドル宣言+", bbox=box, payload={"card_id": 99, "name": "アイドル宣言+"}),
            LegalAction(choice_id=2, action_type="play_card", description="認めてくれるひと+", bbox=box, payload={"card_id": 922, "name": "認めてくれるひと+"}),
            LegalAction(choice_id=3, action_type="end_turn", description="ターン終了", bbox=box, payload={"name": "SKIP"}),
        ]
        captured_roots: list[Optional[set[int]]] = []
        orig_sel = None
        planner._train_for_battle(obs_pre, tr)
        orig_sel = planner._ismcts.select_action

        def _cap_root_pre(s_arg, r_arg, allowed_root_actions=None):
            captured_roots.append(set(allowed_root_actions) if allowed_root_actions is not None else None)
            return orig_sel(s_arg, r_arg, allowed_root_actions=allowed_root_actions)

        planner._ismcts.select_action = _cap_root_pre
        try:
            cid_pre, _r_pre, _ = planner.decide_battle_action(obs_pre, acts_pre, tracker=tr)
        finally:
            planner._ismcts.select_action = orig_sel
        assert cid_pre != 0
        p_d25_aid = planner._codec.drink_id_to_action[25]
        assert captured_roots and captured_roots[0] is not None and p_d25_aid not in captured_roots[0]

        # Now arm #25 in planner._sim and present Active #305 + Mental #520 -> planner must restrict root to Active #305
        obs_follow = obs_pre.model_copy(
            update={
                "cards": [
                    RecognizedCard(card_id=305, name="輝くキミへ+", upgraded=True, playable=True, visual_playable=True, bbox=box),
                    RecognizedCard(card_id=520, name="夢色リップ", upgraded=False, playable=True, visual_playable=True, bbox=box),
                ],
                "drinks": [],
            }
        )
        acts_follow = [
            LegalAction(choice_id=0, action_type="play_card", description="輝くキミへ+", bbox=box, payload={"card_id": 305, "name": "輝くキミへ+"}),
            LegalAction(choice_id=1, action_type="play_card", description="夢色リップ", bbox=box, payload={"card_id": 520, "name": "夢色リップ"}),
            LegalAction(choice_id=2, action_type="end_turn", description="ターン終了", bbox=box, payload={"name": "SKIP"}),
        ]
        planner._sim.step("drink", 0)
        assert planner._sim.has_armed_active_retrigger(same_turn_only=True)
        cid_follow, _r_follow, _ = planner.decide_battle_action(obs_follow, acts_follow, tracker=tr)
        assert cid_follow == 0
    finally:
        db.close()
