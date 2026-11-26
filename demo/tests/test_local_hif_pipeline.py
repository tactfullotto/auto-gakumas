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
        # 1a. When any stat is below MIN_STAT_FLOOR (1150), 授業 prioritizes raising that lowest stat (Da=1100),
        # while レッスン & 公開レッスン keep targeting SP or highest uncapped stat (Vo=2400).
        obs_below_floor = ScreenObservation(
            scene_type="produce_schedule",
            plan="logic",
            scenario_id="hif_honsen",
            week=3,
            stamina=40,
            max_stamina=44,
            idol_stats={"vocal": 2400, "dance": 1100, "visual": 1800},
        )
        jugyo_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="Vo.授業", bbox=BoundingBox(xmin=50, ymin=1000, xmax=200, ymax=1180), payload={"name": "Vo.授業"}),
            LegalAction(choice_id=1, action_type="pick_option", description="Da.授業", bbox=BoundingBox(xmin=250, ymin=1000, xmax=400, ymax=1180), payload={"name": "Da.授業"}),
            LegalAction(choice_id=2, action_type="pick_option", description="Vi.授業", bbox=BoundingBox(xmin=450, ymin=1000, xmax=600, ymax=1180), payload={"name": "Vi.授業"}),
        ]
        d_jugyo_floor = agent.decide("", jugyo_acts, obs=obs_below_floor)
        assert d_jugyo_floor.choice_id == 1

        # Even if multiple stats < 1150, 授業 picks the strictly lowest stat first (Vo=896 over Vi=1050)
        obs_multi_low = obs_below_floor.model_copy(
            update={"idol_stats": {"vocal": 896, "dance": 2112, "visual": 1050}}
        )
        assert agent.decide("", jugyo_acts, obs=obs_multi_low).choice_id == 0

        # 1b. Once all 3 stats >= 1150, 授業 falls back to non-highest stat (Vi=1800 over Vo=2400)
        obs_above_floor = obs_below_floor.model_copy(
            update={"idol_stats": {"vocal": 2400, "dance": 1200, "visual": 1800}}
        )
        d_jugyo = agent.decide("", jugyo_acts, obs=obs_above_floor)
        assert d_jugyo.choice_id == 2

        # 2. レッスン & 公開レッスン (% multiplier) ignore <1150 patching and pick highest uncapped stat (Vo=2400) or SP
        lesson_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="Vo.レッスン", bbox=BoundingBox(xmin=50, ymin=1000, xmax=200, ymax=1180), payload={"name": "Vo.レッスン"}),
            LegalAction(choice_id=1, action_type="pick_option", description="Da.レッスン", bbox=BoundingBox(xmin=250, ymin=1000, xmax=400, ymax=1180), payload={"name": "Da.レッスン"}),
            LegalAction(choice_id=2, action_type="pick_option", description="Vi.レッスン", bbox=BoundingBox(xmin=450, ymin=1000, xmax=600, ymax=1180), payload={"name": "Vi.レッスン"}),
        ]
        d_lesson = agent.decide("", lesson_acts, obs=obs_below_floor)
        assert d_lesson.choice_id == 0

        koukai_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="Vo.公開レッスン", bbox=BoundingBox(xmin=50, ymin=1000, xmax=200, ymax=1180), payload={"name": "Vo.公開レッスン"}),
            LegalAction(choice_id=1, action_type="pick_option", description="Da.公開レッスン", bbox=BoundingBox(xmin=250, ymin=1000, xmax=400, ymax=1180), payload={"name": "Da.公開レッスン"}),
            LegalAction(choice_id=2, action_type="pick_option", description="Vi.公開レッスン", bbox=BoundingBox(xmin=450, ymin=1000, xmax=600, ymax=1180), payload={"name": "Vi.公開レッスン"}),
        ]
        d_koukai = agent.decide("", koukai_acts, obs=obs_below_floor)
        assert d_koukai.choice_id == 0

        koukai_sp_acts = [
            koukai_acts[0],
            koukai_acts[1],
            LegalAction(choice_id=2, action_type="pick_option", description="Vi.SP公開レッスン", bbox=BoundingBox(xmin=450, ymin=1000, xmax=600, ymax=1180), payload={"name": "Vi.SP公開レッスン", "is_sp": True}),
        ]
        assert agent.decide("", koukai_sp_acts, obs=obs_below_floor).choice_id == 2

        # 3a. Clean 授業 event raises <1150 stat (Da=1100) over >=1150 stats
        ev_obs = ScreenObservation(
            scene_type="produce_event",
            plan="logic",
            scenario_id="hif_honsen",
            week=3,
            week_activity="授業",
            raw_text="授業",
            idol_stats={"vocal": 2400, "dance": 1100, "visual": 1800},
        )
        ev_clean_acts = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="Visual(Vi)属性强化 | パラメータ+40",
                bbox=BoundingBox(xmin=60, ymin=900, xmax=660, ymax=980),
                payload={"name": "Vi パラメータ+40"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="Dance(Da)属性强化 | パラメータ+40",
                bbox=BoundingBox(xmin=60, ymin=1010, xmax=660, ymax=1090),
                payload={"name": "Da パラメータ+40"},
            ),
        ]
        assert agent.decide("", ev_clean_acts, obs=ev_obs).choice_id == 1

        # 3b. 授業 event strongly avoids adding トラブル / 眠気 even for a <1150 stat
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
        # Post-Battle-1 W8 final shop: fill all 4 drinks FIRST (Pドリンク 2400 > リフレッシュ 2100); no カスタマイズ until 4 drinks full!
        assert agent.decide("", acts_refresh_vs_drink_upg, obs=w8_obs_rich).choice_id == 1
        agent._shop_clicked_slots.clear()
        agent._shop_noop_features.clear()
        # Once all 4 drinks are filled, カスタマイズ (2250) beats リフレッシュ (1350) and 強化 (950)
        four_drinks = [
            RecognizedDrink(drink_id=22, name="初星黒酢", bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)),
            RecognizedDrink(drink_id=18, name="プロテイン", bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)),
            RecognizedDrink(drink_id=14, name="センブリソーダ", bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)),
            RecognizedDrink(drink_id=12, name="ブーストエキス", bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)),
        ]
        w8_obs_full4 = w8_obs_rich.model_copy(update={"drinks": four_drinks})
        assert agent.decide("", acts_refresh_vs_drink_upg, obs=w8_obs_full4).choice_id == 3
        agent._shop_clicked_slots.clear()
        agent._shop_noop_features.clear()
        acts_no_cust = [a for a in acts_refresh_vs_drink_upg if a.choice_id != 3]
        assert agent.decide("", acts_no_cust, obs=w8_obs_rich).choice_id == 1
        agent._shop_clicked_slots.clear()
        agent._shop_noop_features.clear()
        acts_upg_vs_exit = [acts_refresh_vs_drink_upg[0], acts_refresh_vs_drink_upg[2], acts_refresh_vs_drink_upg[5]]
        assert agent.decide("", acts_upg_vs_exit, obs=w8_obs_rich).choice_id == 2
        acts_cust_only = [a for a in acts_refresh_vs_drink_upg if a.choice_id not in (1, 2)]

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
        assert GakumasFeatureEncoder.SCALAR_DIMS == 83

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
        w6_obs = w8_obs.model_copy(update={"week": 6})
        d1 = agent_no_tr.decide("", shop_acts, obs=w6_obs)
        assert d1.choice_id == 0
        # Still in shop at same p_points=250 -> 強化 marked no-op, picks カスタマイズ
        d2 = agent_no_tr.decide("", shop_acts, obs=w6_obs)
        assert d2.choice_id == 1
        # Simulate P-point change after buying a drink -> _shop_clicked_slots clears, yet 強化 stays blocked!
        w6_obs_spent = w6_obs.model_copy(update={"p_points": 150})
        d3 = agent_no_tr.decide("", [shop_acts[0], shop_acts[2]], obs=w6_obs_spent)
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


def test_produce_preferences_card_weights_and_custom_banned_drinks():
    """Verify user card priorities/avoids/keeps, custom banned drinks, and shop/reroll toggles propagate end-to-end."""
    from demo.core.hif_catalog import ProducePreferences
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. Agent picks user-prioritized #64 over default higher-scoring card #521, and avoids #711
        prefs = ProducePreferences(
            priority_card_ids=[64],
            keep_card_ids=[1],
            avoid_card_ids=[711],
            card_score_overrides={68: 2600},
            preferred_drink_ids=[2],
            banned_drink_ids=[19, 1],  # ban 青汁 (#19) and 初星水 (#1)
            spend_all_final_shop=False,
            reroll_early_non_exhaust_mental=False,
        )
        agent = LocalHybridDecisionAgent(db=db, preferences=prefs)
        tracker = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        agent.bind_tracker(tracker)
        box = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)

        # Prioritized #64 beats normal #711 (and #711 is avoided anyway)
        obs_rew = ScreenObservation(scene_type="card_reward", plan="logic", scenario_id="hif_honsen", week=2, stamina=30, max_stamina=40)
        acts_rew = [
            LegalAction(choice_id=0, action_type="pick_option", description="スキルカード1", bbox=box, payload={"card_id": 521, "name": "夢色リップ+"}),
            LegalAction(choice_id=1, action_type="pick_option", description="スキルカード2", bbox=box, payload={"card_id": 64, "name": "カード64"}),
            LegalAction(choice_id=2, action_type="pick_option", description="再抽選", bbox=box, payload={"name": "再抽選"}),
        ]
        dec_rew = agent.decide("", acts_rew, obs=obs_rew)
        assert dec_rew.choice_id == 1

        # Protected keep_card_ids=[1] resists deletion compared to avoided #711
        rem_score_kept = agent._score_skill_card(card_id=1, title="アピールの基本", desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=True)
        rem_score_avoid = agent._score_skill_card(card_id=711, title="冒険心+", desc="", target_effect="goodImpressionTurns", guide=None, plan="logic", for_removal=True)
        assert rem_score_avoid > rem_score_kept

        # Early M-card reroll disabled via reroll_early_non_exhaust_mental=False -> accepts #68 instead of 再抽選
        acts_early = [
            LegalAction(choice_id=0, action_type="pick_option", description="励まし", bbox=box, payload={"card_id": 68, "name": "励まし"}),
            LegalAction(choice_id=1, action_type="pick_option", description="再抽選", bbox=box, payload={"name": "再抽選"}),
        ]
        dec_early = agent.decide("", acts_early, obs=obs_rew)
        assert dec_early.choice_id == 0

        # Final shop refresh disabled via spend_all_final_shop=False -> exits instead of リフレッシュ
        obs_shop = ScreenObservation(scene_type="shop", plan="logic", scenario_id="hif_honsen", week=8, total_weeks=9, p_points=120, drinks=[])
        acts_shop = [
            LegalAction(choice_id=0, action_type="pick_option", description="リフレッシュ (10P)", bbox=box, payload={"name": "リフレッシュ", "shop_slot_key": "リフレッシュ", "price": 10}),
            LegalAction(choice_id=1, action_type="pick_option", description="終了", bbox=box, payload={"name": "終了"}),
        ]
        dec_shop = agent.decide("", acts_shop, obs=obs_shop)
        assert dec_shop.choice_id == 1

        # Preferred drink #2 beats #5, while custom banned drink #1 初星水 is rejected & discarded
        assert agent._score_drink(drink_id=2, title="烏龍茶", desc="", target_effect="goodImpressionTurns", plan="logic") > agent._score_drink(drink_id=5, title="ルイボスティー", desc="", target_effect="goodImpressionTurns", plan="logic")
        assert agent._is_banned_drink(1, "初星水") is True
        assert agent._score_drink(drink_id=1, title="初星水", desc="", target_effect="goodImpressionTurns", plan="logic") <= -100_000

        # LiveScreenRecognizer generates 破棄候補_1 when drink #1 is banned via set_banned_drink_ids
        reco = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        reco.set_banned_drink_ids([19, 1])
        opts_disc: list[dict] = [{"choice_id": 0, "name": "終了", "bbox": box.model_dump()}]
        LiveScreenRecognizer._append_banned_drink_discards(
            opts_disc,
            [RecognizedDrink(drink_id=1, name="初星水", playable=True, bbox=box, dsl_summary="パラメータ+10")],
            banned_ids=reco.banned_drink_ids,
            banned_names=reco._banned_drink_names,
        )
        assert any(o["name"] == "破棄候補_1" for o in opts_disc)
        agent.battle_planner.set_banned_drink_ids([19, 1])
        assert 1 in agent.battle_planner.banned_drink_ids
    finally:
        db.close()


def test_manual_hif_memory_hooks_extra_use_limit_and_draw_in_battle_sim():
    """Verify extra_use_limit=2, draw_on_play=1, memory_passives, and manual/hybrid build_local_hif_runner."""
    from demo.run_hif_local import build_local_hif_runner
    from demo.sim.battle_simulator import HeadlessBattleSimulator
    from demo.sim.live_battle_planner import build_live_battle_config

    db = GakumasDB("gakumas_library.db")
    try:
        # #89 ラブリーウインク+ (no built-in extra use) configured with extra_use_limit=2 & draw_on_play=1;
        # #127 ファンシーチャーム+ explicitly set force_initial_hand=False
        cfg = build_live_battle_config(
            db=db,
            deck_card_ids=[89, 127, 920, 922, 137, 171],
            p_idol_id=163,
            plan="logic",
            buffs={},
            card_memory_hooks={
                "89": {"free_extra_use_first_play": True, "extra_use_limit": 2, "draw_on_play": 1},
                "127": {"force_initial_hand": False},
            },
            memory_passives=[
                {"key": "start_good_impression", "value": 3},
                {"key": "start_motivation", "value": 4},
                {"key": "start_genki", "value": 5},
            ],
        )
        assert cfg.initial_good_impression >= 3
        assert cfg.initial_motivation >= 4
        assert cfg.initial_genki >= 5
        c89_cfg = next(c for c in cfg.deck if c.card_id == 89)
        c127_cfg = next(c for c in cfg.deck if c.card_id == 127)
        assert c89_cfg.memory_extra_use_limit == 2
        assert c89_cfg.memory_draw_on_play == 1
        assert c127_cfg.force_initial_hand is False

        sim = HeadlessBattleSimulator(cfg, db=db, seed=11, record_logs=True)
        sim.buffs["motivation"] = 20
        # Put #89 in hand and fillers in draw_pile so draw_on_play=1 is observable
        sim.draw_pile = [c for c in list(sim.hand) + list(sim.draw_pile) if c.cfg.card_id != 89]
        c89 = next(c for c in sim._all_instances.values() if c.cfg.card_id == 89)
        sim.hand = [c89]
        if c89 in sim.draw_pile:
            sim.draw_pile.remove(c89)
        sim.card_uses_remaining = 1

        # Play 1 (free extra use 1/2 + draws 1 card)
        sim.play_card(0)
        assert sim.card_uses_remaining == 1
        assert len(sim.hand) == 1
        assert sim.memory_extra_use_counts.get(c89.cfg.base_id) == 1

        # Move #89 back to hand and Play 2 (free extra use 2/2 + draws 1 card)
        sim.discard_pile.remove(c89)
        sim.hand.insert(0, c89)
        hand_before_2 = len(sim.hand)
        sim.play_card(0)
        assert sim.card_uses_remaining == 1
        assert len(sim.hand) == hand_before_2  # popped 1, drew 1
        assert sim.memory_extra_use_counts.get(c89.cfg.base_id) == 2

        # Move #89 back for Play 3 -> memory limit exhausted, consumes last card_uses_remaining -> end_turn
        sim.discard_pile.remove(c89)
        sim.hand.insert(0, c89)
        turn_before_3 = sim.turn
        sim.play_card(0)
        assert sim.turn == turn_before_3 + 1

        # Verify manual vs hybrid runner construction
        r_manual = build_local_hif_runner(
            plan="sense",
            p_idol_id=158,
            loadout_mode="manual",
            initial_loadout_dict={"p_idol_id": 158, "plan": "sense", "deck_ids": [884, 83], "p_items": []},
            preferences={"banned_drink_ids": [19, 3]},
        )
        try:
            assert r_manual.loadout_scanner is None
            assert r_manual.tracker.initial_loadout_captured is True
            assert r_manual.tracker.produce_deck == [884, 83]
            assert 3 in r_manual.agent.preferences.banned_drink_ids
        finally:
            r_manual.agent.close()
            r_manual.db.close()

        r_hybrid = build_local_hif_runner(
            plan="logic",
            p_idol_id=163,
            loadout_mode="hybrid",
            initial_loadout_dict={
                "p_idol_id": 163,
                "plan": "logic",
                "card_memory_hooks": {"299": {"free_extra_use_first_play": True, "extra_use_limit": 2}},
            },
        )
        try:
            assert r_hybrid.loadout_scanner is not None
            assert r_hybrid.manual_memory_hooks[299]["extra_use_limit"] == 2
        finally:
            r_hybrid.agent.close()
            r_hybrid.loadout_scanner.close()
            r_hybrid.db.close()
    finally:
        db.close()


def test_card_customizations_applied_in_battle_simulator_and_tracker():
    """Verify official カスタマイズ (#36/#29/#55/#34/#80/#37) & custom_modifiers in HeadlessBattleSimulator and Tracker."""
    from demo.sim.battle_simulator import HeadlessBattleSimulator
    from demo.sim.live_battle_planner import build_live_battle_config

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. #127 ファンシーチャーム+ (base cost=4, GI+=5, limit=1) customized with:
        #    - #36 開始時手札に入る (force_initial_hand=True)
        #    - #29 コスト値- (g.cost+=2 -> cost becomes 2)
        #    - #55 好印象+ Lv2 (g.goodImpressionTurns+=4 -> GI becomes 9)
        #    #299 オトメゴコロ+ (typed cost motivation-=3, GI+=5) customized with:
        #    - #34 やる気コスト値- (g.typedCost+=1 -> motivation cost becomes 2)
        #    - #80 好印象+ (g.goodImpressionTurns+=2 -> base GI becomes 7)
        #    - modifiers.score_delta = 12 (adds direct score action!)
        cfg = build_live_battle_config(
            db=db,
            deck_card_ids=[127, 299, 920, 922, 137, 171],
            p_idol_id=163,
            plan="logic",
            buffs={},
            card_memory_hooks={"127": {"force_initial_hand": False}},
            card_customizations={
                "127": {"levels": {"36": 1, "29": 1, "55": 2}},
                "299": {"levels": {"34": 1, "80": 1}, "modifiers": {"score_delta": 12}},
            },
        )
        c127_cfg = next(c for c in cfg.deck if c.card_id == 127)
        c299_cfg = next(c for c in cfg.deck if c.card_id == 299)
        assert c127_cfg.force_initial_hand is True  # #36 turns on force_initial_hand

        sim = HeadlessBattleSimulator(cfg, db=db, seed=7, record_logs=True)
        c127 = next(c for c in sim._all_instances.values() if c.cfg.card_id == 127)
        c299 = next(c for c in sim._all_instances.values() if c.cfg.card_id == 299)

        # #127 must start in initial hand due to #36, with effective cost 2 (down from 4)
        assert c127 in sim.hand
        assert sim._effective_cost(c127.cfg) == ("cost", 2)

        sim.stamina = 30
        sim.buffs["goodImpressionTurns"] = 0
        sim.card_uses_remaining = 2
        idx_127 = sim.hand.index(c127)
        sim.play_card(idx_127)
        assert sim.stamina == 28  # cost 2
        assert sim.buffs["goodImpressionTurns"] == 9  # 5 + 4 (Lv2 of #55)
        assert c127 in sim.exhaust_pile  # still limit=1 without #37

        # Test #299: motivation cost 2 (down from 3), GI +7 (5+2), plus positive score from score_delta=12
        sim.buffs["motivation"] = 5
        sim.buffs["goodImpressionTurns"] = 0
        score_before = sim.score
        if c299 not in sim.hand:
            if c299 in sim.draw_pile:
                sim.draw_pile.remove(c299)
            sim.hand.insert(0, c299)
        assert sim._effective_cost(c299.cfg) == ("motivation", 2)
        sim.play_card(sim.hand.index(c299))
        assert sim.buffs["motivation"] == 3  # 5 - 2
        assert sim.buffs["goodImpressionTurns"] >= 7
        assert sim.score > score_before

        # 2. #37 (レッスン中1回の制限削除) removes limit_count=1 in both Simulator and Tracker
        cfg_nolim = build_live_battle_config(
            db=db,
            deck_card_ids=[127, 137, 299],
            p_idol_id=163,
            plan="logic",
            card_customizations={"127": {"levels": {"37": 1}}},
        )
        c127_nl_cfg = next(c for c in cfg_nolim.deck if c.card_id == 127)
        assert c127_nl_cfg.limit_count is None
        sim_nl = HeadlessBattleSimulator(cfg_nolim, db=db, seed=3)
        inst_127_nl = next(c for c in sim_nl._all_instances.values() if c.cfg.card_id == 127)
        if inst_127_nl not in sim_nl.hand:
            sim_nl.draw_pile.remove(inst_127_nl)
            sim_nl.hand.insert(0, inst_127_nl)
        sim_nl.card_uses_remaining = 2
        sim_nl.play_card(sim_nl.hand.index(inst_127_nl))
        assert inst_127_nl in sim_nl.discard_pile
        assert inst_127_nl not in sim_nl.exhaust_pile

        tracker = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tracker.set_initial_loadout(
            deck_ids=[127, 299],
            p_items=[],
            card_customizations={127: {"levels": {"37": 1}}},
            db=db,
        )
        tracker.in_battle = True
        tracker.record_card_play(127, db=db)
        assert 127 in tracker.battle_discard_pile
        assert 127 not in tracker.battle_exhaust_pile
    finally:
        db.close()


def test_sense_simulator_score_math_and_p_items():
    from demo.sim.battle_simulator import (
        BattleConfig,
        HeadlessBattleSimulator,
        TurnMultiplierSpec,
        _build_sense_hif_p_items,
        _make_card_config,
        _make_drink_config,
    )

    db = GakumasDB("gakumas_library.db")
    try:
        # 1) #35 ハイタッチ+ with gc=6, conc=6: (23 + 6*2) * 1.5 = 52.5 -> 53
        cfg_hi = BattleConfig(
            scenario_id="hif_honsen",
            stage_name="sense_hi",
            p_idol_id=139,
            idol_name="姫崎 莉波",
            plan="sense",
            idol_stats={"vocal": 1000, "dance": 1000, "visual": 1000},
            initial_stamina=40,
            max_stamina=40,
            initial_good_condition=6,
            initial_concentration=6,
            total_turns=3,
            turn_schedule=[TurnMultiplierSpec("Vo", 1.0)] * 3,
            deck=[_make_card_config(db, 35)] * 6,
        )
        sim_hi = HeadlessBattleSimulator(cfg_hi, db=db, seed=1)
        sim_hi.play_card(0)
        assert sim_hi.score == 53
        assert sim_hi.concentration_multiplier == 1.0

        # 2) #762 脚光+ played twice + #814 自然体の魅力+ with perfectConditionTurns
        # + #22 初星黒酢 (moveSelectedToHand + nullifyCostCards) + #585 夏夜に咲く思い出+ (removeRandom trouble)
        cfg_combo = BattleConfig(
            scenario_id="hif_honsen",
            stage_name="sense_combo",
            p_idol_id=139,
            idol_name="姫崎 莉波",
            plan="sense",
            idol_stats={"vocal": 1000, "dance": 1000, "visual": 1000},
            initial_stamina=40,
            max_stamina=40,
            initial_good_condition=12,
            initial_perfect_condition=2,
            initial_concentration=0,
            total_turns=4,
            turn_schedule=[
                TurnMultiplierSpec("Vi", 1.0),
                TurnMultiplierSpec("Vo", 1.0),
                TurnMultiplierSpec("Da", 1.0),
                TurnMultiplierSpec("Vo", 1.0),
            ],
            deck=[_make_card_config(db, 23)] * 5,
            drinks=[_make_drink_config(db, 22)],
            p_items=[p for p in _build_sense_hif_p_items() if p.item_id in (9139, 9140)],
        )
        sim = HeadlessBattleSimulator(cfg_combo, db=db, seed=2)
        # 9140 包まれるやさしさ triggers on Vi turn 1 when gc >= 10 (+4 perfectCondition, +10 genki)
        assert sim.buffs["perfectConditionTurns"] == 6
        assert sim.genki == 10

        # Stage deterministic hand: [#762, #762, #585, #814], with #23 in discard and #812 in deck
        sim.hand = [
            sim._new_instance(_make_card_config(db, 762)),
            sim._new_instance(_make_card_config(db, 762)),
            sim._new_instance(_make_card_config(db, 585)),
            sim._new_instance(_make_card_config(db, 814)),
        ]
        sim.draw_pile = [sim._new_instance(_make_card_config(db, 812))]
        sim.discard_pile = [sim._new_instance(_make_card_config(db, 23))]
        sim.card_uses_remaining = 1

        # Use #22 初星黒酢 -> moves best/selected card from deck|discard into hand, nullifyCostCards+=2
        hand_before_drink = len(sim.hand)
        sim.use_drink(0)
        assert len(sim.hand) == hand_before_drink + 1
        assert sim.buffs["nullifyCostCards"] >= 2

        # Play first #762 脚光+ (idx 0): costs 2 goodConditionTurns (12->10), concentration +7, uses +1
        sim.play_card(0)
        assert sim.buffs["concentration"] == 7
        assert sim.buffs["goodConditionTurns"] == 10
        assert sim.score == 0

        # Play second #762 脚光+ (idx 0): costs 2 gc (10->8), concentration +7 (=14); 1st subsequent card for 1st passive
        sim.play_card(0)
        assert sim.buffs["concentration"] == 14
        assert sim.buffs["goodConditionTurns"] == 8

        # Play #585 夏夜に咲く思い出+ (idx 0): removes trouble #23 from discard pile AND is 2nd subsequent card
        # -> triggers 1st #762 afterCardUsed (concentrationMultiplier=4, score+=6, gc-=3 -> gc=5)
        # Gain: (6 + 14*4) * (1.5 + 0.1 * 8) = 62 * 2.3 = 142.6 -> 143
        assert any(c.cfg.card_id == 23 for c in sim.discard_pile)
        sim.play_card(0)
        assert sim.score == 143
        assert sim.buffs["goodConditionTurns"] == 5
        assert sim.concentration_multiplier == 1.0
        assert not any(c.cfg.card_id == 23 for c in sim.discard_pile)
        assert any(c.cfg.card_id == 23 for c in sim.exhaust_pile)

        # Boost gc back to 12 and play #814 自然体の魅力+ (idx 0) with perfectConditionTurns > 0
        sim.buffs["goodConditionTurns"] = 12
        score_before_814 = sim.score
        sim.play_card(0)
        assert sim.score > score_before_814

        # Verify 9139 憧れ続けた輝き triggers after 4 goodCondition skill cards while gc >= 8
        cfg_9139 = BattleConfig(
            scenario_id="hif_honsen",
            stage_name="sense_9139",
            p_idol_id=139,
            idol_name="姫崎 莉波",
            plan="sense",
            idol_stats={"vocal": 1000, "dance": 1000, "visual": 1000},
            initial_stamina=30,
            max_stamina=30,
            initial_good_condition=8,
            total_turns=3,
            turn_schedule=[TurnMultiplierSpec("Vo", 1.0)] * 3,
            deck=[_make_card_config(db, 105)] * 6,
            p_items=[p for p in _build_sense_hif_p_items() if p.item_id == 9139],
        )
        sim_p = HeadlessBattleSimulator(cfg_9139, db=db, seed=3)
        sim_p.hand = [sim_p._new_instance(_make_card_config(db, 812)) for _ in range(4)]
        sim_p.draw_pile = [sim_p._new_instance(_make_card_config(db, 35)) for _ in range(3)]
        sim_p.card_uses_remaining = 4
        for _ in range(4):
            sim_p.play_card(0)
        assert sim_p.buffs["perfectConditionTurns"] >= 1
    finally:
        db.close()


def test_sense_build_live_battle_config_flower_robo():
    db = GakumasDB("gakumas_library.db")
    try:
        cfg = build_live_battle_config(
            db=db,
            deck_card_ids=[812, 814, 762, 35, 105, 113],
            equipped_p_items=[
                {"item_id": 436, "canonical_name": "花ロボ(黄)"},
                {"item_id": 9139, "canonical_name": "憧れ続けた輝き"},
                {"item_id": 447, "canonical_name": "H.I.Fワッペン"},
                {"item_id": 9141, "canonical_name": "ひとりだちシューズ"},
                {"item_id": 9140, "canonical_name": "包まれるやさしさ"},
                {"item_id": 9142, "canonical_name": "ほっこりまんぶく"},
            ],
            drinks=[18, 13, 21, 22],
            idol_stats={"vocal": 896, "dance": 2410, "visual": 2233},
            stamina=44,
            max_stamina=44,
            p_idol_id=139,
            plan="sense",
            scenario_id="hif_honsen",
        )
        assert cfg.plan == "sense"
        assert cfg.initial_good_condition == 6
        assert cfg.initial_concentration == 6
        assert cfg.initial_good_impression == 0
        assert cfg.initial_motivation == 0
        item_ids = {p.item_id for p in cfg.p_items}
        assert 9139 in item_ids
        assert 9140 in item_ids
        assert 509 not in item_ids
        assert 404 not in item_ids
    finally:
        db.close()


def test_sense_feature_encoder_and_draw_pile_order_invariance():
    from demo.sim.battle_simulator import BattleConfig, HeadlessBattleSimulator, TurnMultiplierSpec, _make_card_config
    from demo.sim.rl_env import GakumasBattleEnv, GakumasFeatureEncoder

    assert GakumasFeatureEncoder.SCALAR_DIMS == 83
    db = GakumasDB("gakumas_library.db")
    try:
        cfg = BattleConfig(
            scenario_id="hif_honsen",
            stage_name="sense_enc",
            p_idol_id=139,
            idol_name="姫崎 莉波",
            plan="sense",
            idol_stats={"vocal": 896, "dance": 2410, "visual": 2233},
            initial_stamina=30,
            max_stamina=44,
            initial_good_condition=6,
            initial_perfect_condition=2,
            initial_concentration=9,
            total_turns=4,
            turn_schedule=[TurnMultiplierSpec("Vo", 1.0)] * 4,
            deck=[_make_card_config(db, cid) for cid in (812, 814, 762, 35, 105, 113)],
        )
        env = GakumasBattleEnv(cfg, db=db)
        sim_a = HeadlessBattleSimulator(cfg, db=db, seed=10)
        sim_b = sim_a.clone()
        sim_b.draw_pile = list(reversed(sim_b.draw_pile))
        feat_a = env.encoder.encode(sim_a)
        feat_b = env.encoder.encode(sim_b)
        assert feat_a.shape[0] >= GakumasFeatureEncoder.SCALAR_DIMS
        assert np.allclose(feat_a, feat_b)

        sim_c = sim_a.clone()
        sim_c.buffs["goodConditionTurns"] = 18
        sim_c.buffs["perfectConditionTurns"] = 5
        sim_c.buffs["concentration"] = 20
        feat_c = env.encoder.encode(sim_c)
        assert not np.allclose(feat_a[: GakumasFeatureEncoder.SCALAR_DIMS], feat_c[: GakumasFeatureEncoder.SCALAR_DIMS])
    finally:
        db.close()


def test_w5_banned_drink_discard_and_popup_guard():
    db = GakumasDB("gakumas_library.db")
    try:
        prefs = {
            "preferred_drink_ids": [22, 18, 14, 12, 13],
            "banned_drink_ids": [19, 3, 1, 2, 4],
            "spend_all_final_shop": True,
        }
        agent = LocalHybridDecisionAgent(db=db, preferences=prefs)
        dummy_box = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)

        # W5 drink reward screen where 破棄候補_18 description mentions banned candidate drinks #1 and #3
        held_18 = LegalAction(
            choice_id=1,
            action_type="drink_discard",
            description="廃棄して交換: 初星ホエイプロテイン (候補: #1《スタミナ爆発ドリンク》 / #2《初星水》 / #3《ビタミンドリンク》)",
            bbox=dummy_box,
            payload={
                "name": "破棄候補_18",
                "drink_id": 18,
                "title": "初星ホエイプロテイン",
            },
        )
        skip_act = LegalAction(
            choice_id=2,
            action_type="skip",
            description="受け取らない",
            bbox=dummy_box,
            payload={"name": "受け取らない"},
        )
        assert agent._find_banned_drink_discard_action([held_18, skip_act]) is None

        # Reward offer screen: none of the offered drinks are desirable over held 18, 13, 21, 22 -> should not trigger unexpected discard
        obs_reward = ScreenObservation(
            scene_type="card_reward",
            raw_text="Pドリンク選択 破棄するドリンクを選択 初星ホエイプロテイン",
            drinks=[
                RecognizedDrink(drink_id=18, name="初星ホエイプロテイン", bbox=dummy_box),
                RecognizedDrink(drink_id=13, name="パワフル漢方ドリンク", bbox=dummy_box),
                RecognizedDrink(drink_id=21, name="初星ブーストエナジー", bbox=dummy_box),
                RecognizedDrink(drink_id=22, name="初星黒酢", bbox=dummy_box),
            ],
        )
        res_skip = agent.decide("", [held_18, skip_act], obs=obs_reward)
        assert res_skip.choice_id == skip_act.choice_id

        # If a banned drink (#3 ビタミンドリンク) is actually in held discard slots, select it
        held_3 = LegalAction(
            choice_id=3,
            action_type="drink_discard",
            description="廃棄候補_3",
            bbox=dummy_box,
            payload={"name": "破棄候補_3", "drink_id": 3, "title": "ビタミンドリンク"},
        )
        assert agent._find_banned_drink_discard_action([held_18, held_3, skip_act]) == held_3

        # Popup guard on W5 `廃棄確認` modal:
        # 1) Unintended discard of non-banned #18 -> must click `いいえ`, never background `捨てる` or `はい`
        btn_bg_discard = LegalAction(
            choice_id=10,
            action_type="popup",
            description="捨てる",
            bbox=dummy_box,
            payload={"name": "捨てる"},
        )
        btn_no = LegalAction(
            choice_id=11,
            action_type="popup",
            description="いいえ",
            bbox=dummy_box,
            payload={"name": "いいえ"},
        )
        btn_yes = LegalAction(
            choice_id=12,
            action_type="popup",
            description="はい",
            bbox=dummy_box,
            payload={"name": "はい"},
        )
        obs_confirm_18 = ScreenObservation(
            scene_type="popup",
            raw_text="廃棄確認 本当に初星ホエイプロテインを捨てますか？",
        )
        agent._expecting_drink_discard = False
        res_cancel = agent._decide_popup([btn_bg_discard, btn_no, btn_yes], obs_confirm_18)
        assert res_cancel.choice_id == btn_no.choice_id

        # 2) Expected or banned drink discard -> must click `はい`, NOT background `捨てる`
        obs_confirm_3 = ScreenObservation(
            scene_type="popup",
            raw_text="廃棄確認 本当にビタミンドリンクを捨てますか？",
        )
        res_confirm_banned = agent._decide_popup([btn_bg_discard, btn_no, btn_yes], obs_confirm_3)
        assert res_confirm_banned.choice_id == btn_yes.choice_id

        agent._expecting_drink_discard = True
        res_confirm_expected = agent._decide_popup([btn_bg_discard, btn_no, btn_yes], obs_confirm_18)
        assert res_confirm_expected.choice_id == btn_yes.choice_id
        assert agent._expecting_drink_discard is False
    finally:
        db.close()


def test_card_memory_hooks_family_propagation_74_75():
    from demo.sim.battle_simulator import HeadlessBattleSimulator, _make_card_config
    from demo.sim.rl_env import GakumasBattleEnv

    db = GakumasDB("gakumas_library.db")
    try:
        hooks = {
            "75": {"free_extra_use_first_play": True, "extra_use_limit": 2, "draw_on_play": 0},
            "105": {"free_extra_use_first_play": True, "extra_use_limit": 1, "draw_on_play": 1},
            "593": {"exiles_trouble_on_play": True},
        }
        # 1) build_live_battle_config propagates upgraded hooks to un-upgraded #74, #104, #592
        cfg = build_live_battle_config(
            db=db,
            deck_card_ids=[74, 104, 592, 812, 814, 23],
            p_idol_id=139,
            plan="sense",
            scenario_id="hif_honsen",
            card_memory_hooks=hooks,
            stamina=40,
            max_stamina=44,
            total_turns=6,
        )
        by_id = {c.card_id: c for c in cfg.deck}
        assert by_id[74].free_extra_use_first_play is True
        assert by_id[74].memory_extra_use_limit == 2
        assert by_id[104].free_extra_use_first_play is True
        assert by_id[104].memory_extra_use_limit == 1
        assert by_id[104].memory_draw_on_play == 1
        assert by_id[592].exiles_trouble_on_play is True

        # 2) Simulator honors extra_use_limit=2 on #74 twice before consuming card_uses_remaining on 3rd play
        sim = HeadlessBattleSimulator(cfg, db=db, seed=1, record_logs=True)
        sim.buffs["goodConditionTurns"] = 10
        sim.genki = 20
        sim.hand = [
            sim._new_instance(_make_card_config(db, 74, free_extra_use_first_play=True, memory_extra_use_limit=2)),
            sim._new_instance(_make_card_config(db, 74, free_extra_use_first_play=True, memory_extra_use_limit=2)),
            sim._new_instance(_make_card_config(db, 74, free_extra_use_first_play=True, memory_extra_use_limit=2)),
        ]
        sim.card_uses_remaining = 1
        sim.play_card(0)
        assert sim.turn == 1 and sim.card_uses_remaining == 1
        sim.play_card(0)
        assert sim.turn == 1 and sim.card_uses_remaining == 1
        sim.play_card(0)
        assert sim.turn == 2  # 3rd play consumed the last action point

        # Check rl_env encoder reflects remaining memory extra uses on obs[..., idx+19]
        env = GakumasBattleEnv(cfg, db=db)
        sim_enc_full = HeadlessBattleSimulator(cfg, db=db, seed=2)
        sim_enc_partial = sim_enc_full.clone()
        sim_enc_partial.memory_extra_use_counts[74] = 1
        vec_full = env.encoder.encode(sim_enc_full)
        vec_partial = env.encoder.encode(sim_enc_partial)
        assert vec_full[17 + 22 + 19] > vec_partial[17 + 22 + 19] > 0.0

        # 3) GameStateTracker & PromptBuilder propagate hooks for p_idol_id=139 across #74/#75 and #592
        tr = GameStateTracker(plan="sense", p_idol_id=139, scenario_id="hif_honsen")
        tr.set_initial_loadout(
            p_idol_id=139,
            plan="sense",
            recommended_effect="goodConditionTurns",
            deck_ids=[74, 104, 592, 812, 814, 23],
            p_items=[],
            card_memory_hooks=hooks,
            db=db,
        )
        lib = tr.export_produce_library(db=db)
        card_entries = {c["card_id"]: c for c in lib["cards"]}
        assert card_entries[74]["memory_hooks"]["extra_use_limit"] == 2
        assert card_entries[104]["memory_hooks"]["draw_on_play"] == 1
        assert card_entries[592]["memory_hooks"]["exiles_trouble_on_play"] is True

        pb = PromptBuilder(db=db, compact=False)
        box = BoundingBox(xmin=10, ymin=1000, xmax=200, ymax=1300)
        obs = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=9,
            turn=1,
            turns_remaining=12,
            stamina=40,
            max_stamina=44,
            genki=20,
            buffs={"goodConditionTurns": 8, "concentration": 6},
            cards=[RecognizedCard(card_id=74, name="アドリブ", type="active", playable=True, bbox=box)],
            options=[{"choice_id": 0, "name": "SKIP", "title": "SKIP"}],
        )
        p1, acts1 = pb.build_prompt(obs, tr)
        a74_1 = next(a for a in acts1 if a.action_type == "play_card")
        assert "メモリー免费再动+1使用次数(不耗行动点)" in a74_1.description
        assert "(2/2)" in a74_1.description
        assert "(剩余2/2)" in p1
        assert "cardUsesRemaining+=1" in (a74_1.payload or {}).get("dsl_actions", "")
        assert (a74_1.payload or {}).get("memory_extra_use_remaining") == 2

        tr.in_battle = True
        tr.record_card_play(74, db=db)
        p2, acts2 = pb.build_prompt(obs, tr)
        a74_2 = next(a for a in acts2 if a.action_type == "play_card")
        assert "(1/2)" in a74_2.description
        assert "(剩余1/2)" in p2
        assert (a74_2.payload or {}).get("memory_extra_use_remaining") == 1

        # Playing #75 (same family 74) uses 2nd charge -> 3rd prompt no longer grants free extra use
        tr.record_card_play(75, db=db)
        _, acts3 = pb.build_prompt(obs, tr)
        a74_3 = next(a for a in acts3 if a.action_type == "play_card")
        assert "メモリー免费再动+1使用次数" not in a74_3.description
        assert (a74_3.payload or {}).get("memory_extra_use_remaining") == 0

        # #592 trouble-exile hook works on p_idol_id=139 when #23 is in battle zones
        assert 23 in (tr.battle_draw_pile + tr.battle_hand + tr.battle_discard_pile)
        tr.record_card_play(592, db=db)
        assert 23 in tr.battle_exhaust_pile

        # 4) LiveBattlePlanner._find_or_create_card_instance injects hooks on unseen dynamic cards
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_hidden_dim=32, ppo_episodes=4, ismcts_simulations=2),
        )
        tr2 = GameStateTracker(plan="sense", p_idol_id=139, scenario_id="hif_honsen")
        tr2.set_initial_loadout(
            p_idol_id=139,
            plan="sense",
            recommended_effect="goodConditionTurns",
            deck_ids=[812, 814, 163, 105, 113],
            p_items=[],
            card_memory_hooks=hooks,
            db=db,
        )
        planner.decide_battle_action(obs, acts1, tracker=tr2)
        assert planner._sim is not None
        dyn_74 = next(c for c in planner._sim.hand if c.cfg.card_id == 74)
        assert dyn_74.cfg.free_extra_use_first_play is True
        assert dyn_74.cfg.memory_extra_use_limit == 2
    finally:
        db.close()


def test_sense_163_814_combo_and_finisher_hold_timing():
    from demo.sim.battle_simulator import HeadlessBattleSimulator, _make_card_config

    db = GakumasDB("gakumas_library.db")
    try:
        cfg = build_live_battle_config(
            db=db,
            deck_card_ids=[812, 814, 163, 117, 75, 105, 113, 35],
            equipped_p_items=[{"item_id": 436}, {"item_id": 9139}, {"item_id": 9140}],
            idol_stats={"vocal": 1000, "dance": 2419, "visual": 2632},
            stamina=40,
            max_stamina=44,
            total_turns=12,
            p_idol_id=139,
            plan="sense",
            scenario_id="hif_honsen",
            card_memory_hooks={"75": {"free_extra_use_first_play": True, "extra_use_limit": 2}},
        )
        sim = HeadlessBattleSimulator(cfg, db=db, seed=5)
        sim.buffs["goodConditionTurns"] = 20
        sim.buffs["concentration"] = 15
        sim.genki = 20
        c163 = sim._new_instance(_make_card_config(db, 163))
        c117 = sim._new_instance(_make_card_config(db, 117))
        c75 = sim._new_instance(
            _make_card_config(db, 75, free_extra_use_first_play=True, memory_extra_use_limit=2)
        )
        sim.hand = [c163, c117, c75]
        assert sim.is_safe_dominating_free_card(c163) is False
        assert sim.is_safe_dominating_free_card(c117) is True
        assert sim.is_safe_dominating_free_card(c75) is True

        # Once double_card_effect_cards > 0 (#162/#163 armed), NO free card (Mental or Active) is auto-forced
        sim.double_card_effect_cards = 1
        assert sim.is_safe_dominating_free_card(c117) is False
        assert sim.is_safe_dominating_free_card(c75) is False

        # Codec pruning check: rem=3 holds #814 and #163; rem=2 unlocks both
        from demo.sim.rl_env import GakumasActionCodec

        sim.double_card_effect_cards = 0
        c814 = sim._new_instance(_make_card_config(db, 814))
        c74 = sim._new_instance(_make_card_config(db, 74))
        sim.hand = [c163, c814, c74]
        sim.draw_pile = []
        codec_fin = GakumasActionCodec(cfg, db=db, finisher_play_max_turns_remaining=2)
        aid_c163 = codec_fin.card_slot_index(c163.cfg)
        aid_c814 = codec_fin.card_slot_index(c814.cfg)
        aid_c74 = codec_fin.card_slot_index(c74.cfg)
        sim.turns_remaining = 3
        _, map_rem3 = codec_fin.legal_mask_and_map(sim, prune_dominated=True)
        assert aid_c814 not in map_rem3
        assert aid_c163 not in map_rem3
        assert aid_c74 in map_rem3
        sim.turns_remaining = 2
        _, map_rem2 = codec_fin.legal_mask_and_map(sim, prune_dominated=True)
        assert aid_c814 in map_rem2
        assert aid_c163 in map_rem2

        # LiveBattlePlanner combo & hold tests
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_hidden_dim=32, ppo_episodes=8, ismcts_simulations=4),
        )
        tr = GameStateTracker(plan="sense", p_idol_id=139, scenario_id="hif_honsen")
        tr.set_initial_loadout(
            p_idol_id=139,
            plan="sense",
            recommended_effect="goodConditionTurns",
            deck_ids=[812, 814, 163, 74, 75, 105, 113, 35],
            p_items=[{"item_id": 436}, {"item_id": 9139}, {"item_id": 9140}],
            card_memory_hooks={"75": {"free_extra_use_first_play": True, "extra_use_limit": 2}},
            db=db,
        )
        pb = PromptBuilder(db=db, compact=False)
        box = BoundingBox(xmin=10, ymin=1000, xmax=200, ymax=1300)

        # Case A: Early turn (rem=9), ISMCTS chooses END_TURN -> Anti-SKIP must NOT force #814 or #163
        obs_early = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=9,
            turn=4,
            turns_remaining=9,
            stamina=40,
            max_stamina=44,
            genki=20,
            idol_stats={"vocal": 1000, "dance": 2419, "visual": 2632},
            buffs={"goodConditionTurns": 20, "concentration": 15},
            cards=[
                RecognizedCard(card_id=163, name="国民的アイドル+", type="mental", playable=True, visual_playable=True, bbox=box),
                RecognizedCard(card_id=814, name="自然体の魅力+", type="active", playable=True, visual_playable=True, bbox=box),
            ],
            options=[{"choice_id": 0, "name": "SKIP", "title": "SKIP"}],
        )
        _, acts_early = pb.build_prompt(obs_early, tr)
        planner.decide_battle_action(obs_early, acts_early, tracker=tr)
        end_aid = planner._codec.end_turn_action_id  # type: ignore[union-attr]
        planner._ismcts.select_action = lambda *_a, **_kw: (end_aid, {"best_q_score": 500_000.0, "top_actions": [("END_TURN", 4, 500_000.0)]})  # type: ignore[union-attr,method-assign]
        cid_hold, reason_hold, _ = planner.decide_battle_action(obs_early, acts_early, tracker=tr)
        chosen_hold = next(a for a in acts_early if a.choice_id == cid_hold)
        assert chosen_hold.action_type == "end_turn"
        assert "Anti-SKIP" not in reason_hold

        # Case B: rem=3 (倒数第3回合) -> Even if ISMCTS picks #814 or #163, hold both for last 2 turns
        obs_rem3 = obs_early.model_copy(
            update={
                "turn": 10,
                "turns_remaining": 3,
                "cards": [
                    RecognizedCard(card_id=163, name="国民的アイドル+", type="mental", playable=True, visual_playable=True, bbox=box),
                    RecognizedCard(card_id=814, name="自然体の魅力+", type="active", playable=True, visual_playable=True, bbox=box),
                    RecognizedCard(card_id=74, name="アドリブ", type="active", playable=True, visual_playable=True, bbox=box),
                ],
            }
        )
        _, acts_rem3 = pb.build_prompt(obs_rem3, tr)
        a163_rem3 = next(a for a in acts_rem3 if (a.payload or {}).get("card_id") == 163)
        a814_rem3 = next(a for a in acts_rem3 if (a.payload or {}).get("card_id") == 814)
        a74_rem3 = next(a for a in acts_rem3 if (a.payload or {}).get("card_id") == 74)
        mapped_rem3 = planner._map_legal_actions_to_canonical(acts_rem3)
        aid_814_rem3 = next(aid for aid, act in mapped_rem3.items() if act.choice_id == a814_rem3.choice_id)
        aid_163_rem3 = next(aid for aid, act in mapped_rem3.items() if act.choice_id == a163_rem3.choice_id)
        planner._ismcts.select_action = lambda *_a, **_kw: (aid_814_rem3, {"best_q_score": 600_000.0, "top_actions": []})  # type: ignore[union-attr,method-assign]
        cid_rem3_a, reason_rem3_a, _ = planner.decide_battle_action(obs_rem3, acts_rem3, tracker=tr)
        assert cid_rem3_a == a74_rem3.choice_id
        assert "Hold #814/#163 for last 2 turns" in reason_rem3_a

        planner._ismcts.select_action = lambda *_a, **_kw: (aid_163_rem3, {"best_q_score": 600_000.0, "top_actions": []})  # type: ignore[union-attr,method-assign]
        cid_rem3_b, reason_rem3_b, _ = planner.decide_battle_action(obs_rem3, acts_rem3, tracker=tr)
        assert cid_rem3_b == a74_rem3.choice_id
        assert "Hold #814/#163 for last 2 turns" in reason_rem3_b

        # Without #74 at rem=3, holds via END_TURN instead of burning #814 or #163 early
        obs_rem3_only = obs_rem3.model_copy(
            update={
                "cards": [
                    RecognizedCard(card_id=163, name="国民的アイドル+", type="mental", playable=True, visual_playable=True, bbox=box),
                    RecognizedCard(card_id=814, name="自然体の魅力+", type="active", playable=True, visual_playable=True, bbox=box),
                ],
            }
        )
        _, acts_rem3_only = pb.build_prompt(obs_rem3_only, tr)
        a814_only = next(a for a in acts_rem3_only if (a.payload or {}).get("card_id") == 814)
        aid_814_only = next(aid for aid, act in planner._map_legal_actions_to_canonical(acts_rem3_only).items() if act.choice_id == a814_only.choice_id)
        planner._ismcts.select_action = lambda *_a, **_kw: (aid_814_only, {"best_q_score": 600_000.0, "top_actions": []})  # type: ignore[union-attr,method-assign]
        cid_rem3_skip, reason_rem3_skip, _ = planner.decide_battle_action(obs_rem3_only, acts_rem3_only, tracker=tr)
        assert next(a for a in acts_rem3_only if a.choice_id == cid_rem3_skip).action_type == "end_turn"
        assert "Hold #814/#163 for last 2 turns" in reason_rem3_skip

        # Case C: rem=2 (倒数第2回合) #163 + #814 in hand when ISMCTS suggests #814 -> switches to #163 first ([Combo #162/#163->#814 2x])
        obs_combo = obs_early.model_copy(
            update={
                "turn": 11,
                "turns_remaining": 2,
                "cards": [
                    RecognizedCard(card_id=163, name="国民的アイドル+", type="mental", playable=True, visual_playable=True, bbox=box),
                    RecognizedCard(card_id=814, name="自然体の魅力+", type="active", playable=True, visual_playable=True, bbox=box),
                    RecognizedCard(card_id=74, name="アドリブ", type="active", playable=True, visual_playable=True, bbox=box),
                ],
            }
        )
        _, acts_combo = pb.build_prompt(obs_combo, tr)
        a163 = next(a for a in acts_combo if (a.payload or {}).get("card_id") == 163)
        a814 = next(a for a in acts_combo if (a.payload or {}).get("card_id") == 814)
        aid_814 = next(aid for aid, act in planner._map_legal_actions_to_canonical(acts_combo).items() if act.choice_id == a814.choice_id)  # type: ignore[union-attr]
        planner._ismcts.select_action = lambda *_a, **_kw: (aid_814, {"best_q_score": 600_000.0, "top_actions": []})  # type: ignore[union-attr,method-assign]
        cid_first, reason_first, _ = planner.decide_battle_action(obs_combo, acts_combo, tracker=tr)
        assert cid_first == a163.choice_id
        assert "Combo #162/#163->#814 2x" in reason_first

        # Execute #163 into internal simulator -> arms double_card_effect_cards=1; next step must pick #814 over #74
        planner.notify_action_executed(a163)
        assert planner._sim.double_card_effect_cards == 1
        obs_armed = obs_combo.model_copy(
            update={
                "buffs": {"goodConditionTurns": 19, "concentration": 15},
                "cards": [
                    RecognizedCard(card_id=814, name="自然体の魅力+", type="active", playable=True, visual_playable=True, bbox=box),
                    RecognizedCard(card_id=74, name="アドリブ", type="active", playable=True, visual_playable=True, bbox=box),
                ],
            }
        )
        _, acts_armed = pb.build_prompt(obs_armed, tr)
        a814_armed = next(a for a in acts_armed if (a.payload or {}).get("card_id") == 814)
        a74_armed = next(a for a in acts_armed if (a.payload or {}).get("card_id") == 74)
        aid_74 = next(aid for aid, act in planner._map_legal_actions_to_canonical(acts_armed).items() if act.choice_id == a74_armed.choice_id)  # type: ignore[union-attr]
        planner._ismcts.select_action = lambda *_a, **_kw: (aid_74, {"best_q_score": 400_000.0, "top_actions": []})  # type: ignore[union-attr,method-assign]
        cid_second, reason_second, _ = planner.decide_battle_action(obs_armed, acts_armed, tracker=tr)
        assert cid_second == a814_armed.choice_id
        assert "Armed 2x->#814" in reason_second

        # Case D: rem=2 cross-turn 2x carry-over -> #814 in draw_pile (len <= 3) with 2x armed must preserve 2x via END_TURN
        planner._sim.double_card_effect_cards = 1
        planner._sim.draw_pile = [planner._sim._new_instance(_make_card_config(db, 814))]
        obs_carry = obs_combo.model_copy(
            update={
                "buffs": {"goodConditionTurns": 19, "concentration": 15},
                "cards": [
                    RecognizedCard(card_id=74, name="アドリブ", type="active", playable=True, visual_playable=True, bbox=box),
                ],
            }
        )
        _, acts_carry = pb.build_prompt(obs_carry, tr)
        a74_carry = next(a for a in acts_carry if (a.payload or {}).get("card_id") == 74)
        aid_74_carry = next(aid for aid, act in planner._map_legal_actions_to_canonical(acts_carry).items() if act.choice_id == a74_carry.choice_id)
        planner._ismcts.select_action = lambda *_a, **_kw: (aid_74_carry, {"best_q_score": 300_000.0, "top_actions": []})  # type: ignore[union-attr,method-assign]
        cid_carry, reason_carry, _ = planner.decide_battle_action(obs_carry, acts_carry, tracker=tr)
        assert next(a for a in acts_carry if a.choice_id == cid_carry).action_type == "end_turn"
        assert "Carry 2x->rem1 #814" in reason_carry
        planner._ismcts.select_action = lambda *_a, **_kw: (end_aid, {"best_q_score": 500_000.0, "top_actions": []})  # type: ignore[union-attr,method-assign]
        cid_carry_skip, reason_carry_skip, _ = planner.decide_battle_action(obs_carry, acts_carry, tracker=tr)
        assert next(a for a in acts_carry if a.choice_id == cid_carry_skip).action_type == "end_turn"
        assert "Anti-SKIP" not in reason_carry_skip
    finally:
        db.close()


def test_upgrade_confirm_modal_recognized_as_popup():
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=8, ismcts_simulations=4),
        )
        # 1. Popup with "決定" and "キャンセル" on 強化 confirmation modal picks confirm button
        pop_obs_1 = ScreenObservation(
            scene_type="popup",
            plan="sense",
            scenario_id="hif_honsen",
            week=8,
            raw_text="スキルカード強化\nPポイントを使用してスキルカードを強化しますか?\nキャンセル\n強化",
        )
        acts_1 = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="底部橙色主按钮: ノ強化",
                bbox=BoundingBox(xmin=379, ymin=1444, xmax=623, ymax=1515),
                payload={"name": "決定", "title": "決定"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="弹窗按钮: キャンセル",
                bbox=BoundingBox(xmin=139, ymin=1448, xmax=344, ymax=1513),
                payload={"name": "キャンセル", "title": "キャンセル"},
            ),
        ]
        dec_1 = agent.decide("", acts_1, obs=pop_obs_1, p_idol_id=139)
        assert dec_1.choice_id == 0

        # 2. Popup where confirm button text is literally "強化" (not "強化する" or "決定") also confirms instead of cancelling
        acts_2 = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="弹窗按钮: キャンセル",
                bbox=BoundingBox(xmin=139, ymin=1448, xmax=344, ymax=1513),
                payload={"name": "キャンセル", "title": "キャンセル"},
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="弹窗按钮: 強化",
                bbox=BoundingBox(xmin=379, ymin=1444, xmax=623, ymax=1515),
                payload={"name": "強化", "title": "強化"},
            ),
        ]
        dec_2 = agent.decide("", acts_2, obs=pop_obs_1, p_idol_id=139)
        assert dec_2.choice_id == 1

        # 3. Verify recognizer directly on the captured W8 stuck modal screenshot when present on disk
        stuck_img_path = Path(
            r"C:\Users\josha\.gemini\antigravity\brain\a7d6f894-5801-42de-a9e3-ab8624d8418b\scratch\run2_stuck_screen.png"
        )
        if stuck_img_path.exists():
            import cv2
            from demo.vision.live_adb_recognizer import LiveScreenRecognizer

            bgr = cv2.imread(str(stuck_img_path))
            assert bgr is not None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rec = LiveScreenRecognizer(db)
            try:
                obs_live = rec.recognize(rgb, plan="sense")
                assert obs_live.scene_type == "popup"
                opt_names = {str(o.get("name") or "") for o in (obs_live.options or [])}
                assert "決定" in opt_names or "強化" in opt_names
            finally:
                if hasattr(rec, "close"):
                    rec.close()
    finally:
        if hasattr(agent, "close"):
            agent.close()
        db.close()


def test_upgrade_grid_slot_tracking_picks_best_card():
    db = GakumasDB("gakumas_library.db")
    try:
        agent = LocalHybridDecisionAgent(
            db=db,
            preferences={"priority_card_ids": [112]},
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=8, ismcts_simulations=4),
        )
        base_obs = ScreenObservation(
            scene_type="card_reward",
            plan="sense",
            scenario_id="hif_honsen",
            week=8,
            raw_text="スキルカード強化\n強化するスキルカードを選んでください",
        )
        box = BoundingBox(xmin=79, ymin=700, xmax=198, ymax=822)

        def _grid_previews() -> list[LegalAction]:
            return [
                LegalAction(
                    choice_id=idx + 1,
                    action_type="pick_option",
                    description=f"切换选中卡库第{idx + 1}张待选卡",
                    bbox=box,
                    payload={"name": f"预览第{idx + 1}张待选卡", "slot_idx": idx},
                )
                for idx in range(4)
            ]

        def _frame_for_slot(slot_idx: int, cid: int, title: str, desc: str) -> list[LegalAction]:
            return [
                LegalAction(
                    choice_id=0,
                    action_type="pick_option",
                    description=desc,
                    bbox=BoundingBox(xmin=451, ymin=1449, xmax=595, ymax=1511),
                    payload={"name": "強化する", "card_id": cid, "slot_idx": slot_idx, "title": title},
                ),
                *_grid_previews(),
                LegalAction(
                    choice_id=5,
                    action_type="pick_option",
                    description="取消操作并返回",
                    bbox=BoundingBox(xmin=81, ymin=1450, xmax=344, ymax=1510),
                    payload={"name": "キャンセル"},
                ),
            ]

        # Initial empty selection screen: 待选卡 1..4 -> picks 待选卡 1 (slot 0)
        init_acts = [
            LegalAction(
                choice_id=idx,
                action_type="pick_option",
                description=f"待选卡 {idx + 1}",
                bbox=box,
                payload={"name": f"待选卡{idx + 1}", "slot_idx": idx},
            )
            for idx in range(4)
        ]
        d0 = agent.decide("", init_acts, obs=base_obs, p_idol_id=139)
        assert d0.choice_id == 0

        # Slot 0: #106 演出計画 -> inspects slot 1 next (and records slot 0)
        d1 = agent.decide("", _frame_for_slot(0, 106, "演出計画", "絶好調3ターン"), obs=base_obs, p_idol_id=139)
        assert d1.choice_id == 2  # 预览第2张待选卡 (slot 1)

        # Slot 1: #112 始まりの合図 (highest-scored card) -> inspects slot 2 next (preserving slot 1!)
        d2 = agent.decide("", _frame_for_slot(1, 112, "始まりの合図", "好調5ターン"), obs=base_obs, p_idol_id=139)
        assert d2.choice_id == 3  # 预览第3张待选卡 (slot 2)

        # Slot 2: #104 大声援 -> inspects slot 3 next
        d3 = agent.decide("", _frame_for_slot(2, 104, "大声援", "元気+6"), obs=base_obs, p_idol_id=139)
        assert d3.choice_id == 4  # 预览第4张待选卡 (slot 3)

        # Slot 3: #23 眠気 (low score) -> must NOT upgrade slot 3; switches back to slot 1 (#112)
        d4 = agent.decide("", _frame_for_slot(3, 23, "眠気", "使用不可"), obs=base_obs, p_idol_id=139)
        assert d4.choice_id == 2  # 预览第2张待选卡 (best_slot = 1)
        assert sorted(agent._reward_seen_slots.keys()) == [0, 1, 2, 3]

        # Back on Slot 1 (#112 始まりの合図) -> confirms 強化する!
        d5 = agent.decide("", _frame_for_slot(1, 112, "始まりの合図", "好調5ターン"), obs=base_obs, p_idol_id=139)
        assert d5.choice_id == 0
        assert "LocalAgent(reward): Claiming best evaluated card" in d5.reason
    finally:
        if hasattr(agent, "close"):
            agent.close()
        db.close()


def test_w6_schedule_consultation_bubble_not_misclassified_as_shop():
    w6_img_path = Path(
        r"C:\Users\josha\.gemini\antigravity\brain\a7d6f894-5801-42de-a9e3-ab8624d8418b\scratch\run3_w6_stuck.png"
    )
    w8_img_path = Path(
        r"C:\Users\josha\.gemini\antigravity\brain\a7d6f894-5801-42de-a9e3-ab8624d8418b\scratch\run3_w8_stuck.png"
    )
    if (not w6_img_path.exists()) and (not w8_img_path.exists()):
        return
    import cv2
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    agent = LocalHybridDecisionAgent(
        db=db,
        planner_cfg=LiveBattlePlannerConfig(ppo_episodes=8, ismcts_simulations=4),
    )
    rec = LiveScreenRecognizer(db)
    try:
        if w6_img_path.exists():
            bgr = cv2.imread(str(w6_img_path))
            assert bgr is not None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            obs = rec.recognize(rgb, plan="sense")
            assert obs.scene_type == "produce_schedule"
            opt_names = [str(o.get("name") or "") for o in (obs.options or [])]
            assert "相談" in opt_names

            tr = GameStateTracker(plan="sense", p_idol_id=139, scenario_id="hif_honsen")
            pb = PromptBuilder(db=db, compact=False)
            prompt, acts = pb.build_prompt(obs, tr)
            dec = agent.decide(prompt, acts, obs=obs, p_idol_id=139)
            chosen = next(a for a in acts if a.choice_id == dec.choice_id)
            assert (chosen.payload or {}).get("name") == "相談"
            cx = (chosen.bbox.xmin + chosen.bbox.xmax) // 2
            cy = (chosen.bbox.ymin + chosen.bbox.ymax) // 2
            assert 280 <= cx <= 450
            assert 1240 <= cy <= 1450

        if w8_img_path.exists():
            bgr_w8 = cv2.imread(str(w8_img_path))
            assert bgr_w8 is not None
            rgb_w8 = cv2.cvtColor(bgr_w8, cv2.COLOR_BGR2RGB)
            obs_w8 = rec.recognize(rgb_w8, plan="sense")
            assert obs_w8.scene_type == "shop"
            w8_by_name = {str(o.get("name") or ""): o for o in (obs_w8.options or [])}
            assert "強化" in w8_by_name
            assert "リフレッシュ" in w8_by_name
            assert "終了" in w8_by_name
            assert "削除" not in w8_by_name
            assert w8_by_name.get("Pドリンク3", {}).get("drink_id") == 3
            assert w8_by_name.get("Pドリンク4", {}).get("drink_id") == 15
            rb = w8_by_name["リフレッシュ"]["bbox"]
            rcx = (rb["xmin"] + rb["xmax"]) // 2
            rcy = (rb["ymin"] + rb["ymax"]) // 2
            assert 80 <= rcx <= 145
            assert 1212 <= rcy <= 1242
            assert rb["ymax"] <= 1250
    finally:
        if hasattr(rec, "close"):
            rec.close()
        if hasattr(agent, "close"):
            agent.close()
        db.close()


def test_w6_vs_w8_shop_template_isolation_and_hif_upper_rest_sp_recovery():
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    rec = LiveScreenRecognizer(db=db, plan="sense", scenario_id="hif_honsen")
    orig_ocr = rec.ocr
    orig_en_ocr = rec.en_ocr
    try:
        # 1. W6 相談 shop merged OCR ("強化 100 削除 125") must clamp 強化 left, 削除 right, and use W6 template
        w6_frame = np.full((1600, 720, 3), 235, dtype=np.uint8)
        w6_frame[1090:1165, 55:340] = (240, 95, 135)
        w6_frame[1090:1165, 380:665] = (80, 195, 250)
        w6_lines = [
            {"text": "相談", "bbox": [90, 60, 210, 100]},
            {"text": "80", "bbox": [95, 968, 175, 1005]},
            {"text": "90", "bbox": [245, 968, 325, 1005]},
            {"text": "強化 100 削除 125", "bbox": [101, 1110, 680, 1150]},
            {"text": "終了", "bbox": [585, 1220, 680, 1265]},
        ]
        meta6 = {
            "week": 6,
            "total_weeks": 9,
            "week_label": "6日目",
            "stats": {"vocal": 1000, "dance": 2651, "visual": 2372},
            "p_points": 300,
            "stamina": 35,
            "max_stamina": 44,
            "drinks": [],
        }
        rec.ocr = SimpleNamespace(recognize_array=lambda *a, **kw: {"text": "", "lines": []})
        rec.en_ocr = SimpleNamespace(recognize_array=lambda *a, **kw: {"text": "", "lines": []})
        obs_w6 = rec._parse_shop_screen(w6_frame, w6_lines, "sense", meta6)
        w6_by_name = {o.get("name"): o for o in obs_w6.options}
        assert "強化" in w6_by_name and "削除" in w6_by_name
        upg_cx = (w6_by_name["強化"]["bbox"]["xmin"] + w6_by_name["強化"]["bbox"]["xmax"]) // 2
        del_cx = (w6_by_name["削除"]["bbox"]["xmin"] + w6_by_name["削除"]["bbox"]["xmax"]) // 2
        assert upg_cx < 350
        assert del_cx > 370
        assert w6_by_name["強化"].get("price") == 100
        assert w6_by_name["削除"].get("price") == 125
        # W6 grid row 2 uses Pドリンク1..4 (never W8's スキルカード5..6)
        assert "スキルカード5" not in w6_by_name and "スキルカード6" not in w6_by_name
        assert "Pドリンク1" in w6_by_name and "Pドリンク2" in w6_by_name

        # 2. H.I.F 3-tile 公開レッスン with [休む] on upper row (y=1137..1231) must synthesize missing
        # middle SP tile Da at cx~358 (NOT cx=277) and tag it as SP!
        sp_frame = np.full((1600, 720, 3), 40, dtype=np.uint8)
        for slot_cx in (172, 360, 545):
            sp_frame[1240:1440, slot_cx - 75 : slot_cx + 75] = (235, 235, 235)
        sp_frame[1135:1232, 610:682] = (235, 235, 235)  # Upper-right [休む]
        # Gold + magenta SP badge above middle Da slot at cx=360
        sp_frame[1235:1260, 300:345] = (245, 200, 50)
        sp_frame[1262:1287, 300:345] = (240, 80, 210)
        hif_lines = [
            {"text": "休む", "bbox": [615, 1185, 675, 1222]},
            {"text": "Vo.公開レッスン", "bbox": [90, 1385, 254, 1418]},
            {"text": "Vi.公開レッスン", "bbox": [463, 1385, 627, 1418]},
        ]
        opts = rec._find_keyword_options(sp_frame, hif_lines, 720, 1600)
        da_opts = [o for o in opts if o["name"].startswith("Da.")]
        assert len(da_opts) == 1
        assert da_opts[0]["name"] == "Da.SP公開レッスン"
        assert da_opts[0].get("is_sp") is True
        da_cx = (da_opts[0]["bbox"]["xmin"] + da_opts[0]["bbox"]["xmax"]) // 2
        assert 340 <= da_cx <= 380
    finally:
        rec.ocr = orig_ocr
        rec.en_ocr = orig_en_ocr
        if hasattr(rec, "close"):
            rec.close()
        db.close()


def test_p_point_middle_digit_drop_and_drink_22_tutor_and_multiplier_timing(monkeypatch):
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    agent = LocalHybridDecisionAgent(
        db=db,
        planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
    )
    try:
        # 1a. Recognizer P-point guard drops middle-digit truncations (449->49, 439->49)
        # while allowing valid ~100-150P spends (120->20, 155->30)
        rec = LiveScreenRecognizer(db=db, plan="sense", scenario_id="hif_honsen")
        frame = np.full((1600, 720, 3), 240, dtype=np.uint8)
        hp_lines = [{"text": "Pポイント 42/44", "bbox": [250, 32, 330, 58]}]
        cur_v = {"v": 449}
        monkeypatch.setattr(rec, "_ocr_bw_banner_number", lambda *_a, **_kw: cur_v["v"])
        monkeypatch.setattr(rec, "_ocr_crop_upscaled", lambda *_a, **_kw: "")
        for raw_v, exp_v in ((449, 449), (49, 449), (439, 439), (49, 439), (120, 120), (20, 20), (155, 155), (30, 30)):
            cur_v["v"] = raw_v
            assert rec._parse_week_and_stats(hp_lines, frame)["p_points"] == exp_v

        # 1b. GakumasRunner preserves expected 439P after 10P リフレッシュ even if OCR persistently drops '3' -> 49
        btn_box = BoundingBox(xmin=18, ymin=1207, xmax=198, ymax=1246)
        exit_box = BoundingBox(xmin=563, ymin=1209, xmax=715, ymax=1280)
        obs_pre_ref = ScreenObservation(
            scene_type="shop",
            plan="sense",
            scenario_id="hif_honsen",
            week=8,
            stamina=42,
            max_stamina=44,
            p_points=449,
            options=[
                {"choice_id": 0, "name": "リフレッシュ", "title": "リフレッシュ (10P)", "shop_slot_key": "リフレッシュ", "price": 10, "bbox": btn_box.model_dump()},
                {"choice_id": 1, "name": "終了", "title": "終了", "bbox": exit_box.model_dump()},
            ],
        )
        obs_bad_post = ScreenObservation(
            scene_type="shop",
            plan="sense",
            scenario_id="hif_honsen",
            week=8,
            stamina=42,
            max_stamina=44,
            p_points=49,
            options=[
                {"choice_id": 0, "name": "リフレッシュ", "title": "リフレッシュ (10P)", "shop_slot_key": "リフレッシュ", "price": 10, "bbox": btn_box.model_dump()},
                {"choice_id": 1, "name": "終了", "title": "終了", "bbox": exit_box.model_dump()},
            ],
        )
        inner_r = SimpleNamespace(_last_p_points=449, _shop_visited_slots=set(), _shop_last_clicked_slot=None)
        phase_n = {"n": 0}

        def _fake_ref_rec(_fr):
            phase_n["n"] += 1
            return obs_pre_ref if phase_n["n"] == 1 else obs_bad_post

        ctrl_r = SimpleNamespace(
            is_connected=True,
            connect=lambda: True,
            disconnect=lambda: None,
            screencap=lambda: frame,
            click=lambda _x, _y: True,
        )
        runner_p = GakumasRunner(
            mode="adb",
            controller=ctrl_r,  # type: ignore[arg-type]
            agent=agent,
            db=db,
            plan="sense",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        runner_p._has_entered_produce = True
        runner_p._shop_settle_sleep = 0.0
        runner_p.tracker.p_points = 449
        runner_p.scene_routing_reco = SimpleNamespace(recognizer=inner_r, recognize=_fake_ref_rec)
        runner_p.run_once()  # clicks リフレッシュ (-10P -> expected 439)
        assert runner_p._last_shop_expected_p == 439
        res_post = runner_p.run_once()
        assert res_post.observation is not None and res_post.observation.p_points == 439

        # 2. #22 初星黒酢 move-to-hand modal: recognized as card_reward ('移動する'), never bottom-scrolled,
        # avoids #23 眠気 and selects #814 自然体の魅力+!
        tutor_frame = np.full((1600, 720, 3), 35, dtype=np.uint8)
        for sx in (138, 282, 426):
            tutor_frame[835:955, sx - 58 : sx + 58] = (230, 230, 230)
        tutor_frame[832:958, 76:200] = (245, 145, 30)  # slot 0 selected
        tutor_frame[1444:1514, 238:481] = (245, 130, 30)  # active orange 移動 button
        tutor_lines = [
            {"text": "山札か捨札から手札に移動するスキルカードを選択", "bbox": [70, 120, 650, 155]},
            {"text": "ノ移動", "bbox": [310, 1458, 410, 1502]},
            {"text": "キャンセル", "bbox": [160, 1458, 300, 1502]},
        ]
        orig_rec_ocr = rec.ocr
        orig_rec_en = rec.en_ocr
        pv_toggle = {"card": "眠気"}
        rec.ocr = SimpleNamespace(
            recognize_array=lambda arr, *a, **kw: {
                "text": "",
                "lines": tutor_lines if arr.shape[:2] == (1600, 720) else [{"text": pv_toggle["card"], "bbox": [20, 10, 260, 45]}],
            }
        )
        rec.en_ocr = SimpleNamespace(recognize_array=lambda *a, **kw: {"text": "", "lines": []})
        rec._last_turns_remaining = 2  # Propagates active late-turn battle HUD into #22 modal observation
        obs_tutor0 = rec.recognize(tutor_frame, plan="sense")
        assert obs_tutor0.scene_type == "card_reward"
        assert obs_tutor0.turns_remaining == 2
        assert obs_tutor0.options[0]["name"] == "移動する"
        assert not any("换出" in o["name"] for o in obs_tutor0.options)

        tutor_obs_nemuki = obs_tutor0
        tutor_acts0 = [
            LegalAction(choice_id=o["choice_id"], action_type="pick_option", description=o["title"], bbox=BoundingBox(**o["bbox"]), payload=o)
            for o in tutor_obs_nemuki.options
        ]
        agent.reset_session()
        d_t0 = agent.decide("", tutor_acts0, obs=tutor_obs_nemuki, p_idol_id=139)
        assert tutor_acts0[d_t0.choice_id].payload.get("slot_idx") == 1  # Reject #23 眠気 on slot 0, inspect slot 1

        pv_toggle["card"] = "自然体の魅力+"
        obs_tutor1 = rec.recognize(tutor_frame, plan="sense")
        obs_tutor1.options[0]["slot_idx"] = 1
        tutor_acts1 = [
            LegalAction(choice_id=o["choice_id"], action_type="pick_option", description=o["title"], bbox=BoundingBox(**o["bbox"]), payload=o)
            for o in obs_tutor1.options
        ]
        d_t1 = agent.decide("", tutor_acts1, obs=obs_tutor1, p_idol_id=139)
        assert d_t1.choice_id == 0  # Immediately confirm 移動する for #814 自然体の魅力+
        rec.ocr = orig_rec_ocr
        rec.en_ocr = orig_rec_en

        # 3. Battle planner timing: drinks #12 before firing armed 2x #814, and holds #12/#14/#776/#22 early
        tr = GameStateTracker(plan="sense", p_idol_id=139, scenario_id="hif_honsen")
        tr.set_initial_loadout(
            p_idol_id=139,
            plan="sense",
            recommended_effect="goodConditionTurns",
            deck_ids=[812, 814, 162, 776, 105, 113, 75, 23],
            p_items=[{"item_id": 447}, {"item_id": 9139}],
            db=db,
        )
        box_c = BoundingBox(xmin=100, ymin=1050, xmax=260, ymax=1350)
        planner = agent.battle_planner

        # 3a. rem=7 (early): holds #12/#14/#22/#776/#162/#814; chooses #105
        obs_early = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=9,
            turn=6,
            turns_remaining=7,
            stamina=40,
            max_stamina=44,
            idol_stats={"vocal": 2400, "dance": 2200, "visual": 2000},
            buffs={"goodConditionTurns": 15, "perfectConditionTurns": 3, "concentration": 12},
            cards=[
                RecognizedCard(card_id=105, name="大声援+", upgraded=True, playable=True, visual_playable=True, bbox=box_c),
                RecognizedCard(card_id=776, name="話題沸騰+", upgraded=True, playable=True, visual_playable=True, bbox=box_c),
                RecognizedCard(card_id=162, name="国民的アイドル", upgraded=False, playable=True, visual_playable=True, bbox=box_c),
                RecognizedCard(card_id=814, name="自然体の魅力+", upgraded=True, playable=True, visual_playable=True, bbox=box_c),
            ],
            drinks=[
                RecognizedDrink(drink_id=12, name="ブーストエキス", playable=True, bbox=box_c),
                RecognizedDrink(drink_id=14, name="センブリソーダ", playable=True, bbox=box_c),
                RecognizedDrink(drink_id=22, name="初星黒酢", playable=True, bbox=box_c),
            ],
        )
        acts_early = [
            LegalAction(choice_id=0, action_type="play_card", description="大声援+", bbox=box_c, payload={"card_id": 105, "name": "大声援+"}),
            LegalAction(choice_id=1, action_type="play_card", description="話題沸騰+", bbox=box_c, payload={"card_id": 776, "name": "話題沸騰+"}),
            LegalAction(choice_id=2, action_type="play_card", description="国民的アイドル", bbox=box_c, payload={"card_id": 162, "name": "国民的アイドル"}),
            LegalAction(choice_id=3, action_type="play_card", description="自然体の魅力+", bbox=box_c, payload={"card_id": 814, "name": "自然体の魅力+"}),
            LegalAction(choice_id=4, action_type="use_drink", description="使用饮料: ブーストエキス", bbox=box_c, payload={"drink_id": 12, "name": "ブーストエキス"}),
            LegalAction(choice_id=5, action_type="use_drink", description="使用饮料: センブリソーダ", bbox=box_c, payload={"drink_id": 14, "name": "センブリソーダ"}),
            LegalAction(choice_id=6, action_type="use_drink", description="使用饮料: 初星黒酢", bbox=box_c, payload={"drink_id": 22, "name": "初星黒酢"}),
            LegalAction(choice_id=7, action_type="end_turn", description="SKIP", bbox=box_c, payload={"name": "SKIP"}),
        ]
        cid_e, r_e, _ = planner.decide_battle_action(obs_early, acts_early, tracker=tr)
        assert cid_e == 0, f"Expected early hold to play #105, got {cid_e} ({r_e})"
        assert planner._sim is not None and planner._sim.in_unfresh_phase is False

        # 3b. rem=2 with armed 2x (double_card_effect_cards=1), #814 in hand, #12 available -> drink #12 first!
        planner._sim.double_card_effect_cards = 1
        obs_rem2 = obs_early.model_copy(
            update={
                "turn": 11,
                "turns_remaining": 2,
                "cards": [
                    RecognizedCard(card_id=814, name="自然体の魅力+", upgraded=True, playable=True, visual_playable=True, bbox=box_c),
                ],
                "drinks": [
                    RecognizedDrink(drink_id=12, name="ブーストエキス", playable=True, bbox=box_c),
                ],
            }
        )
        acts_rem2 = [
            LegalAction(choice_id=0, action_type="play_card", description="自然体の魅力+", bbox=box_c, payload={"card_id": 814, "name": "自然体の魅力+"}),
            LegalAction(choice_id=1, action_type="use_drink", description="使用饮料: ブーストエキス", bbox=box_c, payload={"drink_id": 12, "name": "ブーストエキス"}),
            LegalAction(choice_id=2, action_type="end_turn", description="SKIP", bbox=box_c, payload={"name": "SKIP"}),
        ]
        cid_pre, r_pre, _ = planner.decide_battle_action(obs_rem2, acts_rem2, tracker=tr)
        assert cid_pre == 1, f"Expected #12 pre-#814 score-buff drink first, got {cid_pre} ({r_pre})"
        assert ("[Pre-#814 ScoreBuff Drink #12]" in r_pre) or ("ブーストエキス" in r_pre)
    finally:
        if hasattr(agent, "close"):
            agent.close()
        db.close()


def test_empty_loadout_scan_returns_home_and_retries_cleanly(tmp_path: Path, monkeypatch):
    import demo.vision.loadout_scanner as ls_mod
    from demo.vision.loadout_scanner import HifLoadoutScanner, LoadoutScanResult

    monkeypatch.setattr(ls_mod.time, "sleep", lambda _s: None)
    db = GakumasDB("gakumas_library.db")
    lib_path = tmp_path / "retry_loadout_lib.json"
    agent = LocalHybridDecisionAgent(
        db=db,
        planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
        library_out_path=lib_path,
    )
    try:
        # 1. Empty-deck scan keeps configured p_idol_id=139/sense/[811, 813], returns Home, and retries cleanly
        tracker = GameStateTracker(plan="sense", p_idol_id=139, scenario_id="hif_honsen")
        tracker.produce_deck = [811, 813]
        assert tracker.initial_loadout_captured is False

        frame = np.full((1600, 720, 3), 240, dtype=np.uint8)
        obs_home = ScreenObservation(
            scene_type="home",
            plan="sense",
            scenario_id="hif_honsen",
            options=[
                {
                    "choice_id": 0,
                    "name": "プロデュース",
                    "title": "プロデュース",
                    "bbox": {"xmin": 211, "ymin": 1278, "xmax": 508, "ymax": 1360},
                }
            ],
        )
        obs_setup1 = ScreenObservation(
            scene_type="produce_setup",
            plan="sense",
            scenario_id="hif_honsen",
            week_activity="1.アイドル選択",
            options=[
                {"choice_id": 0, "name": "次へ", "title": "次へ", "bbox": {"xmin": 240, "ymin": 1368, "xmax": 480, "ymax": 1452}},
                {
                    "choice_id": 1,
                    "name": "アイドル詳細(長押し)",
                    "title": "アイドル詳細(長押し)",
                    "is_loadout_target": True,
                    "open_via_long_press": True,
                    "bbox": {"xmin": 170, "ymin": 1105, "xmax": 310, "ymax": 1265},
                },
            ],
        )

        scan_calls = []

        def _fake_scan(_ctrl, *, open_xy=(0, 0), open_via_long_press=True, plan="sense", **_kw):
            scan_calls.append((open_xy, open_via_long_press, plan))
            if len(scan_calls) == 1:
                return LoadoutScanResult(
                    p_idol_id=138,
                    plan="logic",
                    recommended_effect="motivation",
                    deck_card_ids=[],
                    p_items=[{"item_id": 40067, "name": "サポート"}],
                    screenshots_scanned=2,
                )
            return LoadoutScanResult(
                p_idol_id=139,
                idol_name="姫崎 莉波",
                title="ガラクタロード",
                plan="sense",
                recommended_effect="goodConditionTurns",
                deck_card_ids=[812, 814, 75, 163],
                p_items=[{"item_id": 9139, "name": "憧れ続けた輝き"}],
                screenshots_scanned=4,
            )

        state = {"stage": "setup1"}
        clicks = []
        snapshots = []

        class _RetryCtrl:
            is_connected = True

            def connect(self) -> bool:
                return True

            def disconnect(self) -> None:
                return None

            def screencap(self) -> np.ndarray:
                return frame

            def click(self, x: int, y: int) -> bool:
                clicks.append((x, y))
                if abs(x - 84) <= 10 and abs(y - 1532) <= 10:
                    state["stage"] = "home"
                elif state["stage"] == "home":
                    state["stage"] = "setup2"
                elif abs(x - 360) <= 30 and abs(y - 1410) <= 30:
                    state["stage"] = "confirmed"
                return True

        runner = GakumasRunner(
            mode="mock",
            controller=_RetryCtrl(),  # type: ignore[arg-type]
            agent=agent,
            db=db,
            tracker=tracker,
            plan="sense",
            scenario_id="hif_honsen",
            p_idol_id=139,
            session_mode=False,
            retry_home_on_empty_loadout=True,
            max_loadout_home_retries=2,
            loadout_scanner=SimpleNamespace(scan_via_controller=_fake_scan),
            loadout_library_path=lib_path,
            on_step_completed=lambda res: snapshots.append(
                (
                    res.step_idx,
                    tracker.p_idol_id,
                    tracker.plan,
                    list(tracker.produce_deck),
                    tracker.initial_loadout_captured,
                    runner._loadout_lp_attempts,
                )
            ),
        )
        runner.scene_routing_reco = SimpleNamespace(
            recognize=lambda _fr: obs_home if state["stage"] == "home" else obs_setup1
        )
        # Simulate runner having seen setup previously so Home exit guard matters if flag breaks
        runner._has_entered_produce = False
        results = runner.run_loop(max_steps=3, delay_between_steps=0.0, stop_on_home_return=True)
        assert len(results) == 3
        assert results[0].click_coords == (84, 1532)
        assert results[0].action_taken is not None and results[0].action_taken.payload.get("name") == "ホーム"
        assert snapshots[0] == (1, 139, "sense", [811, 813], False, 0)
        assert results[1].scene_type == "home"
        assert results[1].action_taken is not None and results[1].action_taken.payload.get("name") == "プロデュース"
        assert results[2].action_taken is not None and results[2].action_taken.payload.get("name") == "次へ"
        assert snapshots[2] == (3, 139, "sense", [812, 814, 75, 163], True, 1)
        assert len(scan_calls) == 2
        assert state["stage"] == "confirmed"

        # 2. _parse_header_metadata disambiguates shared title 'ガラクタロード' by default_plan='sense'
        scanner = HifLoadoutScanner(db=db)
        try:
            scanner.ocr = SimpleNamespace(
                recognize_array=lambda *_a, **_kw: {"text": "ガラクタロード", "lines": [{"text": "ガラクタロード", "bbox": [200, 40, 440, 72]}]}
            )
            scanner.reco.en_ocr = SimpleNamespace(recognize_array=lambda *_a, **_kw: {"text": "", "lines": []})
            meta = scanner._parse_header_metadata(frame, default_plan="sense")
            assert meta["p_idol_id"] == 139
            assert meta["plan"] == "sense"

            # 3. scan_via_controller stuck in icon-grid view (never shows list_markers) closes modal with 0 swipes
            st_cnt = {"in_modal": False, "clicks": [], "swipes": 0}

            class _StuckGridCtrl:
                def long_press(self, _x: int, _y: int, duration_ms: int = 950) -> bool:
                    st_cnt["in_modal"] = True
                    return True

                def click(self, x: int, y: int) -> bool:
                    st_cnt["clicks"].append((x, y))
                    if abs(x - 88) <= 25 and abs(y - 1532) <= 25:
                        st_cnt["in_modal"] = False
                    return True

                def swipe(self, *args, **kwargs) -> bool:
                    st_cnt["swipes"] += 1
                    return True

                def screencap(self) -> np.ndarray:
                    if st_cnt["in_modal"]:
                        fr = np.full((1600, 720, 3), 245, dtype=np.uint8)
                        return fr
                    return frame

            scanner.ocr = SimpleNamespace(
                recognize_array=lambda *_a, **_kw: (
                    {"text": "基本情報 スキルカード", "lines": []}
                    if st_cnt["in_modal"]
                    else {"text": "1.アイドル選択 次へ", "lines": []}
                )
            )
            stuck_res = scanner.scan_via_controller(_StuckGridCtrl(), open_xy=(240, 1185), open_via_long_press=True, plan="sense")  # type: ignore[arg-type]
            assert any(abs(cx - 648) <= 5 and abs(cy - 1312) <= 5 for cx, cy in st_cnt["clicks"])
            assert any(abs(bx - 88) <= 5 and abs(by - 1532) <= 5 for bx, by in st_cnt["clicks"])
            assert st_cnt["swipes"] == 0
            assert stuck_res.deck_card_ids == []
            assert stuck_res.p_items == []
            assert stuck_res.screenshots_scanned == 0
        finally:
            scanner.close()
    finally:
        if hasattr(agent, "close"):
            agent.close()
        db.close()


def test_jugyo_card_reward_selected_orange_receive_not_misclassified_as_event(monkeypatch):
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    rec = LiveScreenRecognizer(db=db, plan="sense", scenario_id="hif_honsen")
    try:
        frame = np.full((1600, 720, 3), 242, dtype=np.uint8)
        # Card slots in y=1000..1148: slots 0 & 1 have dark-ink card art (slot 0 selected with top/bottom orange bracket),
        # while slot 2 has bright orange/gold art (filtered by ~is_orange_halo unless symmetric 3rd-slot recovery triggers)
        frame[1005:1143, 157:280] = (95, 115, 195)
        frame[1005:1143, 296:418] = (120, 145, 200)
        frame[1005:1143, 435:558] = (252, 165, 45)
        frame[1035:1110, 465:525] = (250, 235, 120)
        # Selected orange bracket around slot 0
        frame[988:1002, 165:272] = (250, 145, 30)
        frame[1146:1158, 165:272] = (250, 145, 30)
        # Bottom active orange 受け取る button at y=1444..1514, x=230..490
        frame[1444:1514, 230:490] = (252, 132, 32)
        frame[1468:1490, 315:405] = (255, 255, 255)

        full_lines = [
            {"text": "授業", "bbox": [45, 42, 120, 76]},
            {"text": "大胆不敵", "bbox": [230, 620, 380, 656]},
            {"text": "眠気を山札のランダムな位置に生成", "bbox": [230, 735, 620, 770]},
            {"text": "重複不可", "bbox": [115, 895, 240, 935]},
            {"text": ")まじ", "bbox": [115, 1100, 195, 1140]},
        ]

        def _mock_ocr_arr(arr, *_a, **_kw):
            ah, aw = arr.shape[:2]
            if (ah, aw) == (1600, 720):
                return {"text": " ".join(l["text"] for l in full_lines), "lines": full_lines}
            if ah >= 600 and aw >= 900:
                return {
                    "text": "重複不可 )まじ",
                    "lines": [
                        {"text": "重複不可", "bbox": [25, 25, 180, 65]},
                        {"text": ")まじ", "bbox": [25, 420, 120, 460]},
                    ],
                }
            if ah < 260 and aw < 360:
                return {"text": "受け取る", "lines": [{"text": "受け取る", "bbox": [20, 10, 140, 42]}]}
            return {"text": "大胆不敵\n重複不可", "lines": [{"text": "大胆不敵", "bbox": [30, 20, 210, 56]}]}

        monkeypatch.setattr(rec.ocr, "recognize_array", _mock_ocr_arr)
        monkeypatch.setattr(rec.en_ocr, "recognize_array", lambda *_a, **_kw: {"text": "", "lines": []})
        monkeypatch.setattr(
            rec,
            "_ocr_bottom_orange_button",
            lambda *_a, **_kw: {
                "name": "受け取る",
                "title": "受け取る",
                "bbox": {"xmin": 230, "ymin": 1444, "xmax": 490, "ymax": 1514},
            },
        )

        # Verify _find_event_choice_pills rejects 重複不可 and )まじ
        assert rec._find_event_choice_pills(frame, full_lines, 720, 1600) == []

        # Verify symmetric 3rd-slot recovery finds all 3 card slots including bright orange slot 2
        slots = rec._scan_reward_slot_band(frame, 720, 1600, 1000, 1148)
        assert len(slots) == 3
        assert abs(((slots[2][0] + slots[2][2]) // 2) - 496) <= 10

        obs = rec.recognize(frame, plan="sense")
        assert obs.scene_type == "card_reward"
        opt_names = [str(o.get("name") or "") for o in (obs.options or [])]
        assert "受け取る" in opt_names
        assert obs.options[0]["name"] == "受け取る"
        assert not any("重複不可" in nm or "まじ" in nm for nm in opt_names)
        assert any("第3张候选卡" in nm or "候选卡3" in nm for nm in opt_names)
    finally:
        db.close()


def test_live_planner_hand_cap_5_and_exile_nemuki_over_skip():
    db = GakumasDB("gakumas_library.db")
    agent = LocalHybridDecisionAgent(
        db=db,
        planner_cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=4),
    )
    try:
        tr = GameStateTracker(plan="sense", p_idol_id=139, scenario_id="hif_honsen")
        tr.set_initial_loadout(
            p_idol_id=139,
            plan="sense",
            recommended_effect="goodConditionTurns",
            deck_ids=[812, 814, 162, 163, 86, 294, 776, 105, 113, 23],
            p_items=[{"item_id": 447}, {"item_id": 9139}],
            db=db,
        )
        box_c = BoundingBox(xmin=100, ymin=1050, xmax=260, ymax=1350)
        planner = agent.battle_planner

        # 1. rem=6 with only [#814 (held), #163 (held), #23 眠気] -> must exile #23 眠気, NOT blank SKIP!
        obs_exile = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=9,
            turn=7,
            turns_remaining=6,
            stamina=40,
            max_stamina=44,
            idol_stats={"vocal": 2000, "dance": 2400, "visual": 2000},
            buffs={"goodConditionTurns": 14, "perfectConditionTurns": 2, "concentration": 10},
            cards=[
                RecognizedCard(card_id=814, name="自然体の魅力+", upgraded=True, playable=True, visual_playable=True, bbox=box_c),
                RecognizedCard(card_id=163, name="国民的アイドル+", upgraded=True, playable=True, visual_playable=True, bbox=box_c),
                RecognizedCard(card_id=23, name="眠気", upgraded=False, playable=True, visual_playable=True, bbox=box_c),
            ],
            drinks=[],
        )
        acts_exile = [
            LegalAction(choice_id=0, action_type="play_card", description="自然体の魅力+", bbox=box_c, payload={"card_id": 814, "name": "自然体の魅力+"}),
            LegalAction(choice_id=1, action_type="play_card", description="国民的アイドル+", bbox=box_c, payload={"card_id": 163, "name": "国民的アイドル+"}),
            LegalAction(choice_id=2, action_type="play_card", description="眠気", bbox=box_c, payload={"card_id": 23, "name": "眠気"}),
            LegalAction(choice_id=3, action_type="end_turn", description="SKIP", bbox=box_c, payload={"name": "SKIP"}),
        ]
        cid_ex, r_ex, _ = planner.decide_battle_action(obs_exile, acts_exile, tracker=tr)
        assert cid_ex == 2, f"Expected #23 眠気 exile instead of blank SKIP, got {cid_ex} ({r_ex})"

        # 2. rem=2 with 4 cards [#86 free, #814, #162, #294] + #14 (draws 2 -> would overflow 5-card hand): play a card first!
        obs_4cap = obs_exile.model_copy(
            update={
                "turn": 11,
                "turns_remaining": 2,
                "cards": [
                    RecognizedCard(card_id=86, name="シュプレヒコール", upgraded=False, playable=True, visual_playable=True, bbox=box_c),
                    RecognizedCard(card_id=814, name="自然体の魅力+", upgraded=True, playable=True, visual_playable=True, bbox=box_c),
                    RecognizedCard(card_id=162, name="国民的アイドル", upgraded=False, playable=True, visual_playable=True, bbox=box_c),
                    RecognizedCard(card_id=294, name="スポットライト", upgraded=False, playable=True, visual_playable=True, bbox=box_c),
                ],
                "drinks": [
                    RecognizedDrink(drink_id=14, name="センブリソーダ", playable=True, bbox=box_c),
                ],
            }
        )
        acts_4cap = [
            LegalAction(choice_id=0, action_type="play_card", description="シュプレヒコール", bbox=box_c, payload={"card_id": 86, "name": "シュプレヒコール"}),
            LegalAction(choice_id=1, action_type="play_card", description="自然体の魅力+", bbox=box_c, payload={"card_id": 814, "name": "自然体の魅力+"}),
            LegalAction(choice_id=2, action_type="play_card", description="国民的アイドル", bbox=box_c, payload={"card_id": 162, "name": "国民的アイドル"}),
            LegalAction(choice_id=3, action_type="play_card", description="スポットライト", bbox=box_c, payload={"card_id": 294, "name": "スポットライト"}),
            LegalAction(choice_id=4, action_type="use_drink", description="使用饮料: センブリソーダ", bbox=box_c, payload={"drink_id": 14, "name": "センブリソーダ"}),
            LegalAction(choice_id=5, action_type="end_turn", description="SKIP", bbox=box_c, payload={"name": "SKIP"}),
        ]
        cid_4cap, r_4cap, _ = planner.decide_battle_action(obs_4cap, acts_4cap, tracker=tr)
        assert acts_4cap[cid_4cap].action_type == "play_card", f"Expected card play before #14 drawCard(2) on 4-card hand, got {cid_4cap} ({r_4cap})"

        # 3. 5 cards in hand + #22 初星黒酢: must NOT drink #22 while hand is at 5-card limit
        obs_5cap = obs_4cap.model_copy(
            update={
                "cards": [
                    RecognizedCard(card_id=86, name="シュプレヒコール", upgraded=False, playable=True, visual_playable=True, bbox=box_c),
                    RecognizedCard(card_id=814, name="自然体の魅力+", upgraded=True, playable=True, visual_playable=True, bbox=box_c),
                    RecognizedCard(card_id=162, name="国民的アイドル", upgraded=False, playable=True, visual_playable=True, bbox=box_c),
                    RecognizedCard(card_id=294, name="スポットライト", upgraded=False, playable=True, visual_playable=True, bbox=box_c),
                    RecognizedCard(card_id=105, name="大声援+", upgraded=True, playable=True, visual_playable=True, bbox=box_c),
                ],
                "drinks": [
                    RecognizedDrink(drink_id=22, name="初星黒酢", playable=True, bbox=box_c),
                ],
            }
        )
        acts_5cap = [
            LegalAction(choice_id=0, action_type="play_card", description="シュプレヒコール", bbox=box_c, payload={"card_id": 86, "name": "シュプレヒコール"}),
            LegalAction(choice_id=1, action_type="play_card", description="自然体の魅力+", bbox=box_c, payload={"card_id": 814, "name": "自然体の魅力+"}),
            LegalAction(choice_id=2, action_type="play_card", description="国民的アイドル", bbox=box_c, payload={"card_id": 162, "name": "国民的アイドル"}),
            LegalAction(choice_id=3, action_type="play_card", description="スポットライト", bbox=box_c, payload={"card_id": 294, "name": "スポットライト"}),
            LegalAction(choice_id=4, action_type="play_card", description="大声援+", bbox=box_c, payload={"card_id": 105, "name": "大声援+"}),
            LegalAction(choice_id=5, action_type="use_drink", description="使用饮料: 初星黒酢", bbox=box_c, payload={"drink_id": 22, "name": "初星黒酢"}),
            LegalAction(choice_id=6, action_type="end_turn", description="SKIP", bbox=box_c, payload={"name": "SKIP"}),
        ]
        cid_5cap, r_5cap, _ = planner.decide_battle_action(obs_5cap, acts_5cap, tracker=tr)
        assert acts_5cap[cid_5cap].action_type == "play_card", f"Expected card play instead of #22 at 5-card hand cap, got {cid_5cap} ({r_5cap})"
    finally:
        if hasattr(agent, "close"):
            agent.close()
        db.close()


def test_change_and_delete_grid_not_hijacked_by_upgrade_preview_button_and_first_shop_probe(monkeypatch, tmp_path):
    """Regression test:
    1) After 1st click in チェンジ/削除, preview box displays '強化確認'/'カスタマイズ確認'.
       Must NOT flip is_upgrade_flow=True and must expose all 18-24 slots (not truncated at 16).
    2) First shop (W6 相談) always probes 削除 once even if tracker.produce_deck didn't record #23 眠気 yet,
       and scans past non-trouble cards to find #23 眠気.
    3) build_local_hif_runner(initial_library=web_config_json) loads preferences + memory hooks and keeps
       loadout_scanner active (hybrid) when initial_loadout_captured is False.
    4) Per-idol stat specialty (Misuzu Vo>Vi>Da, Rinami Vi>Da>Vo) and #22 blocked at 5-card hand cap.
    """
    import json
    import numpy as np
    from demo.core.db import GakumasDB
    from demo.core.idol_guides import get_idol_stat_order
    from demo.core.tracker import GameStateTracker
    from demo.agent.local_agent import LocalHybridDecisionAgent
    from demo.run_hif_local import build_local_hif_runner
    from demo.schemas import BoundingBox, LegalAction, RecognizedCard, RecognizedDrink, ScreenObservation
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    rec = LiveScreenRecognizer()
    try:
        h, w = 1600, 720
        frame = np.full((h, w, 3), 24, dtype=np.uint8)
        frame[int(h * 0.145) : int(h * 0.345), int(w * 0.02) : int(w * 0.98)] = 240
        cols = ((0.112, 0.278), (0.322, 0.488), (0.532, 0.698), (0.742, 0.908))
        rows = (
            (0.402, 0.476),
            (0.496, 0.568),
            (0.589, 0.662),
            (0.684, 0.756),
            (0.776, 0.849),
        )
        for r_i, (y1f, y2f) in enumerate(rows):
            for c_i, (x1f, x2f) in enumerate(cols):
                if r_i == 4 and c_i >= 2:
                    frame[int(h * y1f) : int(h * y2f), int(w * x1f) : int(w * x2f)] = 88
                else:
                    frame[int(h * y1f) : int(h * y2f), int(w * x1f) : int(w * x2f)] = [215, 195, 225]
        frame[int(h * 0.91) : int(h * 0.96), int(w * 0.52) : int(w * 0.90)] = [245, 145, 45]

        def _fake_ocr(arr):
            if arr.shape[0] == h and arr.shape[1] == w:
                return {
                    "text": "チェンジ\n天賦の才+\nランダムな山札か捨札にあるスキルカードを山札の一番上に移動\n強化確認\nカスタマイズ確認\nチェンジするスキルカードを選択してください\nキャンセル\nチェンジ",
                    "lines": [
                        {"text": "チェンジ", "bbox": [55, 26, 210, 62]},
                        {"text": "天賦の才+", "bbox": [195, 265, 440, 305]},
                        {"text": "ランダムな山札か捨札にあるスキルカードを山札の一番上に移動", "bbox": [195, 360, 660, 405]},
                        {"text": "強化確認", "bbox": [45, 430, 165, 462]},
                        {"text": "カスタマイズ確認", "bbox": [45, 488, 175, 518]},
                        {"text": "チェンジするスキルカードを選択してください", "bbox": [110, 585, 610, 620]},
                        {"text": "キャンセル", "bbox": [110, 1468, 290, 1510]},
                        {"text": "チェンジ", "bbox": [455, 1468, 610, 1510]},
                    ],
                }
            return {
                "text": "鳴り止まない拍手\n集中+4\n好調2ターン",
                "lines": [
                    {"text": "鳴り止まない拍手", "bbox": [195, 40, 440, 75]},
                    {"text": "集中+4", "bbox": [210, 95, 340, 125]},
                ],
            }

        monkeypatch.setattr(rec.ocr, "recognize_array", _fake_ocr)
        obs = rec.recognize(frame, plan="sense")
        assert obs.scene_type == "card_reward"
        assert obs.options[0]["name"] == "チェンジする", f"Expected チェンジする, got {obs.options[0]}"
        grid_slots = [o for o in obs.options if str(o.get("name", "")).startswith("预览第")]
        assert len(grid_slots) == 18, f"Expected all 18 selectable slots (not truncated at 16), got {len(grid_slots)}"
        row0_col3 = [o for o in grid_slots if o["bbox"]["ymin"] < int(h * 0.45) and o["bbox"]["xmin"] > int(w * 0.70)]
        assert row0_col3, "Missing top-right slot with corrected X/Y coordinates"
    finally:
        rec.db.close()

    db = GakumasDB("gakumas_library.db")
    agent = LocalHybridDecisionAgent(db=db)
    try:
        assert get_idol_stat_order(163, db=db) == ("vocal", "visual", "dance")
        assert get_idol_stat_order(139, db=db) == ("visual", "dance", "vocal")

        tr = GameStateTracker(plan="sense")
        tr.p_idol_id = 139
        tr.produce_deck = [814, 161, 159, 75, 105]
        agent.bind_tracker(tr)
        _b = BoundingBox(xmin=0, ymin=0, xmax=100, ymax=100)
        shop_obs = ScreenObservation(
            scene_type="shop",
            week=6,
            p_points=260,
            raw_text="相談 強化 100 削除 125 終了",
        )
        shop_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="強化", bbox=_b, payload={"name": "強化", "shop_slot_key": "強化", "price": 100}),
            LegalAction(choice_id=1, action_type="pick_option", description="削除", bbox=_b, payload={"name": "削除", "shop_slot_key": "削除", "price": 125}),
            LegalAction(choice_id=2, action_type="pick_option", description="終了", bbox=_b, payload={"name": "終了", "shop_slot_key": "終了"}),
        ]
        d_shop = agent.decide("", shop_acts, obs=shop_obs)
        assert d_shop.choice_id == 1, f"First shop must probe 削除 once even without tracked #23, got {d_shop}"

        del_obs = ScreenObservation(
            scene_type="card_reward",
            week=6,
            raw_text="削除\n表現の基本\n強化確認\nカスタマイズ確認",
        )
        del_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="削除する 表現の基本", bbox=_b, payload={"name": "削除する", "card_id": 44, "slot_idx": 0, "title": "表現の基本"}),
            LegalAction(choice_id=1, action_type="pick_option", description="预览第2张待选卡", bbox=_b, payload={"name": "预览第2张待选卡", "slot_idx": 1, "title": "暗灰卡"}),
            LegalAction(choice_id=2, action_type="pick_option", description="キャンセル", bbox=_b, payload={"name": "キャンセル"}),
        ]
        d_del = agent.decide("", del_acts, obs=del_obs)
        assert d_del.choice_id == 1, f"Should scan slot 1 for untracked T card before deleting #44, got {d_del}"

        # 5-card hand cap hard guard on drink #22 even when only 4 cards OCR'd
        obs_5slots = ScreenObservation(
            scene_type="battle_turn",
            week=7,
            turn=2,
            turns_remaining=10,
            stamina=40,
            max_stamina=50,
            hand_slot_count=5,
            cards=[
                RecognizedCard(card_id=86, name="シュプレヒコール", playable=True, bbox=_b),
                RecognizedCard(card_id=814, name="自然体の魅力+", playable=True, bbox=_b),
                RecognizedCard(card_id=162, name="国民的アイドル", playable=True, bbox=_b),
                RecognizedCard(card_id=294, name="スポットライト", playable=True, bbox=_b),
            ],
            drinks=[RecognizedDrink(drink_id=22, name="初星黒酢", playable=True, bbox=_b)],
        )
        acts_5slots = [
            LegalAction(choice_id=0, action_type="play_card", description="シュプレヒコール", bbox=_b, payload={"card_id": 86, "name": "シュプレヒコール"}),
            LegalAction(choice_id=1, action_type="play_card", description="自然体の魅力+", bbox=_b, payload={"card_id": 814, "name": "自然体の魅力+"}),
            LegalAction(choice_id=2, action_type="play_card", description="国民的アイドル", bbox=_b, payload={"card_id": 162, "name": "国民的アイドル"}),
            LegalAction(choice_id=3, action_type="play_card", description="スポットライト", bbox=_b, payload={"card_id": 294, "name": "スポットライト"}),
            LegalAction(choice_id=4, action_type="use_drink", description="使用饮料: 初星黒酢", bbox=_b, payload={"drink_id": 22, "name": "初星黒酢"}),
            LegalAction(choice_id=5, action_type="end_turn", description="SKIP", bbox=_b, payload={"name": "SKIP"}),
        ]
        cid_5, r_5, _ = agent.battle_planner.decide_battle_action(obs_5slots, acts_5slots, tracker=tr)
        assert cid_5 != 4, f"Must NEVER drink #22 when hand_slot_count==5, got {cid_5} ({r_5})"
    finally:
        agent.close()
        db.close()

    cfg_file = tmp_path / "web_cfg.json"
    cfg_file.write_text(
        json.dumps(
            {
                "initial_loadout_captured": False,
                "p_idol_id": 139,
                "plan": "sense",
                "deck_ids": [812, 814],
                "card_memory_hooks": {"75": {"extra_use_limit": 2}, "593": {"exiles_trouble_on_play": True}},
                "preferences": {
                    "priority_card_ids": [161, 159, 75],
                    "avoid_card_ids": [169, 115],
                    "keep_card_ids": [147, 159, 161],
                    "banned_drink_ids": [19, 1, 2],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    r = build_local_hif_runner(initial_library=str(cfg_file))
    try:
        assert r.plan == "sense"
        assert r.p_idol_id == 139
        assert r.loadout_scanner is not None
        assert r.tracker.initial_loadout_captured is False
        assert 75 in r.tracker.card_memory_hooks and 593 in r.tracker.card_memory_hooks
        assert r.agent.preferences.priority_card_ids == [161, 159, 75]
        assert r.agent.preferences.banned_drink_ids == [19, 1, 2]
    finally:
        r.agent.close()
        if r.loadout_scanner is not None:
            r.loadout_scanner.close()
        r.db.close()


def test_w6_shop_tooltip_does_not_hijack_upgrade_delete_and_sense_orange_cards_detected(monkeypatch):
    """Real-ADB regression tests discovered during live8 run on 127.0.0.1:16384:
    1) W6 相談 white tooltip '持っているスキルカードの強化や削除ができます。' at y=1040..1073 must NOT
       hijack 強化/削除 click bboxes; visual fallback must target the vivid pink 強化 & blue 削除
       buttons at y=1088..1168 (center y~1128) and parse price from noisy OCR like '′⑨125'.
    2) Sense 3-card reward icons with orange silhouettes on white backgrounds must NOT be erased by
       is_orange_halo or falsely marked selected when no exterior orange bracket exists.
    """
    import numpy as np
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    rec = LiveScreenRecognizer()
    try:
        # 1. W6 相談 frame with white tooltip banner at y=1040..1073 + pink 強化 & blue 削除 buttons at y=1090..1168
        h, w = 1600, 720
        shop_frame = np.full((h, w, 3), 215, dtype=np.uint8)
        # White tooltip pill
        shop_frame[1035:1078, 90:630] = [235, 230, 222]
        # Vivid pink-orange 強化 button (left)
        shop_frame[1090:1168, 50:353] = [245, 175, 150]
        # Vivid purple-blue 削除 button (right)
        shop_frame[1090:1168, 366:670] = [160, 195, 238]

        shop_lines = [
            {"text": "相談", "bbox": [70, 35, 150, 70]},
            {"text": "持っているスキルカードの強化や削除が0ます。", "bbox": [118, 1040, 590, 1073]},
            {"text": "′⑨125", "bbox": [188, 1120, 293, 1140]},
            {"text": "′⑨125", "bbox": [520, 1120, 625, 1140]},
            {"text": "交換する", "bbox": [327, 1232, 435, 1258]},
            {"text": "終了", "bbox": [585, 1230, 680, 1264]},
        ]
        meta_shop = {
            "week": 6,
            "total_weeks": 9,
            "week_label": "6日目",
            "stats": {"vocal": 1400, "dance": 1800, "visual": 2400},
            "p_points": 300,
            "stamina": 36,
            "max_stamina": 44,
            "drinks": [],
        }
        monkeypatch.setattr(rec.ocr, "recognize_array", lambda _arr: {"text": "", "lines": []})
        monkeypatch.setattr(rec.en_ocr, "recognize_array", lambda _arr: {"text": "", "lines": []})
        obs_shop = rec._parse_shop_screen(shop_frame, shop_lines, "sense", meta_shop)
        assert obs_shop.scene_type == "shop"
        by_name = {o["name"]: o for o in obs_shop.options}
        assert "強化" in by_name and "削除" in by_name
        upg_cy = (by_name["強化"]["bbox"]["ymin"] + by_name["強化"]["bbox"]["ymax"]) // 2
        del_cy = (by_name["削除"]["bbox"]["ymin"] + by_name["削除"]["bbox"]["ymax"]) // 2
        assert 1090 <= upg_cy <= 1168, f"強化 center y={upg_cy} missed real pink button (1090..1168)!"
        assert 1090 <= del_cy <= 1168, f"削除 center y={del_cy} missed real blue button (1090..1168)!"
        assert by_name["強化"].get("price") == 125
        assert by_name["削除"].get("price") == 125

        # 2. Sense 3-card reward modal with orange artwork on white cards and NO selection bracket
        rew_frame = np.full((h, w, 3), 255, dtype=np.uint8)
        for x1, x2 in ((157, 285), (296, 424), (435, 563)):
            # Thin border + pure orange interior artwork (R=245, G=150, B=55)
            rew_frame[1005:1140, x1:x2] = [252, 252, 252]
            rew_frame[1020:1120, x1 + 20 : x2 - 20] = [245, 150, 55]
        slots = rec._scan_reward_slot_band(rew_frame, w, h, int(h * 0.625), int(h * 0.718))
        assert len(slots) == 3, f"Expected all 3 Sense orange-artwork slots, got {slots}"
        assert not any(s[4] for s in slots), f"No slot should be marked selected without exterior brackets: {slots}"
    finally:
        rec.db.close()


def test_shop_red_price_clamps_stale_p_points_and_refresh_noop_prevents_loop(monkeypatch):
    """Regression test for shop stall at low P-points (e.g. 155 -> 5P):
    1) Red unaffordable prices on the grid or リフレッシュ pill clamp stale p_pts (< price) and hide disabled リフレッシュ.
    2) If clicking リフレッシュ does not reduce p_pts on the next shop frame, LocalHybridDecisionAgent marks リフレッシュ
       as no-op instead of resetting _shop_noop_features/_customize_done_count into an infinite loop.
    """
    import numpy as np
    from demo.core.db import GakumasDB
    from demo.agent.local_agent import LocalHybridDecisionAgent
    from demo.schemas import BoundingBox, LegalAction, ScreenObservation
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    rec = LiveScreenRecognizer()
    try:
        h, w = 1600, 720
        frame = np.full((h, w, 3), 235, dtype=np.uint8)
        # Grayed-out リフレッシュ pill at y=0.754..0.779 + red '20' price digits right underneath it
        frame[int(h * 0.754) : int(h * 0.779), int(w * 0.026) : int(w * 0.276)] = [190, 190, 190]
        frame[int(h * 0.782) : int(h * 0.802), int(w * 0.12) : int(w * 0.22)] = [235, 60, 65]
        lines = [
            {"text": "インターバル", "bbox": [45, 30, 190, 65]},
            {"text": "リフレッシュ", "bbox": [35, 1212, 185, 1242]},
            {"text": "20", "bbox": [110, 1255, 165, 1280]},
            {"text": "カスタマイズ", "bbox": [350, 1065, 520, 1105]},
            {"text": "終了", "bbox": [585, 1225, 680, 1265]},
        ]
        meta_stale = {
            "week": 8,
            "total_weeks": 9,
            "week_label": "8日目",
            "stats": {"vocal": 1400, "dance": 1800, "visual": 2400},
            "p_points": 155,  # stale pre-spend P-points
            "stamina": 44,
            "max_stamina": 44,
            "drinks": [],
        }
        monkeypatch.setattr(rec.ocr, "recognize_array", lambda _arr: {"text": "", "lines": []})
        monkeypatch.setattr(rec.en_ocr, "recognize_array", lambda _arr: {"text": "", "lines": []})
        obs = rec._parse_shop_screen(frame, lines, "sense", meta_stale)
        assert obs.p_points <= 19, f"Red 20P refresh price must clamp stale 155P to <=19, got {obs.p_points}"
        assert "リフレッシュ" not in [o.get("name") for o in obs.options]
    finally:
        rec.db.close()

    db = GakumasDB("gakumas_library.db")
    agent = LocalHybridDecisionAgent(db=db)
    try:
        _b = BoundingBox(xmin=0, ymin=0, xmax=100, ymax=100)
        shop_obs = ScreenObservation(
            scene_type="shop",
            week=8,
            total_weeks=9,
            scenario_id="hif_honsen",
            p_points=155,
            raw_text="インターバル リフレッシュ 20 終了",
        )
        acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="リフレッシュ 20P", bbox=_b, payload={"name": "リフレッシュ", "shop_slot_key": "リフレッシュ", "price": 20}),
            LegalAction(choice_id=1, action_type="pick_option", description="終了", bbox=_b, payload={"name": "終了", "shop_slot_key": "終了"}),
        ]
        d1 = agent.decide("", acts, obs=shop_obs)
        assert d1.choice_id == 0
        # Repeated unchanged-p_points refresh frames => disabled/no-op refresh is blocked and exits!
        _ = agent.decide("", acts, obs=shop_obs)
        _ = agent.decide("", acts, obs=shop_obs)
        d4 = agent.decide("", acts, obs=shop_obs)
        assert d4.choice_id == 1, f"Must exit after repeated no-op refresh instead of looping, got {d4}"
        assert "リフレッシュ" in agent._shop_noop_features
    finally:
        agent.close()
        db.close()


def test_free_action_card_protected_and_avoid_scan_and_battle_unconfirmed_play_guard(monkeypatch):
    """Regression tests for:
    1) Free-action Active card #87 シュプレヒコール+ (スキルカード使用数追加+1) is protected during チェンジ
       (never treated as ordinary throwaway Active card).
    2) During チェンジ when user avoid_card_ids (#81 祝福+) is in deck, agent keeps scanning past weak cards
       until it locates the avoided card (+5000).
    3) Post-Battle-1 W8 shop prioritizes カスタマイズ & Pドリンク over 強化.
    4) If a live battle play_card tap is swallowed by an animation (hand pixels unchanged even after retry),
       GakumasRunner does NOT call tracker.record_card_play() so limit_count=1 cards aren't falsely exhausted.
    """
    import numpy as np
    from demo.core.db import GakumasDB
    from demo.core.hif_catalog import ProducePreferences
    from demo.core.tracker import GameStateTracker
    from demo.agent.local_agent import LocalHybridDecisionAgent
    from demo.maa_bridge.adb_controller import AdbController
    from demo.maa_bridge.runner import GakumasRunner
    from demo.schemas import BoundingBox, DecisionResult, LegalAction, RecognizedCard, ScreenObservation

    db = GakumasDB("gakumas_library.db")
    prefs = ProducePreferences(
        avoid_card_ids=[81],
        keep_card_ids=[147, 159, 161, 812, 814],
    )
    agent = LocalHybridDecisionAgent(db=db, preferences=prefs)
    try:
        tr = GameStateTracker(plan="sense", p_idol_id=139, scenario_id="hif_honsen")
        tr.produce_deck = [812, 814, 87, 81, 68]
        agent.bind_tracker(tr)

        # 1. #87 シュプレヒコール+ must have strongly negative removal score during チェンジ
        s87 = agent._score_skill_card(
            card_id=87,
            title="シュプレヒコール+",
            desc="集中消費2 / パラメータ+6 / 好調3ターン / スキルカード使用数追加+1 / 重複不可",
            target_effect="goodConditionTurns",
            guide=None,
            plan="sense",
            for_removal=True,
            p_idol_id=139,
            ignore_upgrade=True,
        )
        assert s87 < -1_000, f"#87 シュプレヒコール+ (free extra action) must be protected, got {s87}"

        # 2. During チェンジ with #81 (avoid_card_ids) in deck, seeing weak #68 (+1100) must NOT confirm yet;
        #    must scan slot 1 to find #81 (+5000)!
        _b = BoundingBox(xmin=0, ymin=0, xmax=100, ymax=100)
        chg_obs = ScreenObservation(scene_type="card_reward", plan="sense", week=1, raw_text="チェンジするスキルカードを選択してください")
        chg_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="チェンジする #68", bbox=_b, payload={"name": "チェンジする", "card_id": 68, "slot_idx": 0, "title": "トークタイム"}),
            LegalAction(choice_id=1, action_type="pick_option", description="预览第2张待选卡", bbox=_b, payload={"name": "预览第2张待选卡", "slot_idx": 1}),
            LegalAction(choice_id=2, action_type="pick_option", description="キャンセル", bbox=_b, payload={"name": "キャンセル"}),
        ]
        d_chg = agent.decide("", chg_acts, obs=chg_obs)
        assert d_chg.choice_id == 1, f"Must keep scanning for avoid_card_ids #81 instead of stopping on #68, got {d_chg}"

        # 3. W8 final shop fills all 4 drinks FIRST before allowing カスタマイズ; once 4/4 drinks full, カスタマイズ > 強化
        w8_obs = ScreenObservation(scene_type="shop", plan="sense", scenario_id="hif_honsen", week=8, p_points=300, drinks=[])
        w8_acts = [
            LegalAction(choice_id=0, action_type="pick_option", description="強化 100P", bbox=_b, payload={"name": "強化", "shop_slot_key": "強化", "price": 100}),
            LegalAction(choice_id=1, action_type="pick_option", description="Pドリンク1 75P", bbox=_b, payload={"name": "Pドリンク1", "shop_slot_key": "Pドリンク:0", "price": 75}),
            LegalAction(choice_id=2, action_type="pick_option", description="カスタマイズ", bbox=_b, payload={"name": "カスタマイズ", "shop_slot_key": "カスタマイズ"}),
            LegalAction(choice_id=3, action_type="pick_option", description="終了", bbox=_b, payload={"name": "終了"}),
        ]
        assert agent.decide("", w8_acts, obs=w8_obs).choice_id == 1
        agent._shop_clicked_slots.clear()
        agent._shop_noop_features.clear()
        w8_obs_4 = w8_obs.model_copy(
            update={
                "drinks": [
                    RecognizedDrink(drink_id=22, name="初星黒酢", bbox=_b),
                    RecognizedDrink(drink_id=18, name="プロテイン", bbox=_b),
                    RecognizedDrink(drink_id=14, name="センブリソーダ", bbox=_b),
                    RecognizedDrink(drink_id=12, name="ブーストエキス", bbox=_b),
                ]
            }
        )
        assert agent.decide("", w8_acts, obs=w8_obs_4).choice_id == 2
        agent._shop_clicked_slots.clear()
        agent._shop_noop_features.clear()
        assert agent.decide("", [w8_acts[0], w8_acts[1], w8_acts[3]], obs=w8_obs).choice_id == 1
    finally:
        agent.close()

    # 4. Swallowed battle play_card tap (identical hand pixels before & after tap) must NOT exhaust limit_count=1 card
    static_frame = np.full((1600, 720, 3), 200, dtype=np.uint8)

    class _StaticAdbCtrl(AdbController):
        def __init__(self):
            super().__init__()
            self._connected = True
            self.clicks = []

        def connect(self):
            return True

        def disconnect(self):
            return None

        def screencap(self):
            return static_frame.copy()

        def click(self, x: int, y: int) -> bool:
            self.clicks.append((x, y))
            return True

    class _StubReco:
        def recognize(self, _frame):
            return ScreenObservation(
                scene_type="battle_turn",
                plan="sense",
                week=7,
                turn=4,
                turns_remaining=6,
                stamina=30,
                max_stamina=44,
                cards=[
                    RecognizedCard(
                        card_id=113,
                        name="始まりの合図+",
                        playable=True,
                        visual_playable=True,
                        bbox=BoundingBox(xmin=150, ymin=1240, xmax=260, ymax=1400),
                    )
                ],
                options=[{"choice_id": 0, "name": "SKIP", "bbox": {"xmin": 580, "ymin": 1460, "xmax": 700, "ymax": 1540}}],
            )

    class _StubAgent:
        def bind_tracker(self, _t):
            pass

        def decide(self, **_kw):
            return DecisionResult(choice_id=0, reason="play #113", source="test")

        def notify_battle_action(self, act):
            notified.append(act)

    notified = []
    monkeypatch.setattr("demo.maa_bridge.runner.time.sleep", lambda *_a, **_kw: None)
    ctrl = _StaticAdbCtrl()
    runner = GakumasRunner(
        mode="adb",
        controller=ctrl,
        agent=_StubAgent(),
        db=db,
        plan="sense",
        scenario_id="hif_honsen",
        p_idol_id=139,
    )
    runner.scene_routing_reco = _StubReco()
    runner.tracker.produce_deck = [113]
    try:
        res = runner.run_once()
        assert res.success
        assert 113 not in runner.tracker.used_limit_cards, "Unconfirmed/swallowed card tap must NOT mark #113 exhausted!"
        assert len(notified) == 0, "Unconfirmed/swallowed card tap must NOT advance battle simulator!"
        assert len(ctrl.clicks) >= 3, f"Expected initial double-tap + animation retry click, got {len(ctrl.clicks)}"
    finally:
        db.close()


def test_reward_antistall_cancels_instead_of_sacrificing_signature_card_and_scanner_no_bounce_dup(monkeypatch):
    """Regression test for live10 W4 Step 97:
    1) On チェンジ/削除 screens, GakumasRunner's reward anti-stall cap (now >=30 steps) must force キャンセル,
       NEVER blindly click チェンジする on a protected signature card (#812 お姉さんの感覚+).
    2) HifLoadoutScanner must not add +520px when bottom-of-list elastic bounce returns dy < 0 with high corr.
    """
    import numpy as np
    from demo.core.db import GakumasDB
    from demo.maa_bridge.mock_controller import MockController
    from demo.maa_bridge.runner import GakumasRunner
    from demo.schemas import BoundingBox, DecisionResult, ScreenObservation

    db = GakumasDB("gakumas_library.db")
    try:
        class _ChangeStubReco:
            def recognize(self, _frame):
                return ScreenObservation(
                    scene_type="card_reward",
                    plan="sense",
                    week=4,
                    raw_text="チェンジするスキルカードを選択してください\nお姉さんの感覚+",
                    options=[
                        {"choice_id": 0, "name": "チェンジする", "card_id": 812, "title": "确认换出: お姉さんの感覚+", "bbox": {"xmin": 400, "ymin": 1440, "xmax": 620, "ymax": 1520}},
                        {"choice_id": 1, "name": "预览第2张待选卡", "slot_idx": 1, "title": "预览第2张待选卡", "bbox": {"xmin": 240, "ymin": 640, "xmax": 350, "ymax": 760}},
                        {"choice_id": 2, "name": "キャンセル", "title": "キャンセル", "bbox": {"xmin": 100, "ymin": 1440, "xmax": 300, "ymax": 1520}},
                    ],
                )

        class _ScanAgent:
            def bind_tracker(self, _t):
                pass

            def decide(self, **_kw):
                return DecisionResult(choice_id=1, reason="keep scanning", source="test")

        runner = GakumasRunner(
            mode="mock",
            controller=MockController(),
            agent=_ScanAgent(),
            db=db,
            plan="sense",
            scenario_id="hif_honsen",
            p_idol_id=139,
        )
        runner.scene_routing_reco = _ChangeStubReco()
        # At step 21 (< 30), anti-stall must NOT trigger yet
        runner._reward_step_count = 20
        r21 = runner.run_once()
        assert r21.action_taken is not None
        assert r21.action_taken.payload.get("name") == "预览第2张待选卡"

        # At step 31 (>= 30), anti-stall MUST choose キャンセル (NEVER チェンジする on #812!)
        runner._reward_step_count = 30
        r31 = runner.run_once()
        assert r31.action_taken is not None
        assert r31.action_taken.payload.get("name") == "キャンセル", f"Expected キャンセル on anti-stall, got {r31.action_taken}"
    finally:
        db.close()


def test_live11_six_fixes_shop_custom_seishintouitsu_combo163_814_memory_free():
    """Regression suite for all 6 live11 fixes:
    1. W6 相談 prioritized over 休む when trouble in deck / >=125P
    2. Customized card (is_customized=True) hard-protected from チェンジ even on same-card swap
    3. W8 インターバル blocks カスタマイズ/強化 while held drinks < 4 and drinks or refresh available
    4. #751 精神統一 (if:goodConditionTurns<=15) penalized on pickup & removable on 好調 idol (#139) + 授業 prefers 好調 over 集中
    5. Mid-battle #22 移動 modal does NOT reset battle planner + #163->#814 forced combo on rem<=2
    6. Configured free_extra_use_first_play memory hook card is prioritized over non-free cards
    """
    from demo.core.tracker import GameStateTracker
    db = GakumasDB("gakumas_library.db")
    _b = BoundingBox(xmin=0, ymin=0, xmax=100, ymax=100)
    try:
        tracker = GameStateTracker(p_idol_id=139, plan="sense", scenario_id="hif_honsen")
        tracker.produce_deck = [812, 814, 163, 295, 75, 23]
        tracker.card_memory_hooks = {
            75: {"free_extra_use_first_play": True, "extra_use_limit": 2, "draw_on_play": 0},
        }
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=8, ismcts_simulations=4),
        )
        agent.tracker = tracker
        try:
            # 1. W6 schedule chooses 相談 over 休む when trouble is in deck (HP=18/44)
            w6_obs = ScreenObservation(
                scene_type="produce_schedule",
                plan="sense",
                scenario_id="hif_honsen",
                week=6,
                stamina=18,
                max_stamina=44,
                p_points=380,
            )
            w6_acts = [
                LegalAction(choice_id=0, action_type="pick_option", description="相談", bbox=_b, payload={"name": "相談"}),
                LegalAction(choice_id=1, action_type="pick_option", description="おでかけ", bbox=_b, payload={"name": "おでかけ"}),
                LegalAction(choice_id=2, action_type="pick_option", description="休む", bbox=_b, payload={"name": "休む"}),
            ]
            assert agent.decide("", w6_acts, obs=w6_obs).choice_id == 0

            # 2. Customized #295 スポットライト+ MUST NOT be swapped even on same-card チェンジ
            agent._came_from_change_pickup = True
            agent._pending_change_gain_card_id = 294
            agent._pending_change_gain_base_id = 294
            s295_cust = agent._score_skill_card(
                card_id=295,
                title="スポットライト+",
                desc="好調9ターン 【已加点强化(カスタマイズ済)】",
                target_effect="goodConditionTurns",
                guide=None,
                plan="sense",
                for_removal=True,
                p_idol_id=139,
                has_removable_unwanted=False,
                check_same_card=True,
                ignore_upgrade=True,
                is_customized=True,
            )
            assert s295_cust <= -15_000, f"Customized card must be hard-protected from swap, got {s295_cust}"
            agent._came_from_change_pickup = False
            agent._pending_change_gain_card_id = None
            agent._pending_change_gain_base_id = None

            # 3. W8 インターバル blocks カスタマイズ & refreshes for drinks when <4 drinks & no drink on shelf
            w8_no_drink_obs = ScreenObservation(
                scene_type="shop",
                plan="sense",
                scenario_id="hif_honsen",
                week=8,
                p_points=300,
                drinks=[RecognizedDrink(drink_id=22, name="初星黒酢", bbox=_b)],
            )
            w8_acts_ref = [
                LegalAction(choice_id=0, action_type="pick_option", description="カスタマイズ", bbox=_b, payload={"name": "カスタマイズ", "shop_slot_key": "カスタマイズ"}),
                LegalAction(choice_id=1, action_type="pick_option", description="リフレッシュ (10P)", bbox=_b, payload={"name": "リフレッシュ", "shop_slot_key": "リフレッシュ", "price": 10}),
                LegalAction(choice_id=2, action_type="pick_option", description="強化 (100P)", bbox=_b, payload={"name": "強化", "shop_slot_key": "強化", "price": 100}),
                LegalAction(choice_id=3, action_type="pick_option", description="終了", bbox=_b, payload={"name": "終了"}),
            ]
            assert agent.decide("", w8_acts_ref, obs=w8_no_drink_obs).choice_id == 1

            # 4. #751 精神統一 is penalized on pickup and removable on 好調 idol (#139); 授業 event prefers 好調 over 集中
            s751_pick = agent._score_skill_card(
                card_id=751,
                title="精神統一",
                desc="集中+5 / 好調が15ターン以下の場合、集中強化20%",
                target_effect="goodConditionTurns",
                guide=None,
                plan="sense",
                for_removal=False,
                p_idol_id=139,
            )
            s751_rem = agent._score_skill_card(
                card_id=751,
                title="精神統一",
                desc="集中+5 / 好調が15ターン以下の場合、集中強化20%",
                target_effect="goodConditionTurns",
                guide=None,
                plan="sense",
                for_removal=True,
                p_idol_id=139,
            )
            assert s751_pick < 0, f"#751 精神統一 should be penalized on 好調 idol, got {s751_pick}"
            assert s751_rem > 0, f"#751 精神統一 should be removable on 好調 idol, got {s751_rem}"

            ev_obs = ScreenObservation(scene_type="produce_event", plan="sense", week=1, week_activity="授業")
            ev_acts = [
                LegalAction(choice_id=0, action_type="pick_option", description="集中効果のスキルカードを獲得", bbox=_b, payload={"name": "集中"}),
                LegalAction(choice_id=1, action_type="pick_option", description="好調効果のスキルカードを獲得", bbox=_b, payload={"name": "好調"}),
            ]
            assert agent.decide("", ev_acts, obs=ev_obs, p_idol_id=139).choice_id == 1

            # 5. Mid-battle #22 移動 modal must NOT reset battle_planner, and #163->#814 fires on rem=2 even with #776 in hand
            b_obs_rem2 = ScreenObservation(
                scene_type="battle_turn",
                plan="sense",
                scenario_id="hif_honsen",
                week=9,
                turn=11,
                turns_remaining=2,
                stamina=41,
                max_stamina=44,
                score=104198,
                buffs={"goodConditionTurns": 82, "concentration": 24, "perfectConditionTurns": 10},
                cards=[
                    RecognizedCard(card_id=776, name="話題沸騰+", playable=True, visual_playable=True, bbox=_b),
                    RecognizedCard(card_id=814, name="自然体の魅力+", playable=True, visual_playable=True, bbox=_b),
                    RecognizedCard(card_id=163, name="国民的アイドル+", playable=True, visual_playable=True, bbox=_b),
                ],
                drinks=[],
            )
            b_acts_rem2 = [
                LegalAction(choice_id=0, action_type="play_card", description="打出卡牌: 話題沸騰+", bbox=_b, payload={"card_id": 776, "name": "話題沸騰+"}),
                LegalAction(choice_id=1, action_type="play_card", description="打出卡牌: 自然体の魅力+", bbox=_b, payload={"card_id": 814, "name": "自然体の魅力+"}),
                LegalAction(choice_id=2, action_type="play_card", description="打出卡牌: 国民的アイドル+ [⚡免费+1使用次数(不耗行动点)]", bbox=_b, payload={"card_id": 163, "name": "国民的アイドル+"}),
                LegalAction(choice_id=3, action_type="end_turn", description="SKIP", bbox=_b, payload={}),
            ]
            d_combo = agent.decide("", b_acts_rem2, obs=b_obs_rem2, p_idol_id=139)
            assert d_combo.choice_id == 2, f"Expected #163 before #814 at rem=2, got {d_combo}"
            assert agent.battle_planner._sim is not None
            # Mid-battle #22 移動 modal must preserve active battle_planner._sim!
            move_obs = ScreenObservation(scene_type="card_reward", plan="sense", week=9, raw_text="移動 スキルカードが選択されていません")
            move_acts = [
                LegalAction(choice_id=0, action_type="pick_option", description="点击预览并选中该卡库技能卡加入手牌", bbox=_b, payload={"name": "待选卡1", "slot_idx": 0}),
            ]
            agent.decide("", move_acts, obs=move_obs, p_idol_id=139)
            assert agent.battle_planner._sim is not None, "Mid-battle #22 移動 modal must NOT reset battle_planner!"

            # 6. Configured memory free_extra_use_first_play (#75 アドリブ+) must be played over non-free cards
            b_obs_mem = ScreenObservation(
                scene_type="battle_turn",
                plan="sense",
                scenario_id="hif_honsen",
                week=9,
                turn=5,
                turns_remaining=6,
                stamina=40,
                max_stamina=44,
                score=10000,
                buffs={"goodConditionTurns": 20, "concentration": 10},
                cards=[
                    RecognizedCard(card_id=75, name="アドリブ+", playable=True, visual_playable=True, bbox=_b),
                    RecognizedCard(card_id=295, name="スポットライト+", playable=True, visual_playable=True, bbox=_b),
                ],
                drinks=[],
            )
            b_acts_mem = [
                LegalAction(choice_id=0, action_type="play_card", description="打出卡牌: アドリブ+", bbox=_b, payload={"card_id": 75, "name": "アドリブ+"}),
                LegalAction(choice_id=1, action_type="play_card", description="打出卡牌: スポットライト+", bbox=_b, payload={"card_id": 295, "name": "スポットライト+"}),
                LegalAction(choice_id=2, action_type="end_turn", description="SKIP", bbox=_b, payload={}),
            ]
            d_mem = agent.decide("", b_acts_mem, obs=b_obs_mem, p_idol_id=139)
            assert d_mem.choice_id == 0, f"Expected memory free-action #75 アドリブ+ first, got {d_mem}"
        finally:
            agent.close()
    finally:
        db.close()


def test_shop_buys_unowned_priority_card_above_drinks_and_skips_if_owned():
    """User `priority_card_ids` on shop shelf must be bought ABOVE drinks unless deck already owns a copy."""
    from demo.core.hif_catalog import ProducePreferences
    from demo.core.tracker import GameStateTracker

    db = GakumasDB("gakumas_library.db")
    _b = BoundingBox(xmin=0, ymin=0, xmax=100, ymax=100)
    try:
        tracker = GameStateTracker(p_idol_id=139, plan="sense", scenario_id="hif_honsen")
        tracker.produce_deck = [812, 814, 163, 295]
        prefs = ProducePreferences(priority_card_ids=[105, 75])  # #105 大声援+ -> base #104
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=8, ismcts_simulations=4),
            preferences=prefs,
        )
        agent.tracker = tracker
        try:
            # 1. W8 shop with 0/4 drinks AND #104 大声援 (priority, unowned) + #80 祝福 (non-priority) + #22 初星黒酢
            #    -> MUST select #104 大声援 BEFORE #22 初星黒酢!
            w8_obs = ScreenObservation(
                scene_type="shop",
                plan="sense",
                scenario_id="hif_honsen",
                week=8,
                p_points=580,
                drinks=[],
            )
            shop_acts = [
                LegalAction(choice_id=0, action_type="pick_option", description="スキルカード2 《祝福》 50P", bbox=_b, payload={"name": "スキルカード2", "shop_slot_key": "スキルカード:1", "card_id": 80, "price": 50}),
                LegalAction(choice_id=1, action_type="pick_option", description="スキルカード3 《大声援》 50P", bbox=_b, payload={"name": "スキルカード3", "shop_slot_key": "スキルカード:2", "card_id": 104, "price": 50}),
                LegalAction(choice_id=2, action_type="pick_option", description="Pドリンク3 《初星黒酢》 100P", bbox=_b, payload={"name": "Pドリンク3", "shop_slot_key": "Pドリンク:2", "drink_id": 22, "price": 100}),
                LegalAction(choice_id=3, action_type="pick_option", description="カスタマイズ", bbox=_b, payload={"name": "カスタマイズ", "shop_slot_key": "カスタマイズ"}),
                LegalAction(choice_id=4, action_type="pick_option", description="終了", bbox=_b, payload={"name": "終了"}),
            ]
            d1 = agent.decide("", shop_acts, obs=w8_obs)
            assert d1.choice_id == 1, f"Expected priority card #104 above drink #22, got {d1}"

            # 2. Active `交換する` for #104 -> MUST confirm exchange immediately!
            ex_acts = [
                LegalAction(choice_id=0, action_type="pick_option", description="交換する 《大声援》 (-50P)", bbox=_b, payload={"name": "交換する", "card_id": 104, "price": 50}),
                LegalAction(choice_id=1, action_type="pick_option", description="Pドリンク3 《初星黒酢》 100P", bbox=_b, payload={"name": "Pドリンク3", "shop_slot_key": "Pドリンク:2", "drink_id": 22, "price": 100}),
                LegalAction(choice_id=2, action_type="pick_option", description="終了", bbox=_b, payload={"name": "終了"}),
            ]
            d2 = agent.decide("", ex_acts, obs=w8_obs)
            assert d2.choice_id == 0, f"Expected 交換する for #104, got {d2}"

            # 3. Purchase confirmation popup -> MUST click 交換 (NOT キャンセル!) and record #104 in deck
            pop_obs = ScreenObservation(scene_type="popup", plan="sense", week=8, raw_text="交換確認 50Pを消費して交換しますか？")
            pop_acts = [
                LegalAction(choice_id=0, action_type="pick_option", description="キャンセル", bbox=_b, payload={"name": "キャンセル"}),
                LegalAction(choice_id=1, action_type="pick_option", description="交換", bbox=_b, payload={"name": "交換"}),
            ]
            d3 = agent.decide("", pop_acts, obs=pop_obs)
            assert d3.choice_id == 1, f"Expected popup 交換 confirmation for priority card, got {d3}"
            assert 104 in tracker.produce_deck

            # 4. Once #104 is already in produce_deck, seeing #104/#105 again in shop MUST skip it and buy drink #22!
            agent._shop_clicked_slots.clear()
            agent._shop_attempted_card_ids.clear()
            d4 = agent.decide("", shop_acts, obs=w8_obs.model_copy(update={"p_points": 530}))
            assert d4.choice_id == 2, f"Once #104 is owned, must skip it and buy drink #22, got {d4}"
        finally:
            agent.close()
    finally:
        db.close()


def test_change_pickup_card_without_limit_and_ocr_kutasai_stays_card_reward():
    """Live regression: チェンジで獲得 page 1 with non-exhaust card (アドリブ) + '選んでくたさい' OCR typo + orange 次へ must stay card_reward."""
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    rec = LiveScreenRecognizer(db=db, plan="sense", scenario_id="hif_honsen")
    try:
        frame = np.full((1600, 720, 3), 245, dtype=np.uint8)
        # Orange "次へ" button at y=1220..1290
        frame[1220:1290, 237:482] = (245, 130, 30)
        lines = [
            {"text": "審査基準", "bbox": [551, 45, 629, 63]},
            {"text": "チェンジで獲得するスキルカードを選んでくたさい", "bbox": [87, 457, 634, 480]},
            {"text": "アドリプ", "bbox": [304, 679, 418, 706]},
            {"text": "十バラメータ+5", "bbox": [138, 736, 313, 757]},
            {"text": "+イ3好調3ターン", "bbox": [138, 766, 343, 801]},
            {"text": "第を気", "bbox": [160, 976, 576, 1136]},
            {"text": "今次へ", "bbox": [256, 1238, 410, 1273]},
        ]
        pv_lines = [
            {"text": "アドリプ", "bbox": [250, 80, 365, 110]},
            {"text": "十バラメータ+5", "bbox": [80, 140, 260, 165]},
        ]
        btn_lines = [{"text": "→次へ", "bbox": [20, 15, 180, 55]}]

        def _fake_ocr(arr, *a, **kw):
            h_a = arr.shape[0]
            if h_a == 1600:
                return {"text": "", "lines": lines}
            if 100 <= h_a <= 200:
                return {"text": "→次へ", "lines": btn_lines}
            return {"text": "", "lines": pv_lines}

        rec.ocr = SimpleNamespace(recognize_array=_fake_ocr)
        rec.en_ocr = SimpleNamespace(recognize_array=lambda _arr, *a, **kw: {"text": "", "lines": []})

        obs = rec.recognize(frame)
        assert obs.scene_type == "card_reward"
        names = [o.get("name") for o in obs.options]
        assert "受け取る" in names
        claim = next(o for o in obs.options if o.get("name") == "受け取る")
        assert claim.get("card_id") == 74
    finally:
        db.close()


def test_drink22_selected_pill_full_scan_combo163_w6_soudan_and_w8_refresh_fixes(monkeypatch):
    """Regressions from Run 7 live audit:
    1) #22 move-to-hand modal with top-left ◆移動 pill + orange bottom button stays card_reward (never skip/drink).
    2) #22 scans past slot 6 when #814 is in hand and pulls #163 (22_000 > duplicate #814 15_500); planner tutors #163 before solo #814.
    3) W6 H.I.F schedule detects 相談 via focused bottom-center 3x pass when shallow 0.72..0.91 strip misses it, and ignores <0.915h boundary footer text.
    4) Refresh confirmation popup clears _shop_visited_slots despite dimmed shop 交換/強化 backdrop; GakumasRunner clears _live_recognizer slots; _decide_shop caps consecutive fruitless drink refreshes.
    """
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    rec = LiveScreenRecognizer(db=db, plan="sense", scenario_id="hif_honsen")
    try:
        # 1) #22 modal after clicking slot 0: "スキルカードが選択されていません" disappeared, full-frame OCR missed bottom button,
        #    only top-left ◆移動 pill + orange button remain -> must stay card_reward ('移動する'), never skip or drink!
        tutor_frame = np.full((1600, 720, 3), 35, dtype=np.uint8)
        for sx in (138, 282, 426):
            tutor_frame[835:955, sx - 58 : sx + 58] = (230, 230, 230)
        tutor_frame[832:958, 76:200] = (245, 145, 30)  # selected slot border
        tutor_frame[1444:1514, 238:481] = (245, 130, 30)  # active orange 移動 button
        tutor_lines = [
            {"text": "◆移動", "bbox": [20, 39, 118, 71]},
        ]

        def _tutor_ocr(arr, *a, **kw):
            h_a, w_a = arr.shape[:2]
            if (h_a, w_a) == (1600, 720):
                return {"text": "", "lines": tutor_lines}
            if h_a < 260 and w_a > 400 and kw.get("scale", None) is None:
                # bottom orange button crop from _ocr_bottom_orange_button
                return {"text": "ノ移動", "lines": [{"text": "ノ移動", "bbox": [40, 10, 160, 45]}]}
            return {"text": "破竹の勢い", "lines": [{"text": "破竹の勢い", "bbox": [20, 10, 200, 45]}]}

        rec.ocr = SimpleNamespace(recognize_array=_tutor_ocr)
        rec.en_ocr = SimpleNamespace(recognize_array=lambda *_a, **_kw: {"text": "", "lines": []})
        rec._last_turns_remaining = 2
        obs_tutor = rec.recognize(tutor_frame, plan="sense")
        assert obs_tutor.scene_type == "card_reward"
        assert obs_tutor.options[0]["name"] == "移動する"
        assert not any(o.get("drink_id") is not None for o in obs_tutor.options)

        # 2) Agent scans past slot 6 (#22 grid with 10 visible slots) and grabs #163 at slot 8 when #814 is in hand
        agent = LocalHybridDecisionAgent(db=db)
        agent.reset_session()
        planner = agent.battle_planner
        obs_b_turn = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=7,
            turn=10,
            turns_remaining=2,
            stamina=30,
            max_stamina=44,
            genki=20,
            buffs={"goodConditionTurns": 18, "concentration": 10},
            cards=[
                RecognizedCard(card_id=814, name="自然体の魅力+", type="active", playable=True, visual_playable=True, bbox=BoundingBox(xmin=10, ymin=1000, xmax=200, ymax=1300)),
            ],
            drinks=[
                RecognizedDrink(drink_id=22, name="初星黒酢", playable=True, bbox=BoundingBox(xmin=20, ymin=860, xmax=80, ymax=920)),
            ],
        )
        tr_t = GameStateTracker(plan="sense", p_idol_id=139, scenario_id="hif_honsen")
        tr_t.set_initial_loadout(
            p_idol_id=139,
            plan="sense",
            recommended_effect="goodConditionTurns",
            deck_ids=[814, 163, 74, 75, 105, 113, 812, 35],
            p_items=[{"item_id": 436}],
            db=db,
        )
        pb_t = PromptBuilder(db=db, compact=False)
        _, acts_t = pb_t.build_prompt(obs_b_turn, tr_t)
        planner.decide_battle_action(obs_b_turn, acts_t, tracker=tr_t)
        a814_t = next(a for a in acts_t if (a.payload or {}).get("card_id") == 814)
        aid_814_t = next(aid for aid, act in planner._map_legal_actions_to_canonical(acts_t).items() if act.choice_id == a814_t.choice_id)
        planner._ismcts.select_action = lambda *_a, **_kw: (aid_814_t, {"best_q_score": 500_000.0, "top_actions": []})  # type: ignore[union-attr,method-assign]
        cid_combo22, reason_combo22, _ = planner.decide_battle_action(obs_b_turn, acts_t, tracker=tr_t)
        assert next(a for a in acts_t if a.choice_id == cid_combo22).action_type == "use_drink"
        assert "#22->Tutor #163" in reason_combo22

        # Now #22 modal opens with #814 in sim.hand: slot 0..7 are filler, slot 8 is #163 -> must inspect all the way to slot 8!
        for slot_i in range(8):
            obs_loop = ScreenObservation(
                scene_type="card_reward",
                plan="sense",
                scenario_id="hif_honsen",
                week=7,
                turns_remaining=2,
                options=[
                    {"choice_id": 0, "name": "移動する", "title": "移動する《破竹の勢い》", "card_id": 98, "slot_idx": slot_i, "bbox": {"xmin": 240, "ymin": 1440, "xmax": 480, "ymax": 1515}},
                    *[
                        {"choice_id": j + 1, "name": f"待选卡{j + 1}", "title": f"待选卡{j + 1}", "slot_idx": j, "bbox": {"xmin": 80 + (j % 4) * 140, "ymin": 830 + (j // 4) * 140, "xmax": 190 + (j % 4) * 140, "ymax": 950 + (j // 4) * 140}}
                        for j in range(9)
                    ],
                ],
            )
            acts_loop = [
                LegalAction(choice_id=o["choice_id"], action_type="pick_option", description=o["title"], bbox=BoundingBox(**o["bbox"]), payload=o)
                for o in obs_loop.options
            ]
            d_scan = agent.decide("", acts_loop, obs=obs_loop, p_idol_id=139)
            assert acts_loop[d_scan.choice_id].payload.get("name") != "移動する"
            assert acts_loop[d_scan.choice_id].payload.get("slot_idx") == slot_i + 1
        obs_slot8 = ScreenObservation(
            scene_type="card_reward",
            plan="sense",
            scenario_id="hif_honsen",
            week=7,
            turns_remaining=2,
            options=[
                {"choice_id": 0, "name": "移動する", "title": "移動する《国民的アイドル+》", "card_id": 163, "slot_idx": 8, "bbox": {"xmin": 240, "ymin": 1440, "xmax": 480, "ymax": 1515}},
                *[
                    {"choice_id": j + 1, "name": f"待选卡{j + 1}", "title": f"待选卡{j + 1}", "slot_idx": j, "bbox": {"xmin": 80 + (j % 4) * 140, "ymin": 830 + (j // 4) * 140, "xmax": 190 + (j % 4) * 140, "ymax": 950 + (j // 4) * 140}}
                    for j in range(9)
                ],
            ],
        )
        acts_slot8 = [
            LegalAction(choice_id=o["choice_id"], action_type="pick_option", description=o["title"], bbox=BoundingBox(**o["bbox"]), payload=o)
            for o in obs_slot8.options
        ]
        d_slot8 = agent.decide("", acts_slot8, obs=obs_slot8, p_idol_id=139)
        assert d_slot8.choice_id == 0

        # 3) W6 相談 bottom-center 3x focused pass catches 相談 even when full-frame and 0.72..0.945 strip OCR miss it
        w6_frame = np.full((1600, 720, 3), 235, dtype=np.uint8)
        w6_frame[1385:1415, 330:390] = (30, 30, 30)
        w6_full_lines = [
            {"text": "6日目", "bbox": [45, 25, 115, 55]},
            {"text": "行動を選択してください", "bbox": [210, 1080, 510, 1110]},
            {"text": "休む", "bbox": [610, 1197, 665, 1218]},
        ]

        def _w6_ocr(arr, *a, **kw):
            h_a, w_a = arr.shape[:2]
            if (h_a, w_a) == (1600, 720):
                return {"text": "", "lines": w6_full_lines}
            # 3x upscaled tight crop on bottom-center 相談 (patch height ~ 107 -> *3 + 36 = ~357, width ~ 230 -> *3 + 36 = ~726)
            if h_a >= 300 and 700 <= w_a <= 820 and h_a < 500:
                return {"text": "相談", "lines": [{"text": "相談", "bbox": [280, 120, 420, 185]}]}
            return {"text": "", "lines": []}

        rec.ocr = SimpleNamespace(recognize_array=_w6_ocr)
        obs_w6 = rec.recognize(w6_frame, plan="sense")
        assert obs_w6.scene_type == "produce_schedule"
        assert "相談" in [o["name"] for o in obs_w6.options]

        # 4) Refresh popup clears _shop_visited_slots even with 交換/強化/カスタマイズ behind; 交換 popup does NOT;
        #    GakumasRunner uses _live_recognizer; _decide_shop yields refresh to カスタマイズ after 2 fruitless refreshes.
        rec._shop_visited_slots = {"Pドリンク:2", "Pドリンク:3"}
        rec._shop_last_clicked_slot = "Pドリンク:2"
        pop_frame = np.full((1600, 720, 3), 60, dtype=np.uint8)
        pop_exchange_lines = [
            {"text": "交換確認", "bbox": [280, 400, 440, 435]},
            {"text": "このドリンクを交換しますか", "bbox": [180, 520, 540, 555]},
            {"text": "リフレッシュ", "bbox": [40, 1210, 180, 1240]},
            {"text": "交換", "bbox": [430, 1120, 540, 1165]},
            {"text": "キャンセル", "bbox": [160, 1120, 300, 1165]},
        ]
        rec.ocr = SimpleNamespace(recognize_array=lambda *_a, **_kw: {"text": "", "lines": pop_exchange_lines})
        obs_pop_ex = rec.recognize(pop_frame, plan="sense")
        assert obs_pop_ex.scene_type == "popup"
        assert rec._shop_visited_slots == {"Pドリンク:2", "Pドリンク:3"}

        pop_refresh_lines = [
            {"text": "商品更新確認", "bbox": [260, 400, 460, 435]},
            {"text": "ラインナップを更新しますか", "bbox": [170, 520, 550, 555]},
            {"text": "Pポイントと交換 強化 カスタマイズ リフレッシュ", "bbox": [40, 1210, 580, 1240]},
            {"text": "更新する", "bbox": [420, 1120, 560, 1165]},
            {"text": "キャンセル", "bbox": [160, 1120, 300, 1165]},
        ]
        rec.ocr = SimpleNamespace(recognize_array=lambda *_a, **_kw: {"text": "", "lines": pop_refresh_lines})
        obs_pop_ref = rec.recognize(pop_frame, plan="sense")
        assert obs_pop_ref.scene_type == "popup"
        assert "更新する" in [o["name"] for o in obs_pop_ref.options]
        assert len(rec._shop_visited_slots) == 0
        assert rec._shop_last_clicked_slot is None

        # Runner clears _live_recognizer._shop_visited_slots (not just .recognizer!)
        ctrl_r = SimpleNamespace(
            is_connected=True,
            connect=lambda: True,
            disconnect=lambda: None,
            screencap=lambda: pop_frame,
            click=lambda _x, _y: True,
            swipe=lambda *a, **kw: None,
        )
        runner_lr = GakumasRunner(
            mode="adb",
            controller=ctrl_r,  # type: ignore[arg-type]
            agent=agent,
            db=db,
            plan="sense",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        agent.reset_session()
        runner_lr.tracker.p_points = 250
        rec._shop_visited_slots = {"Pドリンク:2", "Pドリンク:3"}
        rec._last_p_points = 250
        obs_ref_btn = ScreenObservation(
            scene_type="shop",
            plan="sense",
            scenario_id="hif_honsen",
            week=8,
            total_weeks=9,
            p_points=250,
            options=[
                {
                    "choice_id": 0,
                    "name": "リフレッシュ",
                    "title": "リフレッシュ (10P)",
                    "shop_slot_key": "リフレッシュ",
                    "price": 10,
                    "bbox": {"xmin": 24, "ymin": 1204, "xmax": 207, "ymax": 1304},
                }
            ],
        )
        runner_lr.scene_routing_reco = SimpleNamespace(_live_recognizer=rec, recognize=lambda _fr, **_kw: obs_ref_btn)
        runner_lr.run_once()
        assert len(rec._shop_visited_slots) == 0
        assert runner_lr.tracker.p_points == 240

        # Consecutive fruitless refresh cap in _decide_shop
        agent.reset_session()
        agent.bind_tracker(tr_t)
        agent._shop_refresh_since_purchase = 2
        obs_w8_cap = ScreenObservation(
            scene_type="shop",
            plan="sense",
            scenario_id="hif_honsen",
            week=8,
            total_weeks=9,
            p_points=190,
            drinks=[RecognizedDrink(drink_id=20, name="初星スープ", bbox=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10))],
        )
        acts_w8_cap = [
            LegalAction(choice_id=0, action_type="pick_option", description="リフレッシュ (30P)", bbox=BoundingBox(xmin=24, ymin=1204, xmax=207, ymax=1304), payload={"name": "リフレッシュ", "shop_slot_key": "リフレッシュ", "price": 30}),
            LegalAction(choice_id=1, action_type="pick_option", description="カスタマイズ", bbox=BoundingBox(xmin=300, ymin=1050, xmax=500, ymax=1120), payload={"name": "カスタマイズ", "shop_slot_key": "カスタマイズ"}),
            LegalAction(choice_id=2, action_type="pick_option", description="終了", bbox=BoundingBox(xmin=580, ymin=1220, xmax=690, ymax=1270), payload={"name": "終了", "shop_slot_key": "終了"}),
        ]
        d_cap = agent.decide("", acts_w8_cap, obs=obs_w8_cap)
        assert d_cap.choice_id == 1
    finally:
        agent.close()
        db.close()


def test_p_drink_reward_move_to_hand_description_not_hijacked_as_tutor_grid():
    """Produce P-drink reward (e.g. パワフル漢方ドリンク) whose description contains
    '山札か捨札にあるアイドル固有スキルカード1枚を手札に移動' and 'スキルカード'
    must remain a 3-slot drink reward with 受け取る active, never a deck-grid #22 scan.
    """
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    rec = LiveScreenRecognizer(db=db, plan="sense", scenario_id="hif_honsen")
    try:
        frame = np.full((1600, 720, 3), 85, dtype=np.uint8)
        frame[630:1290, 52:668] = (248, 248, 248)  # white reward dialog
        frame[1212:1284, 232:488] = (245, 130, 30)  # orange 受け取る button
        for idx, sx in enumerate((215, 360, 505)):
            frame[975:1090, sx - 58 : sx + 58] = (225, 232, 240)
        # Selected middle drink slot (オレンジ枠)
        frame[965:1100, 290:430] = (245, 145, 30)

        full_lines = [
            {"text": "2日目", "bbox": [45, 25, 115, 55]},
            {"text": "Pドリンクを選択して獲得", "bbox": [180, 485, 500, 512]},
            {"text": "パワフル漢方ドリンク", "bbox": [210, 665, 510, 700]},
            {"text": "山札か捨札にあるアイドル固有スキルカード", "bbox": [110, 725, 610, 755]},
            {"text": "1枚を手札に移動", "bbox": [110, 760, 310, 788]},
            {"text": "消費体力減少3ターン", "bbox": [155, 800, 420, 830]},
            {"text": "受け取る", "bbox": [310, 1230, 410, 1265]},
        ]
        pv_lines = [
            {"text": "パワフル漢方ドリンク", "bbox": [160, 40, 440, 75]},
            {"text": "山札か捨札にあるアイドル固有スキルカード1枚を手札に移動", "bbox": [40, 95, 560, 145]},
            {"text": "消費体力減少3ターン", "bbox": [60, 155, 320, 185]},
        ]

        def _fake_ocr(arr, *a, **kw):
            if arr.shape[:2] == (1600, 720):
                return {"text": "", "lines": full_lines}
            return {"text": "", "lines": pv_lines}

        rec.ocr = SimpleNamespace(recognize_array=_fake_ocr)
        rec.en_ocr = SimpleNamespace(recognize_array=lambda *_a, **_kw: {"text": "", "lines": []})
        rec._in_battle_scene = False
        obs = rec.recognize(frame, plan="sense")
        assert obs.scene_type == "card_reward"
        names = [o.get("name") for o in obs.options]
        assert "移動する" not in names
        assert not any("待选卡" in str(n) for n in names)
        assert "受け取る" in names
        claim = next(o for o in obs.options if o.get("name") == "受け取る")
        assert claim.get("drink_id") is not None
        drink_row = db.get_drink_by_id(int(claim["drink_id"]))
        assert drink_row is not None and "パワフル漢方ドリンク" in drink_row["name"]
    finally:
        db.close()


def test_p_item_9139_counter_sync_same_name_customized_priority_spike_guard_and_w8_455_p_points(monkeypatch):
    """Regressions from Run 8 W6/W8/W9 audit:
    1) Stage P-item #9139 only counts goodConditionTurns+= cards (NOT #776 話題沸騰+ or #147).
    2) W9 S228: live p_item_remaining_counts={0: 2} syncs #9139 counter, duplicate #295 does not hijack
       unmapped fallback, and planner plays #776 話題沸騰+ at rem=1 instead of #295 M-card.
    3) Same-name customized priority: hand with [#295 (uncustomized), #295 (customized=True, count=2)]
       always selects the customized copy (even if MCTS scores another duplicate equal or higher).
    4) Upward buff OCR spike (goodConditionTurns: 45 -> 405) is rejected and inflated sim.score resyncs to real obs.score.
    5) W8 455->55 P-point truncation is recovered by wider OCR crop, and W6 休み確認 header is rejected/cancelled.
    """
    from demo.sim.battle_simulator import BattleConfig, HeadlessBattleSimulator, TurnMultiplierSpec, _build_sense_hif_p_items, _make_card_config
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        # 1) #9139 must only count goodConditionTurns+= cards
        cfg_9139 = BattleConfig(
            scenario_id="hif_honsen",
            stage_name="sense_9139_strict",
            p_idol_id=139,
            idol_name="姫崎 莉波",
            plan="sense",
            idol_stats={"vocal": 1360, "dance": 2647, "visual": 2087},
            initial_stamina=40,
            max_stamina=44,
            initial_good_condition=20,
            initial_perfect_condition=3,
            total_turns=4,
            turn_schedule=[TurnMultiplierSpec("Vi", 1.0)] * 4,
            deck=[_make_card_config(db, 105)] * 6,
            p_items=[p for p in _build_sense_hif_p_items() if p.item_id == 9139],
        )
        sim_strict = HeadlessBattleSimulator(cfg_9139, db=db, seed=7)
        sim_strict.card_uses_remaining = 4
        sim_strict.hand = [
            sim_strict._new_instance(_make_card_config(db, 776)),
            sim_strict._new_instance(_make_card_config(db, 147)),
            sim_strict._new_instance(_make_card_config(db, 295)),
        ]
        sim_strict.play_card(0)  # #776 (goodConditionTurnsMultiplier=2.5, NOT +=)
        assert sim_strict.p_item_sense_gc_counters.get(9139, 0) == 0
        sim_strict.play_card(0)  # #147 (if-condition/multiplier only, NOT +=)
        assert sim_strict.p_item_sense_gc_counters.get(9139, 0) == 0
        sim_strict.play_card(0)  # #295 (goodConditionTurns+=6)
        assert sim_strict.p_item_sense_gc_counters.get(9139, 0) == 1

        # 2) W9 S228 + 4) Upward buff OCR spike guard + sim.score resync
        tr = GameStateTracker(plan="sense", p_idol_id=139, scenario_id="hif_honsen")
        tr.set_initial_loadout(
            p_idol_id=139,
            plan="sense",
            recommended_effect="goodConditionTurns",
            deck_ids=[812, 814, 163, 776, 295, 295, 87, 117, 105, 112],
            p_items=[{"item_id": 9139}],
            db=db,
        )
        planner = LiveBattlePlanner(db=db, cfg=LiveBattlePlannerConfig(ppo_episodes=16, ismcts_simulations=32))
        pb = PromptBuilder(db=db, compact=False)
        box = BoundingBox(xmin=10, ymin=1000, xmax=160, ymax=1300)

        obs_t5 = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=9,
            turn=5,
            turns_remaining=8,
            stamina=36,
            max_stamina=44,
            score=12_820,
            buffs={"goodConditionTurns": 45, "concentration": 15, "perfectConditionTurns": 5},
            p_items=[{"item_id": "9139"}],
            p_item_remaining_counts={0: 2},
            cards=[RecognizedCard(card_id=295, name="スポットライト+", type="mental", upgraded=True, playable=True, visual_playable=True, bbox=box)],
        )
        _, acts_t5 = pb.build_prompt(obs_t5, tr)
        planner.decide_battle_action(obs_t5, acts_t5, tracker=tr)
        assert planner._sim is not None
        assert planner._sim.p_item_sense_gc_counters.get(9139, 0) % 4 == 2  # 4 - 2 == 2

        # Inject inflated sim.score and an OCR goodConditionTurns spike (45 -> 405)
        planner._sim.score = 3_611_524
        obs_s228 = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=9,
            turn=12,
            turns_remaining=1,
            stamina=44,
            max_stamina=44,
            score=819_643,
            card_uses_remaining=1,
            idol_stats={"vocal": 1360, "dance": 2647, "visual": 2087},
            buffs={"goodConditionTurns": 405, "concentration": 21, "perfectConditionTurns": 12},
            p_items=[{"item_id": "9139"}],
            p_item_remaining_counts={0: 2},  # 2 cards away from trigger -> #295 will NOT grant +1 play!
            cards=[
                RecognizedCard(card_id=776, name="話題沸騰+", type="active", upgraded=True, playable=True, visual_playable=True, bbox=box),
                RecognizedCard(card_id=295, name="スポットライト+", type="mental", upgraded=True, playable=True, visual_playable=True, customized=True, customization_count=2, bbox=box),
                RecognizedCard(card_id=295, name="スポットライト+", type="mental", upgraded=True, playable=True, visual_playable=True, customized=False, customization_count=0, bbox=box),
            ],
        )
        _, acts_s228 = pb.build_prompt(obs_s228, tr)
        assert "专精强化×2" in acts_s228[1].description
        assert acts_s228[1].payload.get("customization_count") == 2
        cid_s228, reason_s228, _ = planner.decide_battle_action(obs_s228, acts_s228, tracker=tr)
        assert int(planner._sim.buffs["goodConditionTurns"]) < 200
        assert planner._sim.score == 819_643
        assert int((acts_s228[cid_s228].payload or {}).get("card_id") or 0) == 776, f"Expected #776 over #295 at rem=1, got {reason_s228}"

        # 3) Same-name customized priority rule: hand [#295 uncustomized, #295 customized x2]
        obs_dup = obs_s228.model_copy(
            update={
                "cards": [
                    RecognizedCard(card_id=295, name="スポットライト+", type="mental", upgraded=True, playable=True, visual_playable=True, customized=False, customization_count=0, bbox=box),
                    RecognizedCard(card_id=295, name="スポットライト+", type="mental", upgraded=True, playable=True, visual_playable=True, customized=True, customization_count=2, bbox=box),
                ]
            }
        )
        _, acts_dup = pb.build_prompt(obs_dup, tr)
        aid_uncust = next(iter(planner._map_legal_actions_to_canonical([acts_dup[0]]).keys()))
        planner._ismcts.select_action = lambda *_a, **_kw: (aid_uncust, {"best_q_score": 900_000.0, "top_actions": []})  # type: ignore[union-attr,method-assign]
        cid_dup, reason_dup, _ = planner.decide_battle_action(obs_dup, acts_dup, tracker=tr)
        assert cid_dup == 1, f"Expected customized copy choice_id=1, got {cid_dup} ({reason_dup})"
        assert acts_dup[cid_dup].payload.get("customized") is True

        # 5) W8 455->55 P-point truncation override & W6 休み確認 header reject / cancel
        rec = LiveScreenRecognizer(db=db, plan="sense", scenario_id="hif_honsen")
        shop_frame = np.full((1600, 720, 3), 240, dtype=np.uint8)
        rec._last_p_points = 0
        monkeypatch.setattr(rec, "_ocr_bw_banner_number", lambda *_a, **_kw: 55)
        monkeypatch.setattr(
            rec,
            "_ocr_crop_upscaled",
            lambda _fr, box_t, **kw: "455" if box_t[0] < int(720 * 0.45) else ("55" if kw.get("use_en") else ""),
        )
        monkeypatch.setattr(rec.ocr, "recognize_array", lambda *_a, **_kw: {"text": "", "lines": [{"text": "インターバル", "bbox": [50, 38, 170, 56]}, {"text": "42/44", "bbox": [295, 57, 426, 83]}]})
        monkeypatch.setattr(rec.en_ocr, "recognize_array", lambda *_a, **_kw: {"text": "", "lines": []})
        meta_w8 = rec._parse_week_and_stats(
            [{"text": "インターバル", "bbox": [50, 38, 170, 56]}, {"text": "42/44", "bbox": [295, 57, 426, 83]}],
            shop_frame,
        )
        assert meta_w8["p_points"] == 455

        opts_rest = rec._extract_bottom_and_list_options(
            [{"text": "休み確認", "bbox": [260, 410, 460, 445]}, {"text": "キャンセル", "bbox": [160, 1120, 300, 1165]}, {"text": "休む", "bbox": [430, 1120, 540, 1165]}],
            720,
            1600,
            frame=None,
        )
        assert "休み確認" not in [o["name"] for o in opts_rest]

        agent = LocalHybridDecisionAgent(db=db)
        agent.reset_session()
        obs_w6_rest = ScreenObservation(
            scene_type="popup",
            plan="sense",
            scenario_id="hif_honsen",
            week=6,
            p_points=380,
            raw_text="休み確認 休む キャンセル",
        )
        acts_w6_rest = [
            LegalAction(choice_id=0, action_type="pick_option", description="休む", bbox=box, payload={"name": "休む"}),
            LegalAction(choice_id=1, action_type="pick_option", description="キャンセル", bbox=box, payload={"name": "キャンセル"}),
        ]
        d_rest = agent.decide("", acts_w6_rest, obs=obs_w6_rest)
        assert d_rest.choice_id == 1
        agent.close()
    finally:
        db.close()


def test_hand_animation_core_motion_mid_battle_modal_guard_and_week_reuse(monkeypatch, tmp_path):
    """Verify card-core motion filtering, #22 modal in_battle preservation, and week policy reuse."""
    from demo.maa_bridge.adb_controller import AdbController
    from demo.maa_bridge.runner import _compute_hand_card_core_motion
    from demo.sim.maskable_ppo import MaskablePPOTrainer
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    # 1. _compute_hand_card_core_motion ignores 3D background outside card cores, catches real card motion
    card_boxes = [
        BoundingBox(xmin=20, ymin=1184, xmax=181, ymax=1456),
        BoundingBox(xmin=185, ymin=1184, xmax=346, ymax=1456),
    ]
    cards = [
        RecognizedCard(card_id=776, name="話題沸騰+", playable=True, bbox=card_boxes[0]),
        RecognizedCard(card_id=295, name="スポットライト+", playable=True, bbox=card_boxes[1]),
    ]
    frame_a = np.full((1600, 720, 3), 120, dtype=np.uint8)
    for b in card_boxes:
        frame_a[b.ymin : b.ymax, b.xmin : b.xmax] = 60

    frame_bg_moved = np.clip(frame_a.astype(np.int32) + 80, 0, 255).astype(np.uint8)
    for b in card_boxes:
        frame_bg_moved[b.ymin : b.ymax, b.xmin : b.xmax] = 60
    mad_bg, used_cores_bg = _compute_hand_card_core_motion(frame_a, frame_bg_moved, cards)
    assert used_cores_bg is True
    assert mad_bg < 1.0

    frame_card_moved = frame_a.copy()
    frame_card_moved[1260:1420, 50:150] = 220
    mad_card, used_cores_c = _compute_hand_card_core_motion(frame_a, frame_card_moved, cards)
    assert used_cores_c is True
    assert mad_card > 40.0

    # 2. GakumasRunner.run_once(): static hand + animated background -> 1x recognize(), real slide -> settles & reuses frame
    class _FakeAdb(AdbController):
        def __init__(self, frames_seq):
            super().__init__(host="127.0.0.1", port=5555)
            self._connected = True
            self._frames = list(frames_seq)
            self._idx = 0

        @property
        def is_connected(self) -> bool:
            return True

        def connect(self) -> bool:
            return True

        def screencap(self) -> np.ndarray:
            f = self._frames[min(self._idx, len(self._frames) - 1)]
            self._idx += 1
            return f

        def click(self, x: int, y: int) -> bool:
            return True

    db = GakumasDB("gakumas_library.db")
    try:
        box = BoundingBox(xmin=20, ymin=1184, xmax=181, ymax=1456)
        battle_obs_static = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=7,
            turn=1,
            turns_remaining=9,
            stamina=40,
            max_stamina=44,
            cards=cards,
            options=[{"choice_id": 0, "name": "SKIP", "title": "SKIP", "bbox": {"xmin": 580, "ymin": 1100, "xmax": 700, "ymax": 1160}}],
        )

        rec_calls_static = []
        agent_static = SimpleNamespace(
            decide=lambda *_a, **_kw: DecisionResult(choice_id=0, reason="skip", source="test"),
            reset_battle=lambda: None,
        )
        ctrl_static = _FakeAdb([frame_a, frame_bg_moved, frame_bg_moved])
        runner_static = GakumasRunner(
            mode="adb",
            controller=ctrl_static,
            agent=agent_static,  # type: ignore[arg-type]
            db=db,
            plan="sense",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        runner_static.scene_routing_reco = SimpleNamespace(
            recognize=lambda _fr: (rec_calls_static.append(1), battle_obs_static)[1]
        )
        monkeypatch.setattr("demo.maa_bridge.runner.time.sleep", lambda *_a: None)
        runner_static.run_once()
        assert len(rec_calls_static) == 1

        frame_settled = frame_a.copy()
        rec_calls_slide = []
        ctrl_slide = _FakeAdb([frame_a, frame_card_moved, frame_settled, frame_settled, frame_settled])
        runner_slide = GakumasRunner(
            mode="adb",
            controller=ctrl_slide,
            agent=agent_static,  # type: ignore[arg-type]
            db=db,
            plan="sense",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        runner_slide.scene_routing_reco = SimpleNamespace(
            recognize=lambda _fr: (rec_calls_slide.append(1), battle_obs_static)[1]
        )
        runner_slide.run_once()
        assert len(rec_calls_slide) == 2

        # 3. LiveScreenRecognizer preserves _in_battle_scene and rem=1 on typo "スキルカートが選択されていません"
        rec = LiveScreenRecognizer(db=db, plan="sense", scenario_id="hif_honsen")
        rec._in_battle_scene = True
        rec._last_turns_remaining = 1
        blank_img = np.full((1600, 720, 3), 210, dtype=np.uint8)
        monkeypatch.setattr(
            rec.ocr,
            "recognize_array",
            lambda *_a, **_kw: {"text": "スキルカートが選択されていません", "lines": [{"text": "スキルカートが選択されていません", "bbox": [160, 220, 560, 260]}]},
        )
        monkeypatch.setattr(rec.en_ocr, "recognize_array", lambda *_a, **_kw: {"text": "", "lines": []})
        obs_typo = rec.recognize(blank_img, plan="sense")
        assert obs_typo.scene_type == "card_reward"
        assert obs_typo.turns_remaining == 1
        assert rec._in_battle_scene is True

        # Post-battle reward with 受け取る clears _in_battle_scene
        monkeypatch.setattr(
            rec.ocr,
            "recognize_array",
            lambda *_a, **_kw: {
                "text": "スキルカード獲得 受け取る",
                "lines": [
                    {"text": "スキルカード獲得", "bbox": [120, 200, 420, 240]},
                    {"text": "受け取る", "bbox": [280, 1460, 440, 1520]},
                ],
            },
        )
        obs_post = rec.recognize(blank_img, plan="sense")
        assert obs_post.scene_type == "card_reward"
        assert rec._in_battle_scene is False

        # 4. Runner + LocalHybridDecisionAgent keep in_battle & planner._sim on mid-battle card_reward without post-battle cues
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(
                ppo_hidden_dim=32, ppo_episodes=16, ppo_episodes_per_update=8, ismcts_simulations=2, policy_cache_dir=None
            ),
        )
        agent.tracker = GameStateTracker(plan="sense", p_idol_id=139)
        agent.tracker.produce_deck = [814, 163, 295, 776, 642, 147, 159, 105]
        obs_b = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=7,
            turn=1,
            turns_remaining=9,
            stamina=40,
            max_stamina=44,
            buffs={"goodConditionTurns": 6, "concentration": 6},
            cards=cards,
            options=[{"choice_id": 2, "name": "SKIP", "title": "SKIP", "bbox": {"xmin": 580, "ymin": 1100, "xmax": 700, "ymax": 1160}}],
        )
        ctrl_guard = MockController()
        runner_guard = GakumasRunner(
            mode="mock",
            controller=ctrl_guard,
            agent=agent,
            db=db,
            tracker=agent.tracker,
            plan="sense",
            scenario_id="hif_honsen",
            session_mode=True,
        )
        mid_modal_obs = ScreenObservation(
            scene_type="card_reward",
            plan="sense",
            scenario_id="hif_honsen",
            week=7,
            raw_text="スキルカートが選択されていません",
            options=[{"choice_id": 0, "name": "换出候选卡1", "description": "卡库槽位", "bbox": {"xmin": 60, "ymin": 400, "xmax": 200, "ymax": 540}}],
        )
        post_reward_obs = ScreenObservation(
            scene_type="card_reward",
            plan="sense",
            scenario_id="hif_honsen",
            week=7,
            raw_text="スキルカード獲得 受け取る",
            options=[{"choice_id": 0, "name": "受け取る", "description": "接受新卡牌", "bbox": {"xmin": 260, "ymin": 1440, "xmax": 460, "ymax": 1520}}],
        )
        seq_obs = [obs_b, mid_modal_obs, post_reward_obs]
        runner_guard.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: seq_obs.pop(0))

        runner_guard.run_once()
        assert runner_guard.tracker.in_battle is True
        assert agent.battle_planner._sim is not None

        runner_guard.run_once()
        assert runner_guard.tracker.in_battle is True
        assert agent.battle_planner._sim is not None

        runner_guard.run_once()
        assert runner_guard.tracker.in_battle is False
        assert agent.battle_planner._sim is None

        # 5. LiveBattlePlanner reuses active week policy after accidental reset_battle() & caps turns=1 episodes to 256
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_hidden_dim=32, ppo_episodes=32, ppo_episodes_per_update=16, ismcts_simulations=2, policy_cache_dir=None),
        )
        tr_w = GameStateTracker(plan="sense", p_idol_id=139)
        tr_w.produce_deck = [814, 163, 295, 776, 642, 147, 159, 105]
        acts_b = [
            LegalAction(choice_id=0, action_type="play_card", description="話題沸騰+", bbox=box, payload={"card_id": 776, "name": "話題沸騰+"}),
            LegalAction(choice_id=1, action_type="end_turn", description="SKIP", bbox=box, payload={"name": "SKIP"}),
        ]
        train_eps_seen = []
        orig_train = MaskablePPOTrainer.train

        def _spy_train(self_tr, *a, **kw):
            train_eps_seen.append(self_tr.cfg.total_episodes)
            return orig_train(self_tr, *a, **kw)

        monkeypatch.setattr(MaskablePPOTrainer, "train", _spy_train)
        planner.decide_battle_action(obs_b, acts_b, tracker=tr_w)
        assert len(train_eps_seen) == 1

        planner.reset_battle()
        tr_w.produce_deck.append(295)
        obs_rem1 = obs_b.model_copy(update={"turn": 9, "turns_remaining": 1})
        planner.decide_battle_action(obs_rem1, acts_b, tracker=tr_w)
        assert len(train_eps_seen) == 1  # Week policy cache avoided retrain!

        # Verify turns=1 episode cap == 256 on fresh week
        planner_cap = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_hidden_dim=32, ppo_episodes=2000, ppo_episodes_per_update=32, ismcts_simulations=2, policy_cache_dir=str(tmp_path / "cap_cache")),
        )
        monkeypatch.setattr(MaskablePPOTrainer, "train", lambda self_tr, *a, **kw: (train_eps_seen.append(self_tr.cfg.total_episodes), (self_tr.net, {"episodes_trained": self_tr.cfg.total_episodes, "best_val_mean": 1.0, "final_val_mean": 1.0, "history": []}))[1])
        obs_w9_rem1 = obs_b.model_copy(update={"week": 9, "turns_remaining": 1})
        planner_cap.decide_battle_action(obs_w9_rem1, acts_b, tracker=tr_w)
        assert train_eps_seen[-1] == 256
    finally:
        db.close()


def test_drink18_and_drink21_deferred_when_extra_use_or_hand_empty(tmp_path):
    """Reproduce S141 W7 T1: #18 (+1 use) must defer when non-free playable cards <= uses,
    #21 (upgradeHand) must forbid empty hand, and T1 memory_passives floor must survive partial OCR."""
    from demo.sim.battle_simulator import (
        CardInstance,
        HeadlessBattleSimulator,
        _make_card_config,
        _make_drink_config,
    )

    db = GakumasDB("gakumas_library.db")
    try:
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(
                ppo_hidden_dim=32,
                ppo_episodes=32,
                ppo_episodes_per_update=16,
                ismcts_simulations=4,
                policy_cache_dir=str(tmp_path / "cache"),
            ),
        )
        tr = GameStateTracker(plan="sense", p_idol_id=139)
        tr.produce_deck = [814, 163, 585, 87, 812, 159, 105, 113]
        tr.memory_passives = [
            {"key": "start_good_condition", "label": "試験開始時 好調+Xターン", "value": 6},
            {"key": "start_concentration", "label": "試験開始時 集中+X", "value": 6},
        ]

        box = BoundingBox(xmin=0, ymin=0, xmax=100, ymax=100)
        # S141 hand: two free cards (#585, #87) + one non-free card (#812), card_uses_remaining=1,
        # partial T1 OCR buff {goodConditionTurns: 5} (missing concentration!).
        obs_t1 = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=7,
            turn=1,
            turns_remaining=9,
            stamina=29,
            max_stamina=29,
            genki=0,
            card_uses_remaining=1,
            hand_slot_count=3,
            week_activity="H.I.F本戦ラウンド1 (試験 ボーカル4887)",
            buffs={"goodConditionTurns": 5},
            cards=[
                RecognizedCard(card_id=585, name="夏夜に咲く思い出+", playable=True, bbox=box),
                RecognizedCard(card_id=87, name="シュプレヒコール+", playable=True, bbox=box),
                RecognizedCard(card_id=812, name="お姉さんの感覚+", playable=True, bbox=box),
            ],
            drinks=[
                RecognizedDrink(drink_id=18, name="初星ホエイプロテイン", playable=True, bbox=box),
                RecognizedDrink(drink_id=21, name="初星ブーストエナジー", playable=True, bbox=box),
            ],
            options=[{"choice_id": 5, "name": "SKIP", "title": "SKIP"}],
        )
        acts_t1 = [
            LegalAction(choice_id=0, action_type="play_card", description="夏夜に咲く思い出+", bbox=box, payload={"card_id": 585, "name": "夏夜に咲く思い出+"}),
            LegalAction(choice_id=1, action_type="play_card", description="シュプレヒコール+", bbox=box, payload={"card_id": 87, "name": "シュプレヒコール+"}),
            LegalAction(choice_id=2, action_type="play_card", description="お姉さんの感覚+", bbox=box, payload={"card_id": 812, "name": "お姉さんの感覚+"}),
            LegalAction(choice_id=3, action_type="use_drink", description="初星ホエイプロテイン", bbox=box, payload={"drink_id": 18, "name": "初星ホエイプロテイン"}),
            LegalAction(choice_id=4, action_type="use_drink", description="初星ブーストエナジー", bbox=box, payload={"drink_id": 21, "name": "初星ブーストエナジー"}),
            LegalAction(choice_id=5, action_type="end_turn", description="SKIP", bbox=box, payload={"name": "SKIP"}),
        ]

        cid1, _reason1, _ = planner.decide_battle_action(obs_t1, acts_t1, tracker=tr)
        assert planner._battle_cfg is not None
        assert planner._sim is not None
        assert planner._battle_cfg.initial_good_condition >= 6
        assert planner._battle_cfg.initial_concentration >= 6
        assert planner._sim.buffs["goodConditionTurns"] >= 6
        assert planner._sim.buffs["concentration"] >= 6

        d18 = _make_drink_config(db, 18)
        d21 = _make_drink_config(db, 21)
        assert planner._sim.is_pure_extra_use_drink(d18) is True
        assert planner._sim.is_hand_upgrade_drink(d21) is True
        assert planner._sim.count_playable_non_free_cards() == 1
        assert planner._sim.should_defer_drink_now(d18) is True
        assert cid1 != 3  # Never waste Drink #18 when hand only has 1 non-free card

        # Simulate Turn 1 post-play empty hand: neither #18 nor #21 may be drunk on 0 cards!
        planner.notify_action_executed(acts_t1[0])
        planner.notify_action_executed(acts_t1[1])
        planner.notify_action_executed(acts_t1[2])
        obs_empty = obs_t1.model_copy(update={"cards": [], "hand_slot_count": 0, "genki": 6})
        acts_empty = [
            LegalAction(choice_id=3, action_type="use_drink", description="初星ホエイプロテイン", bbox=box, payload={"drink_id": 18, "name": "初星ホエイプロテイン"}),
            LegalAction(choice_id=4, action_type="use_drink", description="初星ブーストエナジー", bbox=box, payload={"drink_id": 21, "name": "初星ブーストエナジー"}),
            LegalAction(choice_id=5, action_type="end_turn", description="SKIP", bbox=box, payload={"name": "SKIP"}),
        ]
        assert planner._sim.can_use_drink(d18, hand_size=0) is False
        assert planner._sim.can_use_drink(d21, hand_size=0) is False
        cid_empty, _r_empty, _ = planner.decide_battle_action(obs_empty, acts_empty, tracker=tr)
        assert cid_empty == 5  # Must SKIP rather than wasting #21/#18 on empty hand

        # With #812 (non-free) + #159 (free cardUsesRemaining+=1) and 1 use remaining, only 1 non-free card -> defer #18
        sim_two = HeadlessBattleSimulator(planner._battle_cfg, db=db, seed=1, record_logs=False)
        sim_two.stamina = 30
        sim_two.card_uses_remaining = 1
        sim_two.hand.clear()
        for c_id in (812, 159):
            c_cfg = _make_card_config(db, c_id)
            assert c_cfg is not None
            sim_two.hand.append(CardInstance(uid=sim_two.next_uid, cfg=c_cfg))
            sim_two.next_uid += 1
        assert sim_two.count_playable_non_free_cards() == 1
        assert sim_two.should_defer_drink_now(d18) is True

        # Replacing #159 with non-free #105 gives 2 playable non-free cards > 1 remaining use -> #18 IS allowed!
        c105 = _make_card_config(db, 105)
        assert c105 is not None
        sim_two.hand[1] = CardInstance(uid=sim_two.next_uid, cfg=c105)
        sim_two.next_uid += 1
        assert sim_two.count_playable_non_free_cards() == 2
        assert sim_two.should_defer_drink_now(d18) is False
    finally:
        db.close()


def test_unfired_163_play_rolls_back_phantom_2x_and_retries_before_814():
    """Regression (W7 Step 202->203): if #162/#163 stays in hand after notify_action_executed, 2x must roll back to 0 so #814 is never fired naked."""
    from demo.core.db import GakumasDB
    from demo.core.tracker import GameStateTracker
    from demo.schemas import BoundingBox, LegalAction, RecognizedCard, ScreenObservation
    from demo.sim.live_battle_planner import LiveBattlePlanner, LiveBattlePlannerConfig

    db = GakumasDB("gakumas_library.db")
    try:
        tr = GameStateTracker()
        tr.p_idol_id = 139
        tr.produce_deck = [812, 814, 112, 145, 147, 295, 166, 117, 165, 599, 163, 706, 776, 301, 23, 87, 593, 585, 889, 74]

        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_episodes=2, ppo_episodes_per_update=2, ismcts_simulations=12, seed=7),
        )
        box = BoundingBox(xmin=0, ymin=0, xmax=80, ymax=120)

        def _rc(cid: int, name: str) -> RecognizedCard:
            return RecognizedCard(card_id=cid, name=name, playable=True, visual_playable=True, bbox=box)

        # Step 202: rem=2, 5-card hand with #162/#163 and #813/#814
        obs_202 = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=7,
            turn=8,
            turns_remaining=2,
            stamina=35,
            max_stamina=44,
            genki=26,
            score=155476,
            buffs={"goodConditionTurns": 109, "perfectConditionTurns": 15, "concentration": 16},
            cards=[
                _rc(162, "国民的アイドル"),
                _rc(813, "自然体の魅力"),
                _rc(74, "アドリブ"),
                _rc(295, "スポットライト+"),
                _rc(775, "話題沸騰"),
            ],
            hand_slot_count=5,
            drinks=[],
        )
        acts_202 = [
            LegalAction(choice_id=0, action_type="play_card", description="打出卡牌: 国民的アイドル [⚡免费+1使用次数(不耗行动点)]", bbox=box, payload={"card_id": 162, "base_id": 162, "limit_count": 1, "dsl_actions": "doubleCardEffectCards+=1; cardUsesRemaining+=1", "name": "国民的アイドル"}),
            LegalAction(choice_id=1, action_type="play_card", description="打出卡牌: 自然体の魅力", bbox=box, payload={"card_id": 813, "base_id": 813, "limit_count": 1, "dsl_actions": "score+=stamina*8; score+=2+cardsUsed*9", "name": "自然体の魅力"}),
            LegalAction(choice_id=2, action_type="play_card", description="打出卡牌: アドリブ", bbox=box, payload={"card_id": 74, "base_id": 74, "dsl_actions": "score+=5; goodConditionTurns+=3", "name": "アドリブ"}),
            LegalAction(choice_id=3, action_type="play_card", description="打出卡牌: スポットライト+", bbox=box, payload={"card_id": 295, "base_id": 294, "dsl_actions": "genki+=9; goodConditionTurns+=6", "name": "スポットライト+"}),
            LegalAction(choice_id=4, action_type="play_card", description="打出卡牌: 話題沸騰", bbox=box, payload={"card_id": 775, "base_id": 775, "limit_count": 1, "dsl_actions": "goodConditionTurnsMultiplier=2; score+=37; setScoreBuff(0.3,1)", "name": "話題沸騰"}),
            LegalAction(choice_id=5, action_type="end_turn", description="SKIP", bbox=box, payload={"name": "SKIP"}),
        ]

        cid_202, _r202, _ = planner.decide_battle_action(obs_202, acts_202, tracker=tr)
        assert cid_202 == 0
        planner.notify_action_executed(acts_202[0])
        assert planner._sim.double_card_effect_cards == 1

        # Step 203: edge tap missed on device -> #162 is STILL in obs.cards!
        obs_203 = obs_202.model_copy(
            update={
                "cards": [
                    _rc(162, "国民的アイドル"),
                    _rc(813, "自然体の魅力"),
                    _rc(74, "アドリブ"),
                    _rc(295, "スポットライト+"),
                ],
                "hand_slot_count": 4,
            }
        )
        acts_203 = [acts_202[0], acts_202[1], acts_202[2], acts_202[3], acts_202[5]]
        cid_203, _r203, _ = planner.decide_battle_action(obs_203, acts_203, tracker=tr)
        # Must roll back phantom 2x and re-play #162 BEFORE firing #813!
        assert planner._sim.double_card_effect_cards == 0
        assert cid_203 == 0
    finally:
        db.close()


def test_stuck_exam_turn_ocr_advances_via_sim_and_fires_163_814_combo():
    """Regression (W9 Steps 285->297): stuck HUD turns_remaining=5 must still count down 5->4->3->2->1 and fire #12/#163->#814."""
    from demo.core.db import GakumasDB
    from demo.core.tracker import GameStateTracker
    from demo.schemas import BoundingBox, LegalAction, RecognizedCard, RecognizedDrink, ScreenObservation
    from demo.sim.live_battle_planner import LiveBattlePlanner, LiveBattlePlannerConfig

    db = GakumasDB("gakumas_library.db")
    try:
        tr = GameStateTracker()
        tr.p_idol_id = 139
        tr.produce_deck = [812, 814, 112, 145, 147, 295, 166, 117, 165, 599, 163, 707, 776, 301, 23, 87, 593, 585, 890, 74, 642, 667]

        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_episodes=2, ppo_episodes_per_update=2, ismcts_simulations=16, seed=11),
        )
        box = BoundingBox(xmin=0, ymin=0, xmax=80, ymax=120)

        def _rc(cid: int, name: str) -> RecognizedCard:
            return RecognizedCard(card_id=cid, name=name, playable=True, visual_playable=True, bbox=box)

        drinks = [
            RecognizedDrink(drink_id=12, name="ブーストエキス", playable=True, bbox=box),
            RecognizedDrink(drink_id=22, name="初星黒酢", playable=True, bbox=box),
        ]

        # Initialize battle at T8/rem=5 (matching Step 285), play non-free #642 to end T8
        obs_t8 = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=9,
            turn=8,
            turns_remaining=5,
            stamina=32,
            max_stamina=44,
            genki=9,
            score=54941,
            buffs={"goodConditionTurns": 70, "perfectConditionTurns": 1, "concentration": 30},
            cards=[_rc(642, "ステージングの基本")],
            drinks=drinks,
        )
        act_642 = LegalAction(choice_id=0, action_type="play_card", description="打出卡牌: ステージングの基本", bbox=box, payload={"card_id": 642, "base_id": 641, "limit_count": 1, "dsl_actions": "goodConditionTurnsMultiplier=2; score+=10", "name": "ステージングの基本"})
        act_end = LegalAction(choice_id=9, action_type="end_turn", description="SKIP", bbox=box, payload={"name": "SKIP"})
        planner.decide_battle_action(obs_t8, [act_642, act_end], tracker=tr)
        # Align initial_turns to 12 like real W9 battle
        planner._battle_initial_turns = 12
        planner._sim.turn = 8
        planner._sim.turns_remaining = 5
        planner.notify_action_executed(act_642)
        assert planner._sim.turns_remaining == 4

        # T9 (actual rem=4, HUD stuck at rem=5): play #301 to finish T9
        obs_t9 = obs_t8.model_copy(update={"turns_remaining": 5, "cards": [_rc(163, "国民的アイドル+"), _rc(301, "至高のエンタメ+"), _rc(814, "自然体の魅力+")]})
        act_301 = LegalAction(choice_id=1, action_type="play_card", description="打出卡牌: 至高のエンタメ+", bbox=box, payload={"card_id": 301, "base_id": 300, "limit_count": 1, "dsl_actions": "at:turn { drawCard; limit:1 }", "name": "至高のエンタメ+"})
        planner.decide_battle_action(obs_t9, [act_301, act_end], tracker=tr)
        assert planner._sim.turns_remaining == 4
        assert obs_t9.turns_remaining == 4
        planner.notify_action_executed(act_301)

        # T10 (actual rem=3, HUD still stuck at rem=5): play #294 to finish T10
        obs_t10 = obs_t8.model_copy(update={"turns_remaining": 5, "cards": [_rc(74, "アドリブ"), _rc(294, "スポットライト"), _rc(814, "自然体の魅力+")]})
        act_294 = LegalAction(choice_id=1, action_type="play_card", description="打出卡牌: スポットライト", bbox=box, payload={"card_id": 294, "base_id": 294, "dsl_actions": "genki+=7; goodConditionTurns+=5", "name": "スポットライト"})
        planner.decide_battle_action(obs_t10, [act_294, act_end], tracker=tr)
        assert planner._sim.turns_remaining == 3
        assert obs_t10.turns_remaining == 3
        planner.notify_action_executed(act_294)

        # T11 (actual rem=2, HUD STILL stuck at rem=5!): hand has [#163, #776, #814, #74] + drinks [#12, #22]
        obs_t11 = obs_t8.model_copy(
            update={
                "turns_remaining": 5,
                "stamina": 41,
                "genki": 24,
                "score": 145668,
                "buffs": {"goodConditionTurns": 120, "perfectConditionTurns": 9, "concentration": 37},
                "cards": [
                    _rc(163, "国民的アイドル+"),
                    _rc(776, "話題沸騰+"),
                    _rc(814, "自然体の魅力+"),
                    _rc(74, "アドリブ"),
                ],
                "drinks": drinks,
            }
        )
        acts_t11 = [
            LegalAction(choice_id=0, action_type="play_card", description="打出卡牌: 国民的アイドル+ [⚡免费+1使用次数(不耗行动点)]", bbox=box, payload={"card_id": 163, "base_id": 162, "limit_count": 1, "dsl_actions": "genki+=8; doubleCardEffectCards+=1; cardUsesRemaining+=1", "name": "国民的アイドル+"}),
            LegalAction(choice_id=1, action_type="play_card", description="打出卡牌: 話題沸騰+", bbox=box, payload={"card_id": 776, "base_id": 775, "limit_count": 1, "dsl_actions": "goodConditionTurnsMultiplier=2.5; score+=50; setScoreBuff(0.3,1)", "name": "話題沸騰+"}),
            LegalAction(choice_id=2, action_type="play_card", description="打出卡牌: 自然体の魅力+", bbox=box, payload={"card_id": 814, "base_id": 813, "limit_count": 1, "dsl_actions": "fixedStamina+=maxStamina*0.2; score+=stamina*10; score+=2+cardsUsed*14", "name": "自然体の魅力+"}),
            LegalAction(choice_id=3, action_type="play_card", description="打出卡牌: アドリブ", bbox=box, payload={"card_id": 74, "base_id": 74, "dsl_actions": "score+=5; goodConditionTurns+=3", "name": "アドリブ"}),
            LegalAction(choice_id=4, action_type="use_drink", description="使用饮料: ブーストエキス", bbox=box, payload={"drink_id": 12, "name": "ブーストエキス"}),
            LegalAction(choice_id=5, action_type="use_drink", description="使用饮料: 初星黒酢", bbox=box, payload={"drink_id": 22, "name": "初星黒酢"}),
            act_end,
        ]
        cid_t11, reason_t11, _ = planner.decide_battle_action(obs_t11, acts_t11, tracker=tr)
        assert planner._sim.turns_remaining == 2
        assert obs_t11.turns_remaining == 2
        assert "rem=2" in reason_t11
        # Must NEVER dump #74 アドリブ when #12 / #163 -> #814 burst is unlocked at rem=2!
        assert cid_t11 in (0, 1, 4)
        assert cid_t11 != 3
    finally:
        db.close()


def test_exam_turn_ring_ocr_and_two_frame_deadlock_recovery():
    """Verify full-stem exam turn-ring OCR on saved live11 frames and 2-frame recovery across >2 turn drops."""
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    rec = LiveScreenRecognizer(scenario_id="hif_honsen", plan="sense")
    p_rem2 = Path("debug_frames_live11/battle_W9_rem2_step296.png")
    p_rem4 = Path("debug_frames_live11/battle_W9_rem4_step291.png")
    p_rem4_w7 = Path("debug_frames_live11/battle_W7_rem4_step175.png")
    p_rem6 = Path("debug_frames_live11/battle_W7_rem6_step168.png")
    for p_img, expected_rem in ((p_rem4, 4), (p_rem4_w7, 4)):
        if p_img.exists():
            rec._in_battle_scene = False
            rec._last_turns_remaining = 12
            obs_chk = rec.recognize(np.array(Image.open(p_img).convert("RGB")))
            assert obs_chk.scene_type == "battle_turn"
            assert obs_chk.turns_remaining == expected_rem
    if p_rem6.exists():
        # Dimmed mid-animation frame: verify _parse_battle_screen dimmed-ring fallback reads 6 cleanly
        img6 = np.array(Image.open(p_rem6).convert("RGB"))
        rec._in_battle_scene = False
        rec._last_turns_remaining = 10
        obs6 = rec._parse_battle_screen(
            img6,
            [{"text": "ビジュアル 4800%", "bbox": [120, 40, 220, 75]}],
            "sense",
            {"week": 7, "week_label": "H.I.F本戦ラウンド1", "total_weeks": 9, "stamina": 42, "max_stamina": 44, "p_points": 255, "stats": {"vocal": 2000, "dance": 2600, "visual": 2300}},
            turn_box_txt="残りターン",
        )
        assert obs6.turns_remaining == 6
    if p_rem2.exists():
        img2 = np.array(Image.open(p_rem2).convert("RGB"))
        rec._in_battle_scene = True
        rec._last_turns_remaining = 5
        rec._turn_drop_candidate = None
        # Frame 1 with diff=3 (5->2) records candidate without immediate jump
        obs_a = rec.recognize(img2)
        assert obs_a.turns_remaining == 5
        assert rec._turn_drop_candidate == (2, 1)
        # Frame 2 confirms rem=2 and breaks out of the stuck=5 deadlock!
        obs_b = rec.recognize(img2)
        assert obs_b.turns_remaining == 2
        assert rec._turn_drop_candidate is None
    rec.db.close()


def test_drink25_and_kimegao778_last_5_turns_hold_and_combo_priority(tmp_path: Path):
    """#777/#778 and #25 hold until turns_remaining<=5, reserve #25 for #778 in rem>=3, and execute #25->#778 cleanly."""
    import random
    from demo.sim.battle_simulator import CardInstance, HeadlessBattleSimulator, _make_card_config
    from demo.sim.ismcts import ISMCTSAgent, ISMCTSConfig
    from demo.sim.maskable_ppo import PolicyValueNet
    from demo.sim.rl_env import GakumasActionCodec, GakumasFeatureEncoder

    db = GakumasDB("gakumas_library.db")
    try:
        box = BoundingBox(xmin=10, ymin=10, xmax=80, ymax=120)
        deck_ids = [920, 922, 299, 179, 175, 137, 778, 127, 129, 159]
        cfg = build_live_battle_config(
            db=db,
            deck_card_ids=deck_ids,
            equipped_p_items=[],
            drinks=[RecognizedDrink(drink_id=25, name="特製ハツボシエキス", playable=True, bbox=box)],
            idol_stats={"vocal": 2663, "dance": 1314, "visual": 1886},
            stamina=34,
            max_stamina=44,
            total_turns=9,
            week=7,
            plan="logic",
            card_memory_hooks={},
        )
        codec = GakumasActionCodec(cfg, db=db)
        enc = GakumasFeatureEncoder(codec)
        net = PolicyValueNet(enc.obs_dim, codec.num_actions, hidden_dim=32, seed=13)
        mcts = ISMCTSAgent(codec, enc, net, ISMCTSConfig(num_simulations=8))
        d25_aid = codec.drink_id_to_action[25]
        c778_cfg = _make_card_config(db, 778)
        c778_aid = codec.card_slot_index(c778_cfg)

        def _setup_sim(rem: int, hand_ids: list[int], draw_ids: list[int], *, stamina: int = 34, genki: int = 20, gi: int = 20):
            s = HeadlessBattleSimulator(cfg, db=db, seed=100 + rem, record_logs=False)
            s.turn = max(1, cfg.total_turns - rem + 1)
            s.turns_remaining = rem
            s.stamina = stamina
            s.genki = genki
            s.buffs["goodImpressionTurns"] = gi
            s.buffs["motivation"] = 15
            s.buffs["nullifyDebuff"] = 0
            s.card_uses_remaining = 1

            def _mk_zone(ids: list[int]) -> list[CardInstance]:
                zone = []
                for cid_z in ids:
                    uid_z = s.next_uid
                    s.next_uid += 1
                    inst_z = CardInstance(uid=uid_z, cfg=_make_card_config(db, cid_z))
                    s._all_instances[uid_z] = inst_z
                    zone.append(inst_z)
                return zone

            s.hand = _mk_zone(hand_ids)
            s.draw_pile = _mk_zone(draw_ids)
            s.discard_pile = []
            s.exhaust_pile = []
            return s

        # 1. Battle 1 Step 151 reproduction (rem=8, hand=[#778], drinks=[#25]): hold both and choose SKIP/end_turn
        sim_step151 = _setup_sim(8, [778], [922, 179], stamina=34, genki=4, gi=17)
        dr25_s1 = sim_step151.available_drinks[0]
        assert sim_step151.should_hold_for_finisher(card_cfg=c778_cfg, fin_window=0)
        assert sim_step151.should_hold_for_finisher(drink_cfg=dr25_s1, fin_window=0)
        mask_151, _ = codec.legal_mask_and_map(sim_step151, prune_dominated=True)
        assert not mask_151[c778_aid]
        assert not mask_151[d25_aid]
        assert mask_151[codec.end_turn_action_id]
        aid_151, _ = mcts.select_action(sim_step151, random.Random(7))
        assert aid_151 == codec.end_turn_action_id

        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(
                ppo_episodes=2,
                ppo_episodes_per_update=2,
                ismcts_simulations=8,
                seed=7,
                policy_cache_dir=str(tmp_path),
            ),
        )
        tr = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tr.produce_deck = deck_ids
        tr.idol_stats = {"vocal": 2663, "dance": 1314, "visual": 1886}

        def _rc(cid: int, name: str) -> RecognizedCard:
            return RecognizedCard(card_id=cid, name=name, upgraded=("+" in name), playable=True, visual_playable=True, bbox=box)

        drinks_obs = [RecognizedDrink(drink_id=25, name="特製ハツボシエキス", playable=True, bbox=box)]
        act_d25 = LegalAction(choice_id=0, action_type="use_drink", description="使用饮料: 特製ハツボシエキス", bbox=box, payload={"drink_id": 25, "name": "特製ハツボシエキス"})
        act_778 = LegalAction(choice_id=1, action_type="play_card", description="打出卡牌: びしっとキメ顔+", bbox=box, payload={"card_id": 778, "base_id": 777, "limit_count": 1, "name": "びしっとキメ顔+"})
        act_922 = LegalAction(choice_id=2, action_type="play_card", description="打出卡牌: 認めてくれるひと+", bbox=box, payload={"card_id": 922, "base_id": 921, "name": "認めてくれるひと+"})
        act_179 = LegalAction(choice_id=3, action_type="play_card", description="打出卡牌: トキメキ+", bbox=box, payload={"card_id": 179, "base_id": 179, "name": "トキメキ+"})
        act_299 = LegalAction(choice_id=5, action_type="play_card", description="打出卡牌: オトメゴコロ+ [⚡免费+1使用次数(不耗行动点)]", bbox=box, payload={"card_id": 299, "base_id": 298, "name": "オトメゴコロ+"})
        act_end = LegalAction(choice_id=9, action_type="end_turn", description="SKIP", bbox=box, payload={"name": "SKIP"})

        obs_151 = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=7,
            turn=2,
            turns_remaining=8,
            stamina=34,
            max_stamina=44,
            genki=4,
            score=1807,
            buffs={"goodImpressionTurns": 17, "motivation": 15},
            cards=[_rc(778, "びしっとキメ顔+")],
            drinks=drinks_obs,
        )
        cid_151, _, _ = planner.decide_battle_action(obs_151, [act_d25, act_778, act_end], tracker=tr)
        assert cid_151 == 9  # Must SKIP/end_turn instead of burning #778 or #25 at rem=8

        # 2. rem=6, hand=[#778, #179], drinks=[#25] -> picks #179 and holds #778 / #25
        planner.reset_battle()
        obs_rem6 = obs_151.model_copy(
            update={
                "turn": 4,
                "turns_remaining": 6,
                "buffs": {"goodImpressionTurns": 35, "motivation": 15},
                "cards": [_rc(778, "びしっとキメ顔+"), _rc(179, "トキメキ+")],
            }
        )
        cid_rem6, _, _ = planner.decide_battle_action(obs_rem6, [act_d25, act_778, act_179, act_end], tracker=tr)
        assert cid_rem6 == 3  # #179

        # 3. rem=4, hand=[#922, #175], #778 in draw_pile, 1 bottle of #25 -> reserve #25; at rem=2 -> #25 unlocks for #922
        sim_rem4 = _setup_sim(4, [922, 175], [778], stamina=30, genki=30, gi=120)
        assert sim_rem4.should_hold_for_finisher(drink_cfg=sim_rem4.available_drinks[0], fin_window=2)
        mask_rem4, _ = codec.legal_mask_and_map(sim_rem4, prune_dominated=True)
        assert not mask_rem4[d25_aid]
        sim_rem2 = _setup_sim(2, [922, 175], [778], stamina=30, genki=30, gi=220)
        assert not sim_rem2.should_hold_for_finisher(drink_cfg=sim_rem2.available_drinks[0], fin_window=2)
        mask_rem2, _ = codec.legal_mask_and_map(sim_rem2, prune_dominated=True)
        assert mask_rem2[d25_aid]

        # 4. rem=5, hand=[#778, #922, #179], drinks=[#25] -> step 1 prioritizes #25; step 2 (armed) locks onto #778 over #922
        planner.reset_battle()
        obs_rem5 = obs_151.model_copy(
            update={
                "turn": 5,
                "turns_remaining": 5,
                "genki": 30,
                "score": 80000,
                "buffs": {"goodImpressionTurns": 150, "motivation": 25},
                "cards": [_rc(778, "びしっとキメ顔+"), _rc(922, "認めてくれるひと+"), _rc(179, "トキメキ+")],
            }
        )
        cid_rem5_step1, _, _ = planner.decide_battle_action(
            obs_rem5, [act_d25, act_778, act_922, act_179, act_end], tracker=tr
        )
        assert cid_rem5_step1 == 0  # #25 first

        planner.notify_action_executed(act_d25)
        assert planner._sim is not None and planner._sim.has_armed_active_retrigger(same_turn_only=True)
        obs_rem5_armed = obs_rem5.model_copy(update={"stamina": 32, "drinks": []})
        cid_rem5_step2, _, _ = planner.decide_battle_action(
            obs_rem5_armed, [act_778, act_922, act_179, act_end], tracker=tr
        )
        assert cid_rem5_step2 == 1  # Locked onto #778 over #922

        # 5. rem=5, hand=[#299 (free Mental), #778], drinks=[#25] -> #299 plays BEFORE #25 -> #778
        planner.reset_battle()
        obs_rem5_free = obs_rem5.model_copy(
            update={"cards": [_rc(299, "オトメゴコロ+"), _rc(778, "びしっとキメ顔+")]}
        )
        cid_rem5_free, _, _ = planner.decide_battle_action(
            obs_rem5_free, [act_d25, act_299, act_778, act_end], tracker=tr
        )
        assert cid_rem5_free == 5  # Free Mental #299 plays first!

        # 6. Unaffordable post-drink cost (rem=5, stamina=7, genki=0, nullifyDebuff=0): can_drink25_combo_kimegao_now=False -> #778 playable without forcing #25
        sim_unaff = _setup_sim(5, [778], [922], stamina=7, genki=0, gi=150)
        assert not sim_unaff.can_drink25_combo_kimegao_now(sim_unaff.available_drinks[0])
        mask_unaff, _ = codec.legal_mask_and_map(sim_unaff, prune_dominated=True)
        assert mask_unaff[c778_aid]
        assert not mask_unaff[d25_aid]
    finally:
        db.close()


def test_plan_rules_strictly_isolated_between_sense_logic_and_anomaly(tmp_path: Path):
    """Verify Sense, Logic, and Anomaly rule classes are independent and never bleed across plans."""
    from demo.sim.battle_simulator import CardInstance, HeadlessBattleSimulator, _make_card_config, _make_drink_config
    from demo.sim.plan_rules import AnomalyBattleRules, LogicBattleRules, SenseBattleRules, get_plan_rules
    from demo.sim.rl_env import GakumasActionCodec

    sense_rules = get_plan_rules("sense")
    logic_rules = get_plan_rules("logic")
    anomaly_rules = get_plan_rules("anomaly")
    assert isinstance(sense_rules, SenseBattleRules)
    assert isinstance(logic_rules, LogicBattleRules)
    assert isinstance(anomaly_rules, AnomalyBattleRules)
    assert type(sense_rules) is not type(logic_rules)
    assert type(logic_rules) is not type(anomaly_rules)

    db = GakumasDB("gakumas_library.db")
    try:
        box = BoundingBox(xmin=10, ymin=10, xmax=80, ymax=120)
        c778 = _make_card_config(db, 778)
        c153 = _make_card_config(db, 153)
        c814 = _make_card_config(db, 814)
        c163 = _make_card_config(db, 163)
        c619 = _make_card_config(db, 619)
        c776 = _make_card_config(db, 776)
        d25 = _make_drink_config(db, 25)
        d22 = _make_drink_config(db, 22)

        # 1. Sense simulator ignores Logic rules (#778, #153, #25 combo)
        cfg_sense = build_live_battle_config(
            db=db,
            deck_card_ids=[812, 814, 163, 778, 153],
            equipped_p_items=[],
            drinks=[RecognizedDrink(drink_id=25, name="特製ハツボシエキス", playable=True, bbox=box)],
            idol_stats={"vocal": 2000, "dance": 2500, "visual": 2100},
            stamina=40,
            max_stamina=44,
            total_turns=9,
            week=7,
            plan="sense",
            p_idol_id=139,
        )
        sim_sense = HeadlessBattleSimulator(cfg_sense, db=db, seed=1, record_logs=False)
        sim_sense.turns_remaining = 8
        sim_sense.stamina = 35
        sim_sense.genki = 20
        sim_sense.buffs["goodImpressionTurns"] = 20
        assert sim_sense.should_hold_for_finisher(card_cfg=c778, fin_window=2) is False
        assert sim_sense.should_hold_for_finisher(card_cfg=c153, fin_window=2) is False
        assert sim_sense.can_drink25_combo_kimegao_now(d25) is False

        # LiveBattlePlanner(plan="sense") does NOT apply Logic hold rule on #778 at rem=8
        planner_sense = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(
                ppo_episodes=2, ppo_episodes_per_update=2, ismcts_simulations=4, seed=5, policy_cache_dir=str(tmp_path)
            ),
        )
        tr_sense = GameStateTracker(plan="sense", p_idol_id=139, scenario_id="hif_honsen")
        tr_sense.produce_deck = [812, 814, 163, 778]
        obs_sense = ScreenObservation(
            scene_type="battle_turn",
            plan="sense",
            scenario_id="hif_honsen",
            week=7,
            turn=2,
            turns_remaining=8,
            stamina=35,
            max_stamina=44,
            genki=20,
            score=2000,
            buffs={"goodConditionTurns": 20, "concentration": 15, "goodImpressionTurns": 20},
            cards=[RecognizedCard(card_id=778, name="びしっとキメ顔+", upgraded=True, playable=True, visual_playable=True, bbox=box)],
            drinks=[],
        )
        acts_sense = [
            LegalAction(choice_id=0, action_type="play_card", description="びしっとキメ顔+", bbox=box, payload={"card_id": 778, "base_id": 777, "name": "びしっとキメ顔+"}),
            LegalAction(choice_id=9, action_type="end_turn", description="SKIP", bbox=box, payload={"name": "SKIP"}),
        ]
        _, reason_sense, _ = planner_sense.decide_battle_action(obs_sense, acts_sense, tracker=tr_sense)
        assert "[Hold #777/#778 for last 5 turns]" not in reason_sense

        # 2. Logic simulator ignores Sense rules (#814, #163, #619, #776, #22)
        cfg_logic = build_live_battle_config(
            db=db,
            deck_card_ids=[920, 922, 814, 163, 619, 776],
            equipped_p_items=[],
            drinks=[RecognizedDrink(drink_id=22, name="初星黒酢", playable=True, bbox=box)],
            idol_stats={"vocal": 2500, "dance": 1300, "visual": 1900},
            stamina=40,
            max_stamina=44,
            total_turns=9,
            week=7,
            plan="logic",
            p_idol_id=163,
        )
        sim_logic = HeadlessBattleSimulator(cfg_logic, db=db, seed=2, record_logs=False)
        sim_logic.turns_remaining = 8
        assert sim_logic.should_hold_for_finisher(card_cfg=c814, fin_window=2) is False
        assert sim_logic.should_hold_for_finisher(card_cfg=c163, fin_window=2) is False
        assert sim_logic.should_hold_for_finisher(card_cfg=c619, fin_window=2) is False
        assert sim_logic.should_hold_for_finisher(card_cfg=c776, fin_window=2) is False
        assert sim_logic.should_hold_for_finisher(drink_cfg=d22, fin_window=2) is False

        # LiveBattlePlanner(plan="logic") does NOT run Sense #163->#814 combo/hold overrides
        planner_logic = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(
                ppo_episodes=2, ppo_episodes_per_update=2, ismcts_simulations=4, seed=5, policy_cache_dir=str(tmp_path)
            ),
        )
        tr_logic = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tr_logic.produce_deck = [920, 922, 814, 163]
        obs_logic = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=7,
            turn=8,
            turns_remaining=2,
            stamina=35,
            max_stamina=44,
            genki=25,
            score=50000,
            buffs={"goodImpressionTurns": 30, "motivation": 15, "goodConditionTurns": 20},
            cards=[
                RecognizedCard(card_id=814, name="自然体の魅力+", upgraded=True, playable=True, visual_playable=True, bbox=box),
                RecognizedCard(card_id=163, name="国民的アイドル+", upgraded=True, playable=True, visual_playable=True, bbox=box),
            ],
            drinks=[],
        )
        acts_logic = [
            LegalAction(choice_id=0, action_type="play_card", description="自然体の魅力+", bbox=box, payload={"card_id": 814, "base_id": 813, "name": "自然体の魅力+"}),
            LegalAction(choice_id=1, action_type="play_card", description="国民的アイドル+", bbox=box, payload={"card_id": 163, "base_id": 162, "name": "国民的アイドル+"}),
            LegalAction(choice_id=9, action_type="end_turn", description="SKIP", bbox=box, payload={"name": "SKIP"}),
        ]
        _, reason_logic, _ = planner_logic.decide_battle_action(obs_logic, acts_logic, tracker=tr_logic)
        assert "[Combo #162/#163->#814 2x]" not in reason_logic

        # 3. Anomaly simulator ignores both Sense and Logic rules
        cfg_anom = cfg_sense.__class__(**{**cfg_sense.__dict__, "plan": "anomaly", "p_idol_id": 1})
        sim_anom = HeadlessBattleSimulator(cfg_anom, db=db, seed=3, record_logs=False)
        sim_anom.turns_remaining = 8
        codec_anom = GakumasActionCodec(cfg_anom, db=db)
        assert codec_anom is not None
        for chk_card in (c778, c153, c814, c163, c619, c776):
            assert sim_anom.should_hold_for_finisher(card_cfg=chk_card, fin_window=2) is False
        for chk_drink in (d25, d22):
            assert sim_anom.should_hold_for_finisher(drink_cfg=chk_drink, fin_window=2) is False
    finally:
        db.close()


def test_card_preferences_unify_base_and_upgraded_and_reroll_all_avoided():
    """Upgraded IDs in avoid/priority/keep/overrides must unify with base IDs, and all-avoided 3-choice rewards must click 再抽選."""
    from demo.core.hif_catalog import ProducePreferences

    db = GakumasDB("gakumas_library.db")
    try:
        prefs = ProducePreferences(
            avoid_card_ids=[71, 69, 95],     # #71 前途洋々+ -> base #70, #69 励まし+ -> #68, #95 キラメキ+ -> #94
            priority_card_ids=[171],          # #171 私がスター+ -> base #170
            keep_card_ids=[127],              # #127 ファンシーチャーム+ -> base #126
            card_score_overrides={71: -500, 129: 777},  # #129 -> base #128
            reroll_early_non_exhaust_mental=False,
        )
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=4, ismcts_simulations=4),
            preferences=prefs,
        )
        base_agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=4, ismcts_simulations=4),
        )

        # 1. Family unification across base <-> upgraded for avoid / priority / keep / numeric overrides
        assert agent._is_avoid_card(70) is True
        assert agent._is_avoid_card(71) is True
        assert 70 in agent._avoid_card_ids and 71 in agent._avoid_card_ids
        assert agent._is_priority_card(170) is True
        assert agent._is_priority_card(171) is True
        assert agent._is_keep_card(126) is True
        assert agent._is_keep_card(127) is True
        assert agent._get_card_score_override(128) == 777
        assert agent._get_card_score_override(129) == 777
        assert (
            agent._score_skill_card(
                card_id=70,
                title="前途洋々",
                desc="パラメータ+8 / 元気+7",
                target_effect="goodImpressionTurns",
                guide=None,
                plan="logic",
            )
            == -20_000
        )
        s_with_ovr = agent._score_skill_card(
            card_id=128,
            title="ワクワクが止まらない",
            desc="",
            target_effect="goodImpressionTurns",
            guide=None,
            plan="logic",
        )
        s_no_ovr = base_agent._score_skill_card(
            card_id=128,
            title="ワクワクが止まらない",
            desc="",
            target_effect="goodImpressionTurns",
            guide=None,
            plan="logic",
        )
        assert s_with_ovr - s_no_ovr == 777

        # 2. W3 3-choice card_reward with ALL 3 slots avoided (no 受け取らない, 再抽選 present)
        #    -> must scan all 3 slots and then click 再抽選 instead of claiming #70 前途洋々!
        box = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)

        def _make_3choice_step(cur_slot: int, cur_cid: int, cur_title: str):
            obs = ScreenObservation(
                scene_type="card_reward",
                plan="logic",
                scenario_id="hif_honsen",
                week=3,
                stamina=33,
                max_stamina=44,
                p_points=250,
                raw_text="スキルカードを選択 再抽選",
            )
            acts = [
                LegalAction(
                    choice_id=0,
                    action_type="pick_option",
                    description=f"确认领取当前选中卡: {cur_title}",
                    bbox=box,
                    payload={"name": "受け取る", "card_id": cur_cid, "title": cur_title, "slot_idx": cur_slot},
                )
            ]
            for idx_s in (0, 1, 2):
                if idx_s == cur_slot:
                    continue
                acts.append(
                    LegalAction(
                        choice_id=len(acts),
                        action_type="pick_option",
                        description=f"预览第{idx_s + 1}张候选卡",
                        bbox=box,
                        payload={"name": f"预览第{idx_s + 1}张候选卡", "slot_idx": idx_s},
                    )
                )
            reroll_id = len(acts)
            acts.append(
                LegalAction(
                    choice_id=reroll_id,
                    action_type="pick_option",
                    description="再抽選 (免费重抽候选卡)",
                    bbox=box,
                    payload={"name": "再抽選", "title": "再抽選"},
                )
            )
            return obs, acts, reroll_id

        # Scan slot 0 (#68) -> requests slot 1; scan slot 1 (#94) -> requests slot 2; scan slot 2 (#70) -> clicks 再抽選
        obs0, acts0, _ = _make_3choice_step(0, 68, "励まし")
        d0 = agent.decide("", acts0, obs=obs0)
        assert acts0[d0.choice_id].payload.get("slot_idx") == 1

        obs1, acts1, _ = _make_3choice_step(1, 94, "キラメキ")
        d1 = agent.decide("", acts1, obs=obs1)
        assert acts1[d1.choice_id].payload.get("slot_idx") == 2

        obs2, acts2, reroll_id2 = _make_3choice_step(2, 70, "前途洋々")
        d2 = agent.decide("", acts2, obs=obs2)
        assert d2.choice_id == reroll_id2, f"Expected 再抽選 for all-avoided cards, got {d2}"
        assert "再抽選" in d2.reason

        # 3. After 3 reroll attempts (or when no 再抽選 exists), falls back to 受け取らない if available
        agent._reward_reroll_attempts = 3
        obs_skip, acts_skip, _ = _make_3choice_step(2, 70, "前途洋々")
        skip_id = len(acts_skip)
        acts_skip.append(
            LegalAction(
                choice_id=skip_id,
                action_type="pick_option",
                description="受け取らない",
                bbox=box,
                payload={"name": "受け取らない", "title": "受け取らない"},
            )
        )
        # Populate seen slots as scanned
        agent._reward_seen_slots = {0: (-20_000, 68, "励まし"), 1: (-20_000, 94, "キラメキ"), 2: (-20_000, 70, "前途洋々")}
        d_skip = agent.decide("", acts_skip, obs=obs_skip)
        assert d_skip.choice_id == skip_id
    finally:
        db.close()


def test_header_meta_hif_stamina_rejects_leading_digit_truncation(monkeypatch):
    """H.I.F _extract_header_meta prioritizes dark_on_light/raw over white and rejects single-frame 40/44->1/44 truncation."""
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        rec = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")
        rec._last_stamina = (40, 44)
        frame = np.full((1600, 720, 3), 240, dtype=np.uint8)
        lines = [{"text": "H.I.F本戦まで 5日", "bbox": [30, 30, 200, 70]}]
        monkeypatch.setattr(rec, "_ocr_bw_banner_number", lambda *_a, **_kw: 5)

        # Case A: white crop truncates 40/44 -> 1/44, dark_on_light/raw reads 40/44 -> must return 40/44
        def _fake_crop_a(_frame, _box, mode="raw", **_kw):
            if mode == "white":
                return "1/44"
            return "40/44"

        monkeypatch.setattr(rec, "_ocr_crop_upscaled", _fake_crop_a)
        meta_a = rec._parse_week_and_stats(lines, frame)
        assert (meta_a["stamina"], meta_a["max_stamina"]) == (40, 44)

        # Case B: transient 1-frame 1/44 across ALL crops when _last_stamina=(40, 44) must preserve 40/44
        monkeypatch.setattr(rec, "_ocr_crop_upscaled", lambda *_a, **_kw: "1/44")
        meta_b = rec._parse_week_and_stats(lines, frame)
        assert (meta_b["stamina"], meta_b["max_stamina"]) == (40, 44)
        assert rec._last_stamina == (40, 44)

        # Case C: Reset clean state with 40/44, then 2 consecutive real 5/44 frames DO commit 5/44 on frame 2
        monkeypatch.setattr(rec, "_ocr_crop_upscaled", _fake_crop_a)
        rec._parse_week_and_stats(lines, frame)
        monkeypatch.setattr(rec, "_ocr_crop_upscaled", lambda *_a, **_kw: "5/44")
        meta_c1 = rec._parse_week_and_stats(lines, frame)
        assert (meta_c1["stamina"], meta_c1["max_stamina"]) == (40, 44)
        meta_c2 = rec._parse_week_and_stats(lines, frame)
        assert (meta_c2["stamina"], meta_c2["max_stamina"]) == (5, 44)
    finally:
        db.close()


def test_runner_popup_fallback_antistall_taps_blank_area(monkeypatch):
    """4 consecutive identical popup generic-fallback clicks ('点击屏幕继续') must trigger blank-area tap instead of looping."""
    db = GakumasDB("gakumas_library.db")
    try:
        frame = np.full((1600, 720, 3), 240, dtype=np.uint8)
        clicks = []

        class _StubPopupCtrl:
            is_connected = True

            def connect(self):
                return True

            def screencap(self):
                return frame

            def click(self, x, y):
                clicks.append((x, y))
                return True

        obs_popup = ScreenObservation(
            scene_type="popup",
            plan="logic",
            scenario_id="hif_honsen",
            week=2,
            stamina=40,
            max_stamina=44,
            options=[
                {
                    "choice_id": 0,
                    "name": "点击屏幕继续",
                    "title": "点击屏幕继续",
                    "description": "过场/对话继续点击",
                    "bbox": BoundingBox(xmin=260, ymin=1248, xmax=460, ymax=1408).model_dump(),
                }
            ],
        )
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=4, ismcts_simulations=4),
        )
        runner = GakumasRunner(
            mode="adb",
            controller=_StubPopupCtrl(),
            agent=agent,
            db=db,
            plan="logic",
            scenario_id="hif_honsen",
            session_mode=False,
        )
        runner.scene_routing_reco = SimpleNamespace(recognize=lambda _fr: obs_popup, recognizer=None)
        monkeypatch.setattr("time.sleep", lambda *_a: None)

        for _ in range(4):
            runner.run_once()

        expected_blank = (int(round(720 * 0.93)), int(round(1600 * 0.26)))
        assert clicks[:3] == [(360, 1328), (360, 1328), (360, 1328)]
        assert clicks[3] == expected_blank
    finally:
        db.close()


def test_all_avoided_no_reroll_fallback_priority_order():
    """When all candidates are excluded/avoided and no 再抽選/skip remains, fallback prefers:
    1-use exhaustible (limit_count==1 / レッスン中1回) > rarity (SSR > SR > R) > type (M > A).
    """
    from demo.core.hif_catalog import ProducePreferences

    db = GakumasDB("gakumas_library.db")
    try:
        # Exclude every card family used across all test cases below
        prefs = ProducePreferences(
            avoid_card_ids=[71, 123, 91, 51, 41, 153, 101, 177, 427, 715],
            reroll_early_non_exhaust_mental=False,
        )
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=4, ismcts_simulations=4),
            preferences=prefs,
        )
        box = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)
        obs = ScreenObservation(
            scene_type="card_reward",
            plan="logic",
            scenario_id="hif_honsen",
            week=4,
            stamina=33,
            max_stamina=44,
            p_points=250,
            raw_text="スキルカードを選択",
        )

        def _pick_direct(cards: list[tuple[int, str]]) -> int:
            acts = [
                LegalAction(
                    choice_id=idx,
                    action_type="pick_option",
                    description=title,
                    bbox=box,
                    payload={"card_id": cid, "name": title, "title": title},
                )
                for idx, (cid, title) in enumerate(cards)
            ]
            dec = agent.decide("", acts, obs=obs)
            return int(acts[dec.choice_id].payload["card_id"])

        # 1. 1-use exhaustible > non-exhaustible (even when exhaustible card has lower rarity!)
        assert _pick_direct([(70, "前途洋々"), (122, "ふれあい"), (90, "ありがとうの言葉")]) == 90
        assert _pick_direct([(70, "前途洋々"), (122, "ふれあい"), (50, "ファーストステップ")]) == 50  # R limit=1 > SR non-exhaust
        assert _pick_direct([(70, "前途洋々"), (40, "ゆるふわおしゃべり")]) == 40                    # R limit=1 A > SR non-exhaust A

        # 2. Rarity: SSR > SR > R when exhaustibility ties
        assert _pick_direct([(100, "ハイテンション"), (50, "ファーストステップ"), (152, "届いて！")]) == 152  # SSR A > SR M > R M
        assert _pick_direct([(50, "ファーストステップ"), (90, "ありがとうの言葉")]) == 90                   # SR A > R M

        # 3. Card Type: Mental (M) > Active (A) when exhaustibility & rarity tie
        assert _pick_direct([(152, "届いて！"), (426, "キセキの魔法"), (176, "手書きのメッセージ")]) == 176  # SSR M > SSR A
        assert _pick_direct([(90, "ありがとうの言葉"), (100, "ハイテンション")]) == 100                      # SR M > SR A
        assert _pick_direct([(70, "前途洋々"), (122, "ふれあい")]) == 122                                  # non-exhaust SR M > SR A

        # 4. End-to-end 3-choice preview flow with 再抽選 exhausted (attempts=3) & no 受け取らない:
        #    scans [slot0=#70(SR A non-ex), slot1=#100(SR M limit=1), slot2=#90(SR A limit=1)],
        #    switches from slot 2 -> slot 1, and claims #100 ハイテンション!
        agent.reset_session()
        agent._reward_reroll_attempts = 3

        def _make_3choice(cur_slot: int, cur_cid: int, cur_title: str):
            acts = [
                LegalAction(
                    choice_id=0,
                    action_type="pick_option",
                    description=f"确认领取当前选中卡: {cur_title}",
                    bbox=box,
                    payload={"name": "受け取る", "card_id": cur_cid, "title": cur_title, "slot_idx": cur_slot},
                )
            ]
            for s in (0, 1, 2):
                if s == cur_slot:
                    continue
                acts.append(
                    LegalAction(
                        choice_id=len(acts),
                        action_type="pick_option",
                        description=f"预览第{s + 1}张候选卡",
                        bbox=box,
                        payload={"name": f"预览第{s + 1}张候选卡", "slot_idx": s},
                    )
                )
            acts.append(
                LegalAction(
                    choice_id=len(acts),
                    action_type="pick_option",
                    description="再抽選",
                    bbox=box,
                    payload={"name": "再抽選", "title": "再抽選"},
                )
            )
            return acts

        acts0 = _make_3choice(0, 70, "前途洋々")
        d0 = agent.decide("", acts0, obs=obs)
        assert acts0[d0.choice_id].payload.get("slot_idx") == 1

        acts1 = _make_3choice(1, 100, "ハイテンション")
        d1 = agent.decide("", acts1, obs=obs)
        assert acts1[d1.choice_id].payload.get("slot_idx") == 2

        acts2 = _make_3choice(2, 90, "ありがとうの言葉")
        d2 = agent.decide("", acts2, obs=obs)
        assert acts2[d2.choice_id].payload.get("slot_idx") == 1  # switches back to #100 ハイテンション!

        acts3 = _make_3choice(1, 100, "ハイテンション")
        d3 = agent.decide("", acts3, obs=obs)
        assert d3.choice_id == 0
        assert acts3[d3.choice_id].payload.get("card_id") == 100
        assert "1-use-exhaust>SSR>SR>R>M>A" in d3.reason
    finally:
        db.close()


def test_ume_136_consumed_genki_3turn_clear_and_854_top_of_deck():
    """Verify p_idol_id=136 full engine loop:
    1) consumedGenki tracking across green card cost, genki=0 clears, and P-item clears;
    2) 夢を創るステッキ (110% default / + 130%) every-3-turn EOT burst+clear+next-turn draw & #800 80% rebound;
    3) #854 証明し続ける一番星 (motivationIncreased -> moveRandomToTopOfDeck + determinize top-deck preservation);
    4) S3 #854 auto-injection, P-item upgrade toggle, #152/#153 release, and 136 guide/scoring alignment.
    """
    import random
    from demo.core.idol_guides import get_p_idol_guide
    from demo.sim.battle_simulator import (
        BattleConfig,
        HeadlessBattleSimulator,
        TurnMultiplierSpec,
        _build_ume_136_hif_p_items,
        _make_card_config,
    )
    from demo.sim.plan_rules import LogicBattleRules

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. consumedGenki + 夢を創るステッキ (110% default) + #800 80% rebound on Turn 4
        cfg_110 = BattleConfig(
            scenario_id="hif_honsen",
            stage_name="ume136_110",
            p_idol_id=136,
            idol_name="[真っ白いページと水彩の主人公] 花海 佑芽",
            plan="logic",
            idol_stats={"vocal": 2000, "dance": 2600, "visual": 1500},
            initial_stamina=40,
            max_stamina=44,
            initial_genki=0,
            total_turns=6,
            turn_schedule=[TurnMultiplierSpec("Da", 1.0)] * 6,
            deck=[_make_card_config(db, cid) for cid in (800,) + (14,) * 10],
            p_items=_build_ume_136_hif_p_items(include_upgraded=False),
        )
        sim = HeadlessBattleSimulator(cfg_110, db=db, seed=7)
        c800_inst = sim._new_instance(_make_card_config(db, 800))
        sim.genki = 10
        sim.discard_pile.extend(sim.hand)
        sim.hand = [c800_inst]
        sim.play_card(0)  # cost-=5 spends 5 genki -> consumed_genki==5; genki+=2 (+0 mot) -> 7; mot+=3 -> 3; auto-ends turn into T2
        assert sim.consumed_genki == 5
        assert sim.genki == 7
        assert sim.buffs["motivation"] == 3
        assert sim.turn == 2

        sim.end_turn()  # T2 -> T3 (genki unchanged, stick does not fire on T2)
        assert sim.turn == 3
        assert sim.p_item_trigger_counts[9136] == 0

        sim.genki = 20
        score_before_t3 = sim.score
        sim.discard_pile.extend(sim.hand)
        sim.hand.clear()
        sim.end_turn()  # T3 end: 夢を創るステッキ fires! +ceil(20*1.10)=22 score, clears 20 genki (consumed=25), T4 draws 4!

        assert sim.turn == 4
        assert sim.p_item_trigger_counts[9136] == 1
        assert sim.score == score_before_t3 + 22
        assert sim.consumed_genki == 25
        # T4 startOfTurn: #800 restores ceil(25*0.8)=20 + motivation(3) = 23 genki!
        assert sim.genki == 23
        # Base TURN_DRAW(3) + next-turn draw(1) = 4 cards drawn
        assert len(sim.hand) == 4

        # 2. Upgraded 夢を創るステッキ+ (130%) scores ceil(20*1.30)=26 on T3 EOT, and #782 reads consumedGenki
        cfg_130 = BattleConfig(
            scenario_id="hif_honsen",
            stage_name="ume136_130",
            p_idol_id=136,
            idol_name="花海 佑芽",
            plan="logic",
            idol_stats={"vocal": 2000, "dance": 2600, "visual": 1500},
            initial_stamina=40,
            max_stamina=44,
            total_turns=4,
            turn_schedule=[TurnMultiplierSpec("Da", 1.0)] * 4,
            deck=[_make_card_config(db, 14)] * 5,
            p_items=_build_ume_136_hif_p_items(include_upgraded=True)[1:],  # only 91361 (+)
        )
        sim_up = HeadlessBattleSimulator(cfg_130, db=db, seed=11)
        sim_up.turn = 3
        sim_up.turns_remaining = 2
        sim_up.genki = 20
        sim_up.score = 0
        sim_up.end_turn()
        assert sim_up.p_item_trigger_counts[91361] == 1
        assert sim_up.score == 26
        assert sim_up.genki == 0
        assert sim_up.consumed_genki == 20
        # On final turn (T4, turns_remaining==1), draw #782 (cardMovedToHand +5 genki) and play it (cost-=1 -> consumed_genki=21, score+=84)
        c782 = sim_up._new_instance(_make_card_config(db, 782))
        sim_up._register_passive_effects_for_card(c782)
        sim_up.hand = []
        sim_up.draw_pile = [c782]
        sim_up.draw_cards(1)
        assert sim_up.genki == 5
        sim_up.score = 0
        sim_up.play_card(0)
        assert sim_up.consumed_genki == 21
        assert sim_up.score == 84  # 21 * 4

        # 3. #854 証明し続ける一番星: motivationIncreased (direct only!) -> moveRandomToTopOfDeck[mental & (deck|discarded)]
        #    and determinize() preserves pinned_top_deck_count!
        cfg_854 = BattleConfig(
            scenario_id="hif_honsen",
            stage_name="ume136_854",
            p_idol_id=136,
            idol_name="花海 佑芽",
            plan="logic",
            idol_stats={"vocal": 2000, "dance": 2600, "visual": 1500},
            initial_stamina=40,
            max_stamina=44,
            total_turns=6,
            turn_schedule=[TurnMultiplierSpec("Da", 1.0)] * 6,
            deck=[_make_card_config(db, cid) for cid in (854, 22, 135, 2, 2, 2, 2)],
            p_items=[],
        )
        sim854 = HeadlessBattleSimulator(cfg_854, db=db, seed=3)
        inst854 = sim854._new_instance(_make_card_config(db, 854))
        inst22 = sim854._new_instance(_make_card_config(db, 22))
        target_135 = sim854._new_instance(_make_card_config(db, 135))
        filler_a = sim854._new_instance(_make_card_config(db, 2))
        filler_b = sim854._new_instance(_make_card_config(db, 2))
        filler_c = sim854._new_instance(_make_card_config(db, 2))
        sim854.hand = [inst854, inst22]
        sim854.draw_pile = [filler_a, filler_b, filler_c]
        sim854.discard_pile = [target_135]  # Only mental card in deck|discarded is #135

        sim854.play_card(0)  # Play #854 (cardUsesRemaining+=1, registers 5T motivationIncreased)
        assert sim854.card_uses_remaining == 1
        sim854.card_uses_remaining = 2  # Keep Turn 1 open after playing #22
        sim854.genki = 2
        sim854.play_card(0)  # Play #22 (direct motivation+=3 -> triggers moveRandomToTopOfDeck)
        assert sim854.draw_pile[0].cfg.card_id == 135
        assert sim854.pinned_top_deck_count == 1
        for k_seed in range(10):
            det = sim854.determinize(random.Random(k_seed))
            assert det.draw_pile[0].cfg.card_id == 135
        sim854.hand.clear()
        sim854.draw_cards(1)
        assert sim854.hand[0].cfg.card_id == 135
        assert sim854.pinned_top_deck_count == 0
        # Indirect motivation trigger with #22 (mental) in discard_pile must NOT trigger if:isDirectEffect
        sim854._fire_phase("motivationIncreased", is_direct=False)
        assert sim854.pinned_top_deck_count == 0
        assert sim854.draw_pile[0].cfg.card_id == 2

        # 4. S3 auto-injection of #854, default 9136 (110%), toggle 91361 (130%), #153 release, & guide/scoring
        cfg_auto = build_live_battle_config(
            db=db,
            deck_card_ids=[800, 137, 159],  # Pre-game loadout omitted #854!
            equipped_p_items=None,
            p_idol_id=136,
            plan="logic",
            scenario_id="hif_honsen",
            total_turns=12,
        )
        assert 854 in [c.card_id for c in cfg_auto.deck]
        stage_ids = [it.item_id for it in cfg_auto.p_items if it.mode == "stage"]
        assert stage_ids == [9136, 429]

        cfg_custom_plus = build_live_battle_config(
            db=db,
            deck_card_ids=[800, 854, 137],
            equipped_p_items=[{"item_id": 91361, "canonical_name": "夢を創るステッキ+", "mode": "stage"}],
            p_idol_id=136,
            plan="logic",
            scenario_id="hif_honsen",
            total_turns=9,
        )
        assert [it.item_id for it in cfg_custom_plus.p_items if it.mode == "stage"] == [91361]

        rules = LogicBattleRules()
        sim_mid = HeadlessBattleSimulator(cfg_auto, db=db, seed=1)
        sim_mid.turns_remaining = 6
        assert rules.should_hold_action(sim_mid, card_cfg=_make_card_config(db, 153), fin_window=2) is False

        cfg_mis = build_live_battle_config(
            db=db,
            deck_card_ids=[920, 922, 137, 159],
            p_idol_id=163,
            plan="logic",
            scenario_id="hif_honsen",
            total_turns=9,
        )
        sim_mis = HeadlessBattleSimulator(cfg_mis, db=db, seed=1)
        sim_mis.turns_remaining = 6
        assert rules.should_hold_action(sim_mis, card_cfg=_make_card_config(db, 153), fin_window=2) is True

        # LocalHybridDecisionAgent does not treat p_idol_id=136 (motivation) as goodImpressionTurns-only GI build
        agent_136 = LocalHybridDecisionAgent(db=db, planner_cfg=LiveBattlePlannerConfig(ppo_episodes=4, ismcts_simulations=4))
        guide_136 = get_p_idol_guide(136)
        assert guide_136 is not None and guide_136.recommended_effect == "motivation"
        s_act_136 = agent_136._score_skill_card(
            card_id=4, title="ポーズの基本+", desc="パラメータ+6\n元気+4", target_effect="motivation", guide=guide_136, plan="logic", p_idol_id=136
        )
        s_act_163 = agent_136._score_skill_card(
            card_id=4, title="ポーズの基本+", desc="パラメータ+6\n元気+4", target_effect="goodImpressionTurns", guide=None, plan="logic", p_idol_id=163
        )
        assert s_act_136 > s_act_163 + 500
    finally:
        db.close()


def test_loadout_scanner_card_customization_direction_inference():
    """HifLoadoutScanner must detect green customization badges, infer exact cust_id+level from text/cost, and merge with manual overrides."""
    import cv2
    from demo.sim.battle_simulator import HeadlessBattleSimulator
    from demo.vision.loadout_scanner import infer_card_customizations_from_scan

    db = GakumasDB("gakumas_library.db")
    scanner = HifLoadoutScanner(db=db)
    try:
        # 1. Multi-level & combo numeric inference on #135 イメトレ+ (custs: 59 max=2, 56 max=2, 12 max=1)
        r135 = db.get_card_by_id(135)
        assert r135 is not None
        assert infer_card_customizations_from_scan(
            db, r135, "元気+15 | やる気+5 | レッスン中1回", badge_detected=True, badge_count=1
        ) == {"levels": {"59": 1}}
        assert infer_card_customizations_from_scan(
            db, r135, "元気+24 | やる気+5 | レッスン中1回", badge_detected=True, badge_count=2
        ) == {"levels": {"59": 2}}
        assert infer_card_customizations_from_scan(
            db, r135, "元気+15 | やる気+6 | レッスン中1回", badge_detected=True, badge_count=2
        ) == {"levels": {"56": 1, "59": 1}}
        assert infer_card_customizations_from_scan(
            db, r135, "元気+11 | やる気+5 | 元気の30%分パラメータ上昇 | レッスン中1回", badge_detected=True, badge_count=1
        ) == {"levels": {"12": 1}}

        # 2. Percentage & typed-cost reduction inference (#153 届いて！+, #427 キセキの魔法+, #299 オトメゴコロ+)
        r153 = db.get_card_by_id(153)
        assert r153 is not None
        assert infer_card_customizations_from_scan(
            db,
            r153,
            "元気が7以上の場合、使用可 | 元気の500%分パラメータ上昇させ、元気を0にする | 重複不可 レッスン中1回",
            badge_detected=True,
            badge_count=2,
        ) == {"levels": {"11": 1, "30": 1}}

        r427 = db.get_card_by_id(427)
        assert r427 is not None
        assert infer_card_customizations_from_scan(
            db,
            r427,
            "体力消費3 | 好印象が10以上の場合、使用可 | 好印象の660%分パラメータ上昇 | 重複不可 レッスン中1回",
            badge_detected=True,
            badge_count=2,
        ) == {"levels": {"6": 1, "30": 1}}

        r299 = db.get_card_by_id(299)
        assert r299 is not None
        assert infer_card_customizations_from_scan(
            db,
            r299,
            "やる気消費2 | 好印象+7 | スキルカード使用数追加+1 | 好印象が10以上の場合、好印象+2 | 重複不可",
            badge_detected=True,
            badge_count=2,
        ) == {"levels": {"34": 1, "80": 1}}

        # 3. Condition & anchor-patch inference (#778 びしっとキメ顔+, #521 夢色リップ+)
        r778 = db.get_card_by_id(778)
        assert r778 is not None
        assert infer_card_customizations_from_scan(
            db,
            r778,
            "好印象が6以上の場合、使用可 | 好印象強化+200%(4ターン) | 以降の4ターンの間、直接効果で好印象が増加後、好印象の130%分パラメータ上昇",
            badge_detected=True,
            badge_count=2,
        ) == {"levels": {"105": 1, "106": 1}}

        r521 = db.get_card_by_id(521)
        assert r521 is not None
        assert infer_card_customizations_from_scan(
            db,
            r521,
            "元気+2 | やる気+5 | パラメータ上昇量増加20% | 好印象が10以上の場合、以降の4ターンの間、ターン開始時、好印象1.1倍 | 元気+4",
            badge_detected=True,
            badge_count=2,
        ) == {"levels": {"27": 1, "88": 1}}

        # 4. Guard against Memory hooks false-positives on #129 & #79 limit removal
        r129 = db.get_card_by_id(129)
        assert r129 is not None
        # Memory hook without customization badge must NOT infer #36
        assert infer_card_customizations_from_scan(
            db,
            r129,
            "レッスン開始時手札に入る | やる気+5 | 以降、メンタルスキルカード使用時、やる気+1 | レッスン中1回 重複不可",
            badge_detected=False,
        ) == {}
        # With green badge(1) and unchanged effect text -> infers #29 (cost-=)
        assert infer_card_customizations_from_scan(
            db,
            r129,
            "やる気+5 | 以降、メンタルスキルカード使用時、やる気+1 | レッスン中1回 重複不可",
            badge_detected=True,
            badge_count=1,
        ) == {"levels": {"29": 1}}
        # With green badge(2) and レッスン開始時手札に入る + やる気+6 -> {36: 1, 56: 1}
        assert infer_card_customizations_from_scan(
            db,
            r129,
            "レッスン開始時手札に入る | やる気+6 | 以降、メンタルスキルカード使用時、やる気+1 | レッスン中1回 重複不可",
            badge_detected=True,
            badge_count=2,
        ) == {"levels": {"36": 1, "56": 1}}

        r79 = db.get_card_by_id(79)
        assert r79 is not None
        assert infer_card_customizations_from_scan(
            db, r79, "パラメータ+13 | 集中が6以上の場合、パラメータ+15", badge_detected=False
        ) == {}
        assert infer_card_customizations_from_scan(
            db, r79, "パラメータ+13 | 集中が6以上の場合、パラメータ+15", badge_detected=True, badge_count=1
        ) == {"levels": {"37": 1}}

        # 5. End-to-end detail-frame badge detection -> parse_detail_frames -> Tracker merge -> Simulator
        doc = np.full((1600, 720, 3), 255, dtype=np.uint8)
        rng = np.random.default_rng(77)
        # Put two card icons in viewport y=406..1372 (rel y=200..315 for #135, rel y=450..565 for #153)
        for by in (606, 856):
            doc[by : by + 115, 55:170] = rng.integers(25, 110, size=(115, 115, 3), dtype=np.uint8)
            doc[by + 10 : by + 100, 200:620] = rng.integers(190, 240, size=(90, 420, 3), dtype=np.uint8)
        # Render a crisp bright-green circular customization badge with white "1" on #135's icon (y=606..721, x=55..170)
        cv2.circle(doc, (112, 686), 14, (45, 200, 60), thickness=-1)
        cv2.line(doc, (112, 678), (112, 694), (255, 255, 255), thickness=3)

        def _fake_ja_cust(arr: np.ndarray, *a, **kw):
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
                        {"text": "スキルカード", "bbox": [24, 150, 160, 175]},
                        {"text": "イメトレ+", "bbox": [200, 206, 420, 236]},
                        {"text": "元気+15 / やる気+5 / レッスン中1回", "bbox": [200, 246, 620, 276]},
                        {"text": "届いて！+", "bbox": [200, 456, 420, 486]},
                        {"text": "体力消費1 / 元気が7以上の場合、使用可 / 元気の500%分パラメータ上昇させ、元気を0にする / 重複不可 レッスン中1回", "bbox": [200, 496, 640, 530]},
                    ],
                }
            return {"text": "", "lines": []}

        def _fake_en_cust(arr: np.ndarray, *a, **kw):
            aw, ah = arr.shape[1], arr.shape[0]
            if ah >= 90 and aw >= 150:
                return {"text": "10+ 1 +10", "lines": []}
            return {"text": "2000", "lines": []}

        scanner.ocr = SimpleNamespace(recognize_array=_fake_ja_cust)
        scanner.reco.en_ocr = SimpleNamespace(recognize_array=_fake_en_cust)
        res = scanner.parse_detail_frames([doc], plan="logic")
        assert 135 in res.deck_card_ids
        assert 153 in res.deck_card_ids
        assert res.card_customizations.get(135) == {"levels": {"59": 1}}
        assert res.card_customizations.get(153) == {"levels": {"11": 1}}

        # Tracker merges scanned customizations with WebUI manual override taking priority for #153
        tracker = GameStateTracker(plan="logic", p_idol_id=163, scenario_id="hif_honsen")
        tracker.set_initial_loadout(
            p_idol_id=res.p_idol_id,
            plan=res.plan,
            deck_ids=res.deck_card_ids,
            p_items=res.p_items,
            card_memory_hooks=res.card_memory_hooks,
            card_customizations=res.card_customizations,
            db=db,
        )
        tracker.merge_manual_memory_hooks(card_customizations={153: {"levels": {"30": 1}}}, db=db)
        assert tracker.card_customizations[135] == {"levels": {"59": 1}}
        assert tracker.card_customizations[153] == {"levels": {"30": 1}}

        cfg = build_live_battle_config(
            db=db,
            deck_card_ids=[135, 153, 137],
            p_idol_id=163,
            plan="logic",
            card_customizations=tracker.card_customizations,
        )
        sim = HeadlessBattleSimulator(cfg, db=db, seed=5)
        c135 = next(c for c in sim._all_instances.values() if c.cfg.card_id == 135)
        sim.hand = [c135]
        sim.genki = 0
        sim.stamina = 20
        sim.play_card(0)
        assert sim.genki == 15  # Customized +4 from #59 Lv1 on top of base +11
    finally:
        scanner.close()
        db.close()


def test_support_stick_429_codec_and_unmapped_basic_card_guard():
    """#429 basic fillers are natively indexed by GakumasActionCodec, blocked from unmapped hijack, and excluded from permanent deck."""
    from demo.sim.rl_env import GakumasActionCodec
    from demo.sim.battle_simulator import BattleConfig, TurnMultiplierSpec, _make_card_config

    db = GakumasDB("gakumas_library.db")
    try:
        # 1. #429 H.I.F応援棒 registers logic N basic fillers (including #652 アイコンタクトの基本) in GakumasActionCodec
        cfg_r2 = build_live_battle_config(
            db=db,
            deck_card_ids=[800, 854, 138, 520, 137],
            p_idol_id=136,
            plan="logic",
            scenario_id="hif_honsen",
            total_turns=12,
        )
        codec = GakumasActionCodec(cfg_r2, db=db)
        assert (652, False, False) in codec.card_key_to_action

        # 2. Defensive unmapped guard: even if a codec lacks #652, unmapped fallback NEVER hijacks real plays/drinks
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_episodes=6, ismcts_simulations=6, policy_cache_dir=None),
        )
        cfg_no_429 = BattleConfig(
            scenario_id="hif_honsen",
            stage_name="guard_652",
            p_idol_id=136,
            idol_name="花海 佑芽",
            plan="logic",
            idol_stats={"vocal": 2000, "dance": 2500, "visual": 1500},
            initial_stamina=40,
            max_stamina=44,
            total_turns=9,
            turn_schedule=[TurnMultiplierSpec("Da", 1.0)] * 9,
            deck=[_make_card_config(db, cid) for cid in (800, 138, 520, 137, 139, 159)],
            p_items=[],
        )
        assert (652, False, False) not in GakumasActionCodec(cfg_no_429, db=db).card_key_to_action

        box = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)
        obs = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=7,
            turn=3,
            turns_remaining=7,
            stamina=40,
            max_stamina=44,
            genki=6,
            score=500,
            target_score=5000,
            buffs={"goodImpressionTurns": 8, "motivation": 12},
            idol_stats={"vocal": 2000, "dance": 2500, "visual": 1500},
            cards=[
                RecognizedCard(card_id=520, name="夢色リップ", type="active", upgraded=False, playable=True, visual_playable=True, bbox=box),
                RecognizedCard(card_id=138, name="ゆめみごこち", type="mental", upgraded=False, playable=True, visual_playable=True, bbox=box),
                RecognizedCard(card_id=652, name="アイコンタクトの基本", type="active", upgraded=False, playable=True, visual_playable=True, bbox=box),
            ],
            drinks=[],
        )
        tracker = GameStateTracker(plan="logic", p_idol_id=136, scenario_id="hif_honsen")
        tracker.set_initial_loadout(
            p_idol_id=136,
            plan="logic",
            deck_ids=[800, 138, 520, 137, 139, 159],
            p_items=[],
            db=db,
        )
        legal_acts = [
            LegalAction(choice_id=0, action_type="play_card", description="打出 #520 夢色リップ", bbox=box, payload={"card_id": 520, "name": "夢色リップ"}),
            LegalAction(choice_id=1, action_type="play_card", description="打出 #138 ゆめみごこち", bbox=box, payload={"card_id": 138, "name": "ゆめみごこち"}),
            LegalAction(choice_id=2, action_type="play_card", description="打出 #652 アイコンタクトの基本", bbox=box, payload={"card_id": 652, "name": "アイコンタクトの基本"}),
            LegalAction(choice_id=3, action_type="end_turn", description="SKIP", bbox=box, payload={}),
        ]
        # Remove #652 from planner codec to explicitly exercise the unmapped fallback guard
        cid_chosen, reason, _dbg = planner.decide_battle_action(obs, legal_acts, tracker=tracker)
        assert int((legal_acts[cid_chosen].payload or {}).get("card_id") or 0) != 652, f"Unmapped #652 leaked: {reason}"

        # 3. Tracker does NOT permanently append temporary #652 filler to produce_deck
        tracker.in_battle = True
        tracker.sync_observed_hand([SimpleNamespace(card_id=520), SimpleNamespace(card_id=652)], db=db, turn=3, turns_remaining=7)
        assert 652 not in tracker.produce_deck
        tracker.record_card_play(652, db=db)
        assert 652 not in tracker.produce_deck
    finally:
        db.close()


def test_logic_senobi_delayed_burst_and_drink25_rules_isolated():
    """LogicBattleRules holds #771/#772+#25 until rem<=6, enforces #25->#772 in rem 5..6, avoids dead #772 at rem<=4, and stays isolated from Sense/Anomaly."""
    from demo.sim.battle_simulator import BattleConfig, HeadlessBattleSimulator, TurnMultiplierSpec, _make_card_config, _make_drink_config
    from demo.sim.plan_rules import AnomalyBattleRules, LogicBattleRules, SenseBattleRules

    db = GakumasDB("gakumas_library.db")
    try:
        logic_rules = LogicBattleRules()
        sense_rules = SenseBattleRules()
        anom_rules = AnomalyBattleRules()
        cfg = BattleConfig(
            scenario_id="hif_honsen",
            stage_name="senobi_rules",
            p_idol_id=136,
            idol_name="花海 佑芽",
            plan="logic",
            idol_stats={"vocal": 2000, "dance": 2500, "visual": 1500},
            initial_stamina=40,
            max_stamina=44,
            total_turns=12,
            turn_schedule=[TurnMultiplierSpec("Da", 1.0)] * 12,
            deck=[_make_card_config(db, cid) for cid in (772, 45, 521, 139, 137, 800, 854)],
            drinks=[_make_drink_config(db, 25)],
            p_items=[],
        )
        sim = HeadlessBattleSimulator(cfg, db=db, seed=9)
        c772 = next(c for c in sim._all_instances.values() if c.cfg.card_id == 772)
        c45 = next(c for c in sim._all_instances.values() if c.cfg.card_id == 45)
        c521 = next(c for c in sim._all_instances.values() if c.cfg.card_id == 521)
        c139 = next(c for c in sim._all_instances.values() if c.cfg.card_id == 139)
        d25 = sim.available_drinks[0]

        # Isolation check on Sense & Anomaly
        assert logic_rules.is_senobi_delayed_burst_card(c772.cfg) is True
        assert sense_rules.is_senobi_delayed_burst_card(c772.cfg) is False
        assert anom_rules.is_senobi_delayed_burst_card(c772.cfg) is False
        assert sense_rules.can_drink25_combo_senobi_now(sim, d25) is False
        assert anom_rules.can_drink25_combo_senobi_now(sim, d25) is False

        # 1. Early turn (rem == 7): both #772 and #25 must be held
        sim.turns_remaining = 7
        sim.stamina = 35
        sim.hand = [c772, c521, c139]
        assert logic_rules.should_hold_action(sim, card_cfg=c772.cfg) is True
        assert logic_rules.should_hold_action(sim, drink_cfg=d25) is True

        # 2. Landing window (rem == 6 and rem == 5): #25 -> #772 combo is enabled and naked #772 is stripped
        for rem in (6, 5):
            sim.turns_remaining = rem
            sim.double_card_effect_cards = 0
            sim.triggers = [tr for tr in sim.triggers if not getattr(tr, "actions", None)]
            assert logic_rules.should_hold_action(sim, card_cfg=c772.cfg) is False
            assert logic_rules.can_drink25_combo_senobi_now(sim, d25) is True
            amap = {0: ("card", 0), 1: ("card", 1), 2: ("drink", 0)}
            narrowed, early = logic_rules.narrow_armed_or_combo_action_map(sim, amap)
            assert early is True and narrowed is not None
            assert 2 in narrowed and 0 not in narrowed

        # 3. Dead window (rem <= 4): armed #25 avoids dead #772 when active #45 is in hand
        sim.turns_remaining = 4
        sim.hand = [c772, c45, c521]
        sim.use_drink(0)  # Arms active retrigger for #25
        assert sim.has_armed_active_retrigger(same_turn_only=True)
        assert logic_rules.should_forbid_double_on_card(sim, c772.cfg, has_other_double_target=True) is True
        amap_dead = {0: ("card", 0), 1: ("card", 1)}
        narrowed_d, _ = logic_rules.narrow_armed_or_combo_action_map(sim, amap_dead)
        assert narrowed_d is not None and 1 in narrowed_d and 0 not in narrowed_d

        # 4. Planner end-to-end picks #25 before #772 at rem=6
        planner = LiveBattlePlanner(
            db=db,
            cfg=LiveBattlePlannerConfig(ppo_episodes=4, ismcts_simulations=4, policy_cache_dir=None),
        )
        box = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)
        obs6 = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=9,
            turn=7,
            turns_remaining=6,
            stamina=32,
            max_stamina=44,
            genki=80,
            score=50000,
            target_score=200000,
            buffs={"goodImpressionTurns": 30, "motivation": 50},
            idol_stats={"vocal": 2000, "dance": 2500, "visual": 1500},
            cards=[
                RecognizedCard(card_id=772, name="せのびの魔法+", type="active", upgraded=True, playable=True, visual_playable=True, bbox=box),
                RecognizedCard(card_id=521, name="夢色リップ+", type="mental", upgraded=True, playable=True, visual_playable=True, bbox=box),
            ],
            drinks=[
                RecognizedDrink(drink_id=25, name="特製ハツボシエキス", playable=True, bbox=box),
            ],
        )
        tracker = GameStateTracker(plan="logic", p_idol_id=136, scenario_id="hif_honsen")
        tracker.set_initial_loadout(
            p_idol_id=136,
            plan="logic",
            deck_ids=[772, 45, 521, 137, 800, 854],
            p_items=[],
            db=db,
        )
        acts6 = [
            LegalAction(choice_id=0, action_type="play_card", description="打出 #772 せのびの魔法+", bbox=box, payload={"card_id": 772, "name": "せのびの魔法+"}),
            LegalAction(choice_id=1, action_type="play_card", description="打出 #521 夢色リップ+", bbox=box, payload={"card_id": 521, "name": "夢色リップ+"}),
            LegalAction(choice_id=2, action_type="use_drink", description="使用 #25 特製ハツボシエキス", bbox=box, payload={"drink_id": 25, "name": "特製ハツボシエキス"}),
            LegalAction(choice_id=3, action_type="end_turn", description="SKIP", bbox=box, payload={}),
        ]
        chosen_id, reason6, _ = planner.decide_battle_action(obs6, acts6, tracker=tracker)
        assert chosen_id == 2, f"Expected drink #25 combo before #772 at rem=6, got {chosen_id}: {reason6}"
    finally:
        db.close()


def test_shop_skill_card_preview_routing_and_priority_card_purchase(monkeypatch):
    """Shop card previews (showing 強化確認/カスタマイズ確認 alongside 交換する) must route to `shop`, and priority_card_ids buy before drinks."""
    import cv2
    from demo.core.hif_catalog import ProducePreferences
    from demo.vision.live_adb_recognizer import LiveScreenRecognizer

    db = GakumasDB("gakumas_library.db")
    try:
        reco = LiveScreenRecognizer(db=db, plan="logic", scenario_id="hif_honsen")

        # 1a. Verify on captured live screenshots from 136 run if present on disk
        for live_img in (
            Path("debug_live_logic_136_20261126_163438/card_reward_W8_step210.png"),
            Path("debug_live_logic_136_20261126_163438/card_reward_W6_step131.png"),
        ):
            if live_img.exists():
                bgr = cv2.imread(str(live_img))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                obs_live = reco.recognize(rgb, plan="logic")
                assert obs_live.scene_type == "shop", f"{live_img.name} misrouted to {obs_live.scene_type}"
                ex_opt = next((o for o in obs_live.options if o.get("name") == "交換する"), None)
                assert ex_opt is not None and ex_opt.get("card_id") == 170

        # 1b. Synthetic CI-safe reproduction of shop skill card preview containing 強化確認 & カスタマイズ確認
        h, w = 1600, 720
        frame = np.full((h, w, 3), 245, dtype=np.uint8)
        frame[int(h * 0.72) : int(h * 0.80), int(w * 0.30) : int(w * 0.56)] = (235, 110, 40)
        syn_lines = [
            {"text": "インターバル", "bbox": [40, 30, 180, 65]},
            {"text": "Pポイント 330", "bbox": [310, 80, 430, 110]},
            {"text": "私がスター", "bbox": [190, 340, 340, 375]},
            {"text": "強化確認", "bbox": [65, 480, 155, 508]},
            {"text": "カスタマイズ確認", "bbox": [65, 520, 180, 548]},
            {"text": "好印象消費2 / ターン追加+1", "bbox": [190, 390, 520, 425]},
            {"text": "Pポイントで利用するものを選んでください", "bbox": [140, 260, 580, 290]},
            {"text": "50", "bbox": [95, 1020, 145, 1045]},
            {"text": "100", "bbox": [410, 1020, 470, 1045]},
            {"text": "リフレッシュ", "bbox": [40, 1160, 180, 1190]},
            {"text": "交換する", "bbox": [290, 1160, 420, 1195]},
            {"text": "終了", "bbox": [600, 1160, 680, 1195]},
        ]
        monkeypatch.setattr(reco.ocr, "recognize_array", lambda *_a, **_kw: {"text": " ".join(l["text"] for l in syn_lines), "lines": syn_lines})
        obs_syn = reco.recognize(frame, plan="logic")
        assert obs_syn.scene_type == "shop"

        # 2. LocalHybridDecisionAgent prioritizes buying priority card #170 BEFORE drinks (#25)
        agent = LocalHybridDecisionAgent(
            db=db,
            planner_cfg=LiveBattlePlannerConfig(ppo_episodes=4, ismcts_simulations=4),
            preferences=ProducePreferences(priority_card_ids=[171]),
        )
        tracker = GameStateTracker(plan="logic", p_idol_id=136, scenario_id="hif_honsen")
        tracker.set_initial_loadout(p_idol_id=136, plan="logic", deck_ids=[800, 129, 139], p_items=[], db=db)
        agent.tracker = tracker

        box = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)
        shop_obs = ScreenObservation(
            scene_type="shop",
            plan="logic",
            scenario_id="hif_honsen",
            week=8,
            total_weeks=9,
            p_points=330,
            drinks=[],
        )
        # Case A: priority card #170 is previewed -> immediately click 交換する
        acts_card_selected = [
            LegalAction(choice_id=0, action_type="pick_option", description="交換する 《私がスター》 (-50P)", bbox=box, payload={"name": "交換する", "card_id": 170, "price": 50}),
            LegalAction(choice_id=1, action_type="pick_option", description="Pドリンク3 《特製ハツボシエキス》 (100P)", bbox=box, payload={"name": "Pドリンク3", "shop_slot_key": "Pドリンク:3", "drink_id": 25, "price": 100}),
            LegalAction(choice_id=2, action_type="pick_option", description="終了", bbox=box, payload={"name": "終了", "shop_slot_key": "終了"}),
        ]
        dec_buy_card = agent.decide("", acts_card_selected, obs=shop_obs)
        assert dec_buy_card.choice_id == 0

        # Case B: drink #25 is currently previewed, while #170 slot is available -> switch to #170 slot FIRST!
        agent._shop_attempted_card_ids.clear()
        agent._shop_expecting_priority_card_confirm = None
        acts_drink_selected_card_avail = [
            LegalAction(choice_id=0, action_type="pick_option", description="交換する 《特製ハツボシエキス》 (-100P)", bbox=box, payload={"name": "交換する", "drink_id": 25, "price": 100}),
            LegalAction(choice_id=1, action_type="pick_option", description="スキルカード5 《私がスター》 (50P)", bbox=box, payload={"name": "スキルカード5", "shop_slot_key": "スキルカード:5", "card_id": 170, "price": 50}),
            LegalAction(choice_id=2, action_type="pick_option", description="終了", bbox=box, payload={"name": "終了", "shop_slot_key": "終了"}),
        ]
        dec_pick_card_first = agent.decide("", acts_drink_selected_card_avail, obs=shop_obs)
        assert dec_pick_card_first.choice_id == 1
    finally:
        db.close()












