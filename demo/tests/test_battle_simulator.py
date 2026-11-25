"""Tests for headless Gakumas battle simulator configuration, P-items, memory limits, and execution."""

from __future__ import annotations

from collections import Counter
import random

from demo.core.db import GakumasDB
from demo.sim.battle_simulator import (
    ActiveTrigger,
    BattleConfig,
    CardConfig,
    HeadlessBattleSimulator,
    TurnMultiplierSpec,
    _build_hif_p_items,
    _make_card_config,
    build_last_run_hif_preset,
)


def test_last_run_preset_loading_and_opening_hand():
    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        assert cfg.p_idol_id == 163
        assert cfg.plan == "logic"
        assert cfg.idol_stats == {"vocal": 2730, "dance": 1572, "visual": 1849}
        assert cfg.initial_stamina == 33 and cfg.max_stamina == 44
        assert cfg.initial_good_impression == 6 and cfg.initial_motivation == 6
        assert cfg.total_turns == 12
        assert cfg.randomize_turn_schedule is True
        assert len(cfg.turn_schedule) == 12
        assert [d.drink_id for d in cfg.drinks] == [13, 25, 25, 25]

        # 22 cards total: 21 upgraded (+) skill cards + 1 #23 眠気
        assert len(cfg.deck) == 22
        skill_cards = [c for c in cfg.deck if c.card_type != "trouble"]
        trouble_cards = [c for c in cfg.deck if c.card_type == "trouble"]
        assert len(skill_cards) == 21
        assert len(trouble_cards) == 1 and trouble_cards[0].card_id == 23
        assert all(c.upgraded for c in skill_cards)

        # Only #159 (innate), #129, and #127 enter opening hand
        init_ids = [c.card_id for c in cfg.deck if c.force_initial_hand]
        assert sorted(init_ids) == [127, 129, 159]

        # Exactly 1 copy of #137 has memory extra-action, and 3 cards have memory trouble-exile
        copies_137 = [c for c in cfg.deck if c.card_id == 137]
        assert len(copies_137) == 2
        assert sum(1 for c in copies_137 if c.free_extra_use_first_play) == 1
        exile_ids = sorted(c.card_id for c in cfg.deck if c.exiles_trouble_on_play)
        assert exile_ids == [129, 297, 299]

        # All 8 equipped P-items loaded (3 stage + 5 produce)
        assert len(cfg.p_items) == 8
        stage_ids = [it.item_id for it in cfg.p_items if it.mode == "stage"]
        produce_ids = [it.item_id for it in cfg.p_items if it.mode == "produce"]
        assert stage_ids == [509, 404, 429]
        assert produce_ids == [436, 447, 61, 510, 487]

        sim = HeadlessBattleSimulator(cfg, db=db, seed=2026)
        sched_counts = Counter((s.attribute, s.multiplier) for s in sim.turn_schedule)
        assert sched_counts == {("Vo", 40.0): 6, ("Vi", 35.0): 3, ("Da", 20.0): 3}
        assert sim.turn_schedule[11] == TurnMultiplierSpec("Vo", 40.0)
        assert sim.turn_schedule[9].attribute != "Vo"
        assert sim.turn_schedule[10].attribute != "Vo"
        assert [c.cfg.card_id for c in sim.hand] == [159, 129, 127]
        assert len(sim.draw_pile) == 19
        assert len(sim.discard_pile) == 0
        assert len(sim.exhaust_pile) == 0
        assert sim.p_item_trigger_counts[429] == 0

        # #429 H.I.F応援棒（黄） pads <22-card decks up to 22 with random basic N cards
        short_cfg = BattleConfig(
            scenario_id="test_pad",
            stage_name="pad_check",
            p_idol_id=163,
            idol_name="test",
            plan="logic",
            idol_stats={"vocal": 2000, "dance": 1500, "visual": 1500},
            initial_stamina=33,
            max_stamina=44,
            initial_good_impression=6,
            initial_motivation=6,
            total_turns=3,
            turn_schedule=[TurnMultiplierSpec("Vo", 1.0)] * 3,
            deck=list(cfg.deck[:20]),
            p_items=_build_hif_p_items(include_cheering_stick=True),
        )
        short_sim = HeadlessBattleSimulator(short_cfg, db=db, seed=7)
        total_padded = len(short_sim.hand) + len(short_sim.draw_pile) + len(short_sim.discard_pile)
        assert total_padded == 22
        assert short_sim.p_item_trigger_counts[429] == 1


def test_live_turn1_saki_bento_and_eot_ceil_alignment():
    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        cfg.randomize_turn_schedule = False
        cfg.turn_schedule = [TurnMultiplierSpec("Vo", 40.0)] * 12
        sim = HeadlessBattleSimulator(cfg, db=db, seed=1)

        # Put an un-upgraded #136 at top of draw_pile to verify #159 Turn 2 upgradeHand
        sim.draw_pile[0].cfg = _make_card_config(db, 136)

        # Play 1st card of Turn 1: #159 アイドル魂+ (cost 2 -> HP 33->31, G=12, uses stays 1)
        assert sim.hand[0].cfg.card_id == 159
        sim.play_card(0)
        assert sim.stamina == 31
        assert sim.genki == 12
        assert sim.card_uses_remaining == 1
        assert sim.p_item_trigger_counts[404] == 0

        # Play 2nd card of Turn 1: #129 ワクワクが止まらない+ (cost 3 -> G=9, Mot=11, exiles #23 眠気)
        # Triggers #404 咲季の完全食・改 (+2 indirect GI -> 8, +25% perm GI effect buff)
        # Since uses reaches 0, Turn 1 settles EOT automatically: ceil(8 * 1.25 * 40.0) == 400!
        assert sim.hand[0].cfg.card_id == 129
        sim.play_card(0)
        assert sim.genki == 9
        assert sim.buffs["motivation"] == 11
        assert sim.p_item_trigger_counts[404] == 1
        assert sim.p_item_gi_counters[509] == 0  # #404 indirect GI does not advance #509
        assert abs(sim.perm_gi_effect_buff - 0.25) < 1e-6
        assert any(c.cfg.card_id == 23 for c in sim.exhaust_pile)
        assert sim.score == 400
        assert sim.buffs["goodImpressionTurns"] == 7
        assert sim.turn == 2
        # #159 Turn 2 after-draw upgradeHand upgraded #136 -> #137
        assert any(inst.cfg.card_id == 137 for inst in sim.hand)


def test_once_per_memory_exile_and_extra_action_limits():
    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        sim = HeadlessBattleSimulator(cfg, db=db, seed=99)
        sim.hand.clear()
        sim.draw_pile.clear()
        sim.discard_pile.clear()
        sim.exhaust_pile.clear()
        sim.triggers.clear()

        def _spawn(cid: int, zone: list, *, exile_t: bool = False, extra_1st: bool = False):
            c = _make_card_config(
                db,
                cid,
                exiles_trouble_on_play=exile_t,
                free_extra_use_first_play=extra_1st,
            )
            inst = sim._new_instance(c)
            sim._register_passive_effects_for_card(inst)
            zone.append(inst)
            return inst

        otome = _spawn(299, sim.hand, exile_t=True)
        nemuke_a = _spawn(23, sim.draw_pile)
        nemuke_b = _spawn(23, sim.draw_pile)
        yaruki_mem = _spawn(137, sim.hand, extra_1st=True)
        yaruki_plain = _spawn(137, sim.hand, extra_1st=False)

        sim.buffs["motivation"] = 10
        sim.buffs["goodImpressionTurns"] = 6
        sim.genki = 20
        sim.card_uses_remaining = 1

        # 1st play of repeatable #299 -> exiles ONLY 1 眠気
        sim.play_card(0)
        assert sim.card_uses_remaining == 1
        assert len(sim.exhaust_pile) == 1 and sim.exhaust_pile[0].uid in (nemuke_a.uid, nemuke_b.uid)
        assert len(sim.draw_pile) == 1
        assert len(sim.memory_exile_used_uids) == 1

        # Simulate redrawing & playing the SAME #299 instance a 2nd time -> must NOT exile 2nd 眠気
        sim.discard_pile.remove(otome)
        sim.hand.insert(0, otome)
        sim.play_card(0)
        assert len(sim.exhaust_pile) == 1
        assert len(sim.draw_pile) == 1  # 2nd 眠気 stays untouched!
        assert sim.p_item_trigger_counts[404] == 1

        # Turn 2: #404 limit:1 must NOT trigger again on 2nd card of Turn 2
        sim.discard_pile.remove(otome)
        sim.draw_pile.clear()
        sim.hand = [yaruki_mem, yaruki_plain]
        sim.end_turn()  # advances to Turn 2
        sim.hand = [yaruki_mem, yaruki_plain]
        sim.card_uses_remaining = 1

        # Play memory copy of #137 (1st play): free extra use (card_uses_remaining stays 1, does not auto end_turn)
        sim.play_card(0)
        assert sim.turn == 2
        assert sim.card_uses_remaining == 1
        assert len(sim.memory_extra_used_uids) == 1

        # Re-play same #137 uid -> no longer free, consumes card_uses_remaining (1 -> 0) and auto ends turn -> Turn 3
        sim.exhaust_pile.remove(yaruki_mem)
        sim.hand.insert(0, yaruki_mem)
        sim.play_card(0)
        assert sim.turn == 3
        assert sim.p_item_trigger_counts[404] == 1  # Still capped at 1

        # Plain #137 copy has no memory free use -> 1st play consumes card_uses_remaining (1 -> 0) -> Turn 4
        sim.hand = [yaruki_plain]
        sim.card_uses_remaining = 1
        sim.play_card(0)
        assert sim.turn == 4


def test_memory_hooks_replay_p_item_and_double_active_combo():
    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        sim = HeadlessBattleSimulator(cfg, db=db, seed=42)
        sim.hand.clear()
        sim.draw_pile.clear()
        sim.discard_pile.clear()
        sim.exhaust_pile.clear()
        sim.triggers.clear()

        def _add_inst(cid: int, zone: list, *, exile_t: bool = False, extra_1st: bool = False):
            c = _make_card_config(
                db,
                cid,
                exiles_trouble_on_play=exile_t,
                free_extra_use_first_play=extra_1st,
            )
            inst = sim._new_instance(c)
            sim._register_passive_effects_for_card(inst)
            zone.append(inst)
            return inst

        _add_inst(137, sim.hand, extra_1st=True)
        _add_inst(920, sim.hand)
        _add_inst(297, sim.hand, exile_t=True)
        _add_inst(23, sim.draw_pile)
        _add_inst(778, sim.hand)

        for _ in range(8):
            _add_inst(179, sim.exhaust_pile)

        sim.card_uses_remaining = 1
        sim.buffs["goodImpressionTurns"] = 6
        sim.buffs["motivation"] = 10
        sim.buffs["halfCostTurns"] = 3
        sim.genki = 30
        sim.stamina = 33

        # 1. #137 やる気は満点+ (memory free use) -> +5 GI (direct #1), uses stays 1
        sim.play_card(0)
        assert sim.card_uses_remaining == 1
        assert sim.p_item_gi_counters[509] == 1

        # 2. #920 ありのままのわたし+ -> +50% GI effect buff (2nd card of turn -> #404 +25% perm + +2 indirect GI)
        # Keep uses=1 for continuation via extra action below
        sim.card_uses_remaining = 2
        sim.play_card(0)
        assert sim.p_item_trigger_counts[404] == 1
        assert sim.p_item_gi_counters[509] == 1  # indirect #404 didn't increment #509
        assert abs(sim.total_gi_effect_buff() - 0.75) < 1e-6  # 0.50 + 0.25

        # 3. #297 止められない想い+ under halfCostTurns=3 -> stamina cost ceil(4/2)==2, HP 33->31!
        # Exiles #23 眠気 & grants +5 GI (direct #2) & +1 use
        sim.play_card(0)
        assert sim.stamina == 31
        assert any(c.cfg.card_id == 23 for c in sim.exhaust_pile)
        assert sim.p_item_gi_counters[509] == 2

        # 4. Drink #25 特製ハツボシエキス then #778 びしっとキメ顔+ -> doubles +200% to +400%
        sim.card_uses_remaining = 3
        idx_25 = next(i for i, d in enumerate(sim.available_drinks) if d.drink_id == 25)
        sim.use_drink(idx_25)
        sim.play_card(0)  # #778
        assert abs(sim.total_gi_effect_buff() - 4.75) < 1e-6

        # 5. Play #299 (+5,+2 GI -> direct #3 & #4 -> triggers #509 +10% & EOT+1, AND triggers #920 Replay [+50% GI effect buff]!)
        _add_inst(299, sim.hand)
        _add_inst(922, sim.hand)
        sc_before = sim.score
        sim.play_card(0)  # #299
        sim.play_card(0)  # #922 (>=11 exhaust burst & 3rd direct GI post-exhaust -> #920 Replay!)
        assert sim.p_item_trigger_counts[509] == 1
        assert sim.stamina >= 29  # #509+ has 0 stamina cost on trigger
        assert sim.score > sc_before
        # 4.75 + 0.10 (#509) + 0.50 (#920 replay) = 5.35
        assert abs(sim.total_gi_effect_buff() - 5.35) < 1e-6


def test_two_random_runs_complete_cleanly_with_conservation():
    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        for seed in (2026, 2027):
            sim = HeadlessBattleSimulator(cfg, db=db, seed=seed)
            res = sim.run_random_episode()
            assert res.turns_played >= 12
            assert sim.turns_remaining == 0
            assert res.final_score > 0
            assert res.exhaust_count > 0
            generated_nemuke = sum("気まぐれハート" in p for p in res.cards_played)
            total_cards = sum(res.zone_counts.values())
            assert total_cards == len(cfg.deck) + generated_nemuke
            assert res.p_item_activations["咲季の完全食・改"] <= 1
            assert res.p_item_activations["H.I.F応援棒（黄）"] == 0
            assert sum("除外:" in m for m in res.memory_triggers) <= 3
            assert sum("再動" in m for m in res.memory_triggers) <= 1


def _new_clean_sim(
    db: GakumasDB,
    *,
    seed: int = 7,
    enable_404: bool = False,
    enable_509: bool = False,
) -> HeadlessBattleSimulator:
    """Create a clean H.I.F simulator with empty zones/triggers for targeted unit checks."""
    cfg = build_last_run_hif_preset(db=db, stage="round2")
    cfg.randomize_turn_schedule = False
    cfg.turn_schedule = [TurnMultiplierSpec("Vo", 40.0)] * 12
    kept_items = []
    for it in cfg.p_items:
        if it.item_id == 404 and not enable_404:
            continue
        if it.item_id == 509 and not enable_509:
            continue
        kept_items.append(it)
    cfg.p_items = kept_items
    sim = HeadlessBattleSimulator(cfg, db=db, seed=seed)
    sim.hand.clear()
    sim.draw_pile.clear()
    sim.discard_pile.clear()
    sim.exhaust_pile.clear()
    sim.triggers.clear()
    sim.p_item_gi_counters = {it.item_id: 0 for it in cfg.p_items}
    sim.p_item_trigger_counts = {it.item_id: 0 for it in cfg.p_items}
    sim.card_effect_counters.clear()
    sim.card_turn_counters.clear()
    sim.fresh_turn_stat_buffs.clear()
    sim.timed_gi_effect_buffs.clear()
    sim.timed_gi_gain_buffs.clear()
    sim.timed_score_buffs.clear()
    sim.timed_extra_gi_eot.clear()
    sim.perm_gi_effect_buff = 0.0
    sim.perm_score_buff = 0.0
    sim.score = 0
    sim.genki = 30
    sim.stamina = 33
    sim.buffs["goodImpressionTurns"] = 0
    sim.buffs["motivation"] = 10
    sim.buffs["nullifyDebuff"] = 0
    sim.buffs["halfCostTurns"] = 0
    sim.buffs["doubleCostTurns"] = 0
    sim.card_uses_remaining = 5
    return sim


def _spawn_test_card(
    sim: HeadlessBattleSimulator,
    db: GakumasDB,
    card_id: int,
    zone: list,
    *,
    exile_t: bool = False,
    extra_1st: bool = False,
):
    cfg = _make_card_config(
        db,
        card_id,
        exiles_trouble_on_play=exile_t,
        free_extra_use_first_play=extra_1st,
    )
    inst = sim._new_instance(cfg)
    sim._register_passive_effects_for_card(inst)
    zone.append(inst)
    return inst


def test_indirect_gi_sources_do_not_trigger_direct_effects():
    with GakumasDB("gakumas_library.db") as db:
        sim = _new_clean_sim(db, enable_509=True)
        m920 = _spawn_test_card(sim, db, 920, sim.exhaust_pile)
        _spawn_test_card(sim, db, 778, sim.hand)
        _spawn_test_card(sim, db, 127, sim.hand)
        _spawn_test_card(sim, db, 181, sim.hand)
        _spawn_test_card(sim, db, 173, sim.hand)

        # Arm #778 chase trigger (requires GI >= 10)
        sim.buffs["goodImpressionTurns"] = 10
        sim.play_card(0)  # #778

        # 1. Play #127: initial GI+5 is DIRECT (#509 -> 1, #920 -> 1, #778 chase fires)
        score_before_127 = sim.score
        sim.play_card(0)  # #127
        assert sim.buffs["goodImpressionTurns"] == 15
        assert sim.p_item_gi_counters[509] == 1
        assert sim.card_effect_counters[m920.uid] == 1
        assert sim.score > score_before_127

        # 2. Play #181: mentalCardUsed fires #127 (+1 INDIRECT -> no #509/#920/#778),
        #    then #181 initial GI+4 is DIRECT (#509 -> 2, #920 -> 2, #778 chase fires once)
        score_before_181 = sim.score
        sim.play_card(0)  # #181
        assert sim.buffs["goodImpressionTurns"] == 20  # 15 + 1 (#127 indirect) + 4 (#181 direct)
        assert sim.p_item_gi_counters[509] == 2
        assert sim.card_effect_counters[m920.uid] == 2
        expected_181_score = sim._compute_final_score_gain(20)
        assert sim.score - score_before_181 == expected_181_score

        # 3. Play #173: mentalCardUsed fires #127 (+1 INDIRECT -> still counter 2),
        #    then #173 GI+7 is DIRECT (#509 -> 3, #920 counter 2 -> triggers 再演!),
        #    and #920 再演 is a mental card so #127 adds +1 INDIRECT -> 20 + 1 + 7 + 1 = 29!
        sim.play_card(0)  # #173
        assert sim.buffs["goodImpressionTurns"] == 29
        assert sim.p_item_gi_counters[509] == 3
        assert sim.card_effect_counters[m920.uid] == 3
        assert sim.total_gi_gain_buff() == 0.5
        assert abs(sim.total_gi_effect_buff() - 2.5) < 1e-6  # +200% (#778) + 50% (#920 再演)

        # 4. End of Turn 1: #181 endOfTurn (+3 * 1.5 -> +5 INDIRECT GI) scales with #173 buff,
        #    does NOT increment #509 or #920 counters, and does NOT trigger #778 direct-GI chase!
        sim.card_uses_remaining = 0
        gi_before_eot = sim.buffs["goodImpressionTurns"]
        score_before_eot = sim.score
        sim.end_turn()
        # Only EOT GI tick (no #778 chase): gi_after_181 = 29 + 5 = 34
        expected_eot_tick = sim._compute_final_score_gain(34 * (1.0 + 2.5))
        assert score_before_eot + expected_eot_tick == sim.score
        assert sim.buffs["goodImpressionTurns"] == gi_before_eot + 5 - 1  # +5 from #181, -1 EOT decrement
        assert sim.p_item_gi_counters[509] == 3
        assert sim.card_effect_counters[m920.uid] == 3


def test_nullify_debuff_blocks_hatsuboshi_double_cost():
    with GakumasDB("gakumas_library.db") as db:
        sim = _new_clean_sim(db)
        _spawn_test_card(sim, db, 159, sim.hand)
        _spawn_test_card(sim, db, 922, sim.hand)

        # Play #159 -> nullifyDebuff = 1
        sim.play_card(0)
        assert sim.buffs["nullifyDebuff"] == 1
        hp_after_159 = sim.stamina

        # 1st #25 drink: nullifyDebuff blocks doubleCostTurns+=1 (1->0), still consumes 2 fixed HP & arms active double
        idx_25_a = next(i for i, d in enumerate(sim.available_drinks) if d.drink_id == 25)
        sim.use_drink(idx_25_a)
        assert sim.buffs["nullifyDebuff"] == 0
        assert sim.buffs["doubleCostTurns"] == 0
        assert sim.stamina == hp_after_159 - 2

        # Play #922: doubleCostTurns=0 so cost is 6 (not 12), and #25 doubles #922 (+3 + +3 = +6 GI)
        genki_before_922 = sim.genki
        sim.play_card(0)
        assert sim.genki == genki_before_922 - 6
        assert sim.buffs["goodImpressionTurns"] == 6

        # 2nd #25 drink: nullifyDebuff=0 so doubleCostTurns becomes 1 (fresh through Turn 2, expires EOT Turn 2)
        idx_25_b = next(i for i, d in enumerate(sim.available_drinks) if d.drink_id == 25)
        sim.use_drink(idx_25_b)
        assert sim.buffs["doubleCostTurns"] == 1
        assert "doubleCostTurns" in sim.fresh_turn_stat_buffs

        sim.card_uses_remaining = 0
        sim.end_turn()  # EOT Turn 1 -> Turn 2
        assert sim.turn == 2
        assert sim.buffs["doubleCostTurns"] == 1

        sim.card_uses_remaining = 0
        sim.end_turn()  # EOT Turn 2 -> Turn 3
        assert sim.turn == 3
        assert sim.buffs["doubleCostTurns"] == 0


def test_condition_snapshot_for_otomegokoro_and_hoshikuzu():
    with GakumasDB("gakumas_library.db") as db:
        # A. #299 at GI=6: pre-action snapshot sees 6 < 10 -> grants only +5 (not +5+2)
        sim_a = _new_clean_sim(db)
        _spawn_test_card(sim_a, db, 299, sim_a.hand)
        sim_a.buffs["goodImpressionTurns"] = 6
        sim_a.play_card(0)
        assert sim_a.buffs["goodImpressionTurns"] == 11

        # B. #299 at GI=9 with #127 ongoing trigger: mentalCardUsed adds +1 -> 10 before processCard snapshot
        sim_b = _new_clean_sim(db)
        _spawn_test_card(sim_b, db, 127, sim_b.hand)
        _spawn_test_card(sim_b, db, 299, sim_b.hand)
        sim_b.buffs["goodImpressionTurns"] = 4
        sim_b.play_card(0)  # #127 (+5 -> GI=9)
        assert sim_b.buffs["goodImpressionTurns"] == 9
        sim_b.play_card(0)  # #299: #127 mentalCardUsed (+1 -> 10) -> snapshot sees 10 >= 10 -> +5 and +2
        assert sim_b.buffs["goodImpressionTurns"] == 17

        # C. #173 at GI=7: pre-action snapshot sees 7 < 10 -> +7 GI, NO +50% GI gain buff
        sim_c = _new_clean_sim(db)
        _spawn_test_card(sim_c, db, 173, sim_c.hand)
        sim_c.buffs["goodImpressionTurns"] = 7
        sim_c.play_card(0)
        assert sim_c.buffs["goodImpressionTurns"] == 14
        assert sim_c.total_gi_gain_buff() == 0.0

        # D. #173 at GI=10: pre-action snapshot sees 10 >= 10 -> +7 GI AND +50% (5T) GI gain buff
        sim_d = _new_clean_sim(db)
        _spawn_test_card(sim_d, db, 173, sim_d.hand)
        sim_d.buffs["goodImpressionTurns"] = 10
        sim_d.play_card(0)
        assert sim_d.buffs["goodImpressionTurns"] == 17
        assert sim_d.total_gi_gain_buff() == 0.5


def test_920_recast_runs_full_free_card_use_and_triggers_listeners():
    with GakumasDB("gakumas_library.db") as db:
        sim = _new_clean_sim(db)
        m920 = _spawn_test_card(sim, db, 920, sim.exhaust_pile)
        # 10 more cards in exhaust_pile so #922 >=11 threshold triggers
        for _ in range(10):
            _spawn_test_card(sim, db, 179, sim.exhaust_pile)
        for _ in range(3):
            _spawn_test_card(sim, db, 179, sim.draw_pile)
        _spawn_test_card(sim, db, 305, sim.hand)
        _spawn_test_card(sim, db, 127, sim.hand)
        _spawn_test_card(sim, db, 129, sim.hand)
        c922 = _spawn_test_card(sim, db, 922, sim.hand)

        sim.buffs["goodImpressionTurns"] = 10
        sim.buffs["motivation"] = 4
        sim.play_card(0)  # #305 (cardUsed -> score+=GI*0.5)
        sim.play_card(0)  # #127 (mentalCardUsed -> GI+=1)
        sim.play_card(0)  # #129 (mentalCardUsed -> Mot+=1)

        # Arm #920 recast counter to 2 so next direct GI triggers 再演; reset turn counters for isolated check
        sim.card_effect_counters[m920.uid] = 2
        sim.card_turn_counters[m920.uid] = 0
        sim.buffs["goodImpressionTurns"] = 10
        sim.buffs["motivation"] = 0
        sim.genki = 30
        sim.score = 0
        sim.turn_cards_used = 0
        sim.card_uses_remaining = 1

        # Playing #922 (last action point!) triggers #920 再演 mid-#922; must not end turn before #922 finishes
        sim.play_card(0)
        assert sim.turn == 2  # Ended turn cleanly once #922 completed
        assert m920 in sim.exhaust_pile
        assert c922 in sim.discard_pile
        assert any("ありのままのわたし+(再演)" in entry for entry in sim.cards_played_log)
        # #129 triggered on #920 再演 mentalCardUsed -> motivation became 1
        assert sim.buffs["motivation"] == 1
        # #920 再演 ran genki+=4 + mot(1) = +5 (minus #922 cost 6) and +50% GI effect buff
        assert sim.genki == 29
        assert abs(sim.total_gi_effect_buff() - 0.5) < 1e-6
        # Score = #305 on #922 (GI=10) + #305 on #920 再演 (GI=13) + #922 burst (GI=19*2.8) + T1 EOT (GI=19*1.5)
        s_305_922 = sim._compute_final_score_gain(10 * 0.5)
        s_305_920 = sim._compute_final_score_gain(13 * 0.5)
        s_922_burst = sim._compute_final_score_gain(19 * 2.8)
        s_eot = sim._compute_final_score_gain(19 * 1.5)
        assert sim.score == s_305_922 + s_305_920 + s_922_burst + s_eot


def test_ttl_lifecycle_for_778_kimegao_and_25_drink():
    with GakumasDB("gakumas_library.db") as db:
        # Part A: #778 +200% buff and direct-GI chase trigger both persist across play turn + 4 turns, expire Turn 6
        sim = _new_clean_sim(db)
        _spawn_test_card(sim, db, 778, sim.hand)
        sim.buffs["goodImpressionTurns"] = 10
        sim.play_card(0)  # Turn 1

        for expected_turn in range(1, 6):
            assert sim.turn == expected_turn
            assert abs(sim.total_gi_effect_buff() - 2.0) < 1e-6
            sc_before = sim.score
            sim._add_good_impression_direct(1)
            assert sim.score > sc_before, f"#778 chase trigger should still fire on Turn {expected_turn}"
            sim.card_uses_remaining = 0
            sim.end_turn()

        assert sim.turn == 6
        assert abs(sim.total_gi_effect_buff() - 0.0) < 1e-6
        sc_t6 = sim.score
        sim._add_good_impression_direct(1)
        assert sim.score == sc_t6  # Both buff and chase trigger expired together on Turn 6

        # Part B: #25 drink (ttl:1) carries over to Turn 2 (ttl==0), expires on Turn 3 (ttl<0)
        sim_b1 = _new_clean_sim(db)
        idx_25 = next(i for i, d in enumerate(sim_b1.available_drinks) if d.drink_id == 25)
        sim_b1.use_drink(idx_25)
        sim_b1.card_uses_remaining = 0
        sim_b1.end_turn()  # EOT Turn 1: ttl 1 -> 0 (still alive for Turn 2)
        assert sim_b1.turn == 2
        _spawn_test_card(sim_b1, db, 922, sim_b1.hand)
        sim_b1.genki = 30
        sim_b1.buffs["goodImpressionTurns"] = 0
        sim_b1.play_card(0)  # Active #922 doubled (+3 + +3 = 6)
        assert sim_b1.buffs["goodImpressionTurns"] == 6

        sim_b2 = _new_clean_sim(db)
        idx_25_2 = next(i for i, d in enumerate(sim_b2.available_drinks) if d.drink_id == 25)
        sim_b2.use_drink(idx_25_2)
        sim_b2.card_uses_remaining = 0
        sim_b2.end_turn()  # EOT Turn 1: ttl 1 -> 0
        sim_b2.card_uses_remaining = 0
        sim_b2.end_turn()  # EOT Turn 2: ttl 0 -> -1 (expired!)
        assert sim_b2.turn == 3
        _spawn_test_card(sim_b2, db, 922, sim_b2.hand)
        sim_b2.genki = 30
        sim_b2.buffs["goodImpressionTurns"] = 0
        sim_b2.play_card(0)  # Active #922 NOT doubled (+3 only)
        assert sim_b2.buffs["goodImpressionTurns"] == 3


def test_card_routed_after_actions_eot_order_and_up_to_5_init_hand():
    with GakumasDB("gakumas_library.db") as db:
        # 1. Repeatable card with drawCard cannot reshuffle/redraw itself during its own play
        sim = _new_clean_sim(db)
        draw_repeatable_cfg = CardConfig(
            card_id=99901,
            name="TestDrawRepeatable",
            base_id=99901,
            upgraded_id=None,
            card_type="mental",
            rarity="R",
            upgraded=True,
            limit_count=None,
            force_initial_hand=False,
            p_idol_id=None,
            is_signature=False,
            parsed_effects={
                "conditions": [],
                "cost": [{"actions": [{"type": "assignment", "lhs": "cost", "op": "-=", "rhs": {"type": "number", "value": 0}}]}],
                "actions": [{"actions": [{"type": "identifier", "name": "drawCard"}]}],
                "effects": [],
            },
        )
        rep_inst = sim._new_instance(draw_repeatable_cfg)
        sim.hand.append(rep_inst)
        sim.play_card(0)
        assert rep_inst not in sim.hand
        assert rep_inst in sim.discard_pile

        # 2. endOfTurn runs before hand discard; voluntary SKIP is in legal_actions and heals +2 HP
        sim2 = _new_clean_sim(db)
        for _ in range(3):
            _spawn_test_card(sim2, db, 179, sim2.draw_pile)
        keeper = _spawn_test_card(sim2, db, 137, sim2.hand)
        sim2.stamina = 20
        sim2.card_uses_remaining = 1
        assert any(a[0] == "end_turn" for a in sim2.legal_actions())
        assert any(a[0] == "card" for a in sim2.legal_actions())
        # Trigger that awards +10 motivation ONLY if hand is still non-empty at endOfTurn
        sim2.triggers.append(
            ActiveTrigger(
                owner_name="EOTHandCheck",
                owner_uid=99999,
                phase="endOfTurn",
                conditions=[
                    {
                        "type": "comparison",
                        "op": ">=",
                        "left": {"type": "call", "name": "countCards", "target": {"type": "identifier", "name": "hand"}, "args": []},
                        "right": {"type": "number", "value": 1},
                    }
                ],
                actions=[{"type": "assignment", "lhs": "motivation", "op": "+=", "rhs": {"type": "number", "value": 10}}],
            )
        )
        sim2.buffs["motivation"] = 0
        sim2.end_turn()
        assert sim2.stamina == 22  # Voluntary skip healed +2 HP
        assert sim2.buffs["motivation"] == 10  # endOfTurn saw keeper still in hand before discard
        assert keeper in sim2.discard_pile

        # 3. Up to 5 force_initial_hand cards all enter opening hand
        init_deck = [_make_card_config(db, 159, force_initial_hand_override=True) for _ in range(5)]
        init_deck.extend(_make_card_config(db, 179, force_initial_hand_override=False) for _ in range(10))
        cfg5 = BattleConfig(
            scenario_id="test_5_init",
            stage_name="5_init_check",
            p_idol_id=163,
            idol_name="test",
            plan="logic",
            idol_stats={"vocal": 2000, "dance": 1500, "visual": 1500},
            initial_stamina=33,
            max_stamina=44,
            total_turns=3,
            turn_schedule=[TurnMultiplierSpec("Vo", 1.0)] * 3,
            deck=init_deck,
        )
        sim5 = HeadlessBattleSimulator(cfg5, db=db, seed=3)
        assert len(sim5.hand) == 5
        assert all(c.cfg.force_initial_hand for c in sim5.hand)
        assert len(sim5.draw_pile) == 10


def test_randomized_turn_schedule_rules_and_clone_invariance():
    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        seen_schedules = set()
        for seed in range(100):
            sim = HeadlessBattleSimulator(cfg, db=db, seed=seed, record_logs=False)
            assert len(sim.turn_schedule) == 12
            counts = Counter((s.attribute, s.multiplier) for s in sim.turn_schedule)
            assert counts == {("Vo", 40.0): 6, ("Vi", 35.0): 3, ("Da", 20.0): 3}
            assert sim.turn_schedule[11] == TurnMultiplierSpec("Vo", 40.0)
            assert sim.turn_schedule[9].attribute != "Vo"
            assert sim.turn_schedule[10].attribute != "Vo"
            seen_schedules.add(tuple((s.attribute, s.multiplier) for s in sim.turn_schedule))

            if seed == 0:
                cloned = sim.clone()
                det = sim.determinize(random.Random(999))
                assert cloned.turn_schedule == sim.turn_schedule
                assert det.turn_schedule == sim.turn_schedule

                sim.turn = 13
                assert sim.current_turn_spec() == TurnMultiplierSpec("Vo", 40.0)

        assert len(seen_schedules) > 10


def test_stacked_hatsuboshi_drinks_trigger_separate_active_cards_only():
    with GakumasDB("gakumas_library.db") as db:
        sim = _new_clean_sim(db)
        _spawn_test_card(sim, db, 778, sim.hand)  # Active 1
        _spawn_test_card(sim, db, 299, sim.hand)  # Mental (must NOT consume/inherit drink double)

        idx_25_a = next(i for i, d in enumerate(sim.available_drinks) if d.drink_id == 25)
        sim.use_drink(idx_25_a)
        idx_25_b = next(i for i, d in enumerate(sim.available_drinks) if d.drink_id == 25)
        sim.use_drink(idx_25_b)

        assert sim.stamina == 29  # 33 - 2 - 2
        assert sim.buffs["doubleCostTurns"] == 2
        assert "doubleCostTurns" in sim.fresh_turn_stat_buffs
        drink_trigs = [tr for tr in sim.triggers if tr.owner_uid == -25]
        assert len(drink_trigs) == 2
        assert all(tr.grants_double_card_effect for tr in drink_trigs)

        # 1st Active card (#778): doubled ONCE (+4.0 GI effect buff, not +6.0); 2nd #25 stays armed
        sim.play_card(0)
        assert abs(sim.total_gi_effect_buff() - 4.0) < 1e-6
        assert sim.double_card_effect_cards == 0
        assert drink_trigs[0].triggers_count == 1
        assert drink_trigs[1].triggers_count == 0

        # Next card is Mental (#299): triggers ONCE (+5 GI, net uses 4->4), 2nd #25 stays armed
        uses_before_299 = sim.card_uses_remaining
        sim.play_card(0)
        assert sim.buffs["goodImpressionTurns"] == 5
        assert sim.card_uses_remaining == uses_before_299
        assert drink_trigs[1].triggers_count == 0

        # End Turn 1 -> Turn 2: fresh clears (doubleCostTurns stays 2), 2nd #25 ttl drops 1 -> 0 (alive)
        sim.card_uses_remaining = 0
        sim.end_turn()
        assert sim.turn == 2
        assert sim.buffs["doubleCostTurns"] == 2
        assert drink_trigs[1].ttl == 0

        # Play 2nd Active card (#922) on Turn 2: 2nd #25 fires & doubles #922 (+3 -> +6 GI)
        c922 = _spawn_test_card(sim, db, 922, sim.hand)
        sim.card_uses_remaining = 1
        sim.genki = 30
        gi_before_922 = sim.buffs["goodImpressionTurns"]
        sim.play_card(sim.hand.index(c922))  # consumes remaining action -> advances to Turn 3
        assert drink_trigs[1].triggers_count == 1
        assert sim.turn == 3
        # #922 added +6 GI (twice +3), minus 1 at Turn 2 EOT => +5 net
        assert sim.buffs["goodImpressionTurns"] == gi_before_922 + 6 - 1
        assert sim.buffs["doubleCostTurns"] == 1  # T2 EOT: 2 -> 1

        # Turn 3 EOT -> Turn 4: stacked doubleCostTurns expires after T1/T2/T3 (3 turns total = 1+1+1)
        sim.card_uses_remaining = 0
        sim.end_turn()
        assert sim.turn == 4
        assert sim.buffs["doubleCostTurns"] == 0


def test_sense_independent_score_buff_multiplier_and_drink_22_tutor():
    from demo.sim.battle_simulator import _make_drink_config

    with GakumasDB("gakumas_library.db") as db:
        sim = _new_clean_sim(db, enable_404=False, enable_509=False)
        sim.plan = "sense"
        sim.turn_schedule = [TurnMultiplierSpec("Vo", 2.0) for _ in range(12)]
        sim.buffs["goodConditionTurns"] = 10
        sim.buffs["perfectConditionTurns"] = 2
        sim.buffs["concentration"] = 0
        sim.in_unfresh_phase = False

        # 1. Play #776 話題沸騰+: goodConditionTurnsMultiplier=2.5 scales ONLY the 好調/絶好調 bracket,
        # and grants independent timed setScoreBuff(0.3, 1) afterwards.
        # cond_bonus = 0.5 + 0.1*10 = 1.5 -> cond_mult = 1.0 + 1.5*2.5 = 4.75 -> ceil(50*4.75)=238 -> x2 = 476
        c776 = _spawn_test_card(sim, db, 776, sim.hand)
        sim.card_uses_remaining = 3
        sim.play_card(sim.hand.index(c776))
        assert sim.score == 476
        assert abs(sim.total_score_buff() - 0.3) < 1e-6
        assert sim.good_condition_multiplier == 1.0  # Reset after card finishes

        # 2. Drink #12 (+0.3, 3t) and #14 (+0.1, 5t): independent multiplier stacks to +70% (0.3+0.3+0.1),
        # completely separate from cond_mult (1.0 + 0.5 + 0.1*10 = 2.5)!
        # score+=50 at gc=10, pc=1, mult=2.0 -> ceil(ceil(50 * 2.5) * 1.7) * 2 = ceil(125 * 1.7) * 2 = 213 * 2 = 426
        sim.available_drinks = [_make_drink_config(db, 12), _make_drink_config(db, 14)]
        sim.use_drink(0)  # #12 (+0.3, 3t)
        sim.use_drink(0)  # #14 (+0.1, 5t)
        assert abs(sim.total_score_buff() - 0.7) < 1e-6
        sim.score = 0
        assert sim._compute_final_score_gain(50, apply_sense_mods=True) == 426

        # 3. #22 初星黒酢 tutor priority: avoids #23 眠気, holds back #814 at rem>2, tutors #814 at rem<=2
        sim_t = _new_clean_sim(db, enable_404=False, enable_509=False)
        sim_t.plan = "sense"
        sim_t.in_unfresh_phase = False
        _spawn_test_card(sim_t, db, 23, sim_t.draw_pile)
        _spawn_test_card(sim_t, db, 814, sim_t.draw_pile)
        _spawn_test_card(sim_t, db, 812, sim_t.discard_pile)
        sim_t.turns_remaining = 6
        sim_t.available_drinks = [_make_drink_config(db, 22)]
        sim_t.use_drink(0)
        assert [c.cfg.card_id for c in sim_t.hand] == [812]

        sim_t.hand.clear()
        sim_t.turns_remaining = 2
        sim_t.available_drinks = [_make_drink_config(db, 22)]
        sim_t.use_drink(0)
        assert [c.cfg.card_id for c in sim_t.hand] == [814]
        assert 23 not in [c.cfg.card_id for c in sim_t.hand]


def test_hand_cap_5_defers_overflow_drinks_and_exiles_nemuki_over_blank_skip():
    from demo.sim.battle_simulator import _make_drink_config
    from demo.sim.rl_env import GakumasActionCodec

    with GakumasDB("gakumas_library.db") as db:
        sim = _new_clean_sim(db, enable_404=False, enable_509=False)
        sim.plan = "sense"
        d14 = _make_drink_config(db, 14)
        d22 = _make_drink_config(db, 22)
        assert sim.drink_draw_count(d14) == 2
        assert sim.drink_draw_count(d22) == 1

        # 5 cards -> #22 deferred; 4 cards with playable non-finisher -> #14 deferred; 3 cards -> #14 allowed
        sim.turns_remaining = 3
        sim.stamina = 40
        sim.card_uses_remaining = 1
        sim.hand.clear()
        for cid in (812, 814, 162, 105):
            _spawn_test_card(sim, db, cid, sim.hand)
        assert sim.should_defer_drink_for_hand_cap(d14) is True
        _spawn_test_card(sim, db, 113, sim.hand)
        assert sim.should_defer_drink_for_hand_cap(d22) is True
        sim.hand = sim.hand[:3]
        assert sim.should_defer_drink_for_hand_cap(d14) is False

        # On rem>=2 with #23 眠気 playable and #814/#162 held, prune_dominated removes blank SKIP -> forces #23 exile
        sim2 = _new_clean_sim(db, enable_404=False, enable_509=False)
        sim2.plan = "sense"
        sim2.available_drinks.clear()
        sim2.turns_remaining = 6
        sim2.stamina = 40
        sim2.card_uses_remaining = 1
        sim2.hand.clear()
        for cid in (814, 162, 23):
            inst = _spawn_test_card(sim2, db, cid, sim2.hand)
            sim2.config.deck.append(inst.cfg)
        codec = GakumasActionCodec(sim2.config, db=db, finisher_play_max_turns_remaining=2)
        _, amap = codec.legal_mask_and_map(sim2, prune_dominated=True)
        assert codec.end_turn_action_id not in amap
        assert list(amap.values()) == [("card", 2)]

        # Playing #23 眠気 sends it cleanly to exhaust_pile
        idx_23 = amap[next(iter(amap))][1]
        sim2.step("card", idx_23)
        assert any(c.cfg.card_id == 23 for c in sim2.exhaust_pile)
        assert all(c.cfg.card_id != 23 for c in sim2.discard_pile)

        # When 2x is armed (double_card_effect_cards > 0), #23 must NOT be forced over SKIP/hold
        sim2.hand.clear()
        _spawn_test_card(sim2, db, 23, sim2.hand)
        sim2.draw_pile.clear()
        _spawn_test_card(sim2, db, 814, sim2.draw_pile)
        sim2.turns_remaining = 2
        sim2.stamina = 40
        sim2.card_uses_remaining = 1
        sim2.double_card_effect_cards = 1
        _, amap2 = codec.legal_mask_and_map(sim2, prune_dominated=True)
        assert codec.end_turn_action_id in amap2


def test_core_engine_hard_rules_no_drink22_at_5_hand_and_no_skip_with_trouble():
    """Verify both foundational rules live at the lowest engine layer (legal_actions & prune_dominated=False mask)."""
    from demo.sim.battle_simulator import _make_drink_config
    from demo.sim.rl_env import GakumasActionCodec

    with GakumasDB("gakumas_library.db") as db:
        sim = _new_clean_sim(db, enable_404=False, enable_509=False)
        sim.plan = "sense"
        sim.turns_remaining = 1
        sim.stamina = 40
        sim.card_uses_remaining = 1
        sim.hand.clear()
        for cid in (294, 112, 162, 23, 814):
            inst = _spawn_test_card(sim, db, cid, sim.hand)
            sim.config.deck.append(inst.cfg)
        d22 = _make_drink_config(db, 22)
        sim.available_drinks = [d22]
        sim.config.drinks = [d22]

        # Rule 1 (even at prune_dominated=False and rem=1): #22 is strictly illegal at 5-card hand cap!
        assert sim.can_use_drink(d22) is False
        assert not any(a[0] == "drink" for a in sim.legal_actions())
        codec = GakumasActionCodec(sim.config, db=db)
        mask_raw, amap_raw = codec.legal_mask_and_map(sim, prune_dominated=False)
        d22_aid = codec.drink_id_to_action[22]
        assert not mask_raw[d22_aid]
        assert d22_aid not in amap_raw

        # Dropping to 4 cards immediately makes #22 legal again
        sim.hand.pop()
        assert sim.can_use_drink(d22) is True
        assert any(a[0] == "drink" for a in sim.legal_actions())
        mask_4, amap_4 = codec.legal_mask_and_map(sim, prune_dominated=False)
        assert mask_4[d22_aid] and d22_aid in amap_4

        # Rule 2 (even at prune_dominated=False, T1/rem=9, low stamina): blank SKIP is strictly illegal when #23 is playable!
        sim.turns_remaining = 9
        sim.stamina = 2
        sim.double_card_effect_cards = 0
        sim.hand.clear()
        _spawn_test_card(sim, db, 23, sim.hand)
        sim.available_drinks.clear()
        assert sim.is_trouble_exile_mandatory() is True
        raw_acts = sim.legal_actions(include_voluntary_skip=True)
        assert not any(a[0] == "end_turn" for a in raw_acts)
        assert any(a[0] == "card" for a in raw_acts)
        mask_tr, amap_tr = codec.legal_mask_and_map(sim, prune_dominated=False)
        assert not mask_tr[codec.end_turn_action_id]
        assert codec.end_turn_action_id not in amap_tr
        assert list(amap_tr.values()) == [("card", 0)]


def test_resource_zeroing_finishers_dsl_and_anti_double_rules():
    """Verify #618/#619 (gc=0), #152/#153 (genki=0), #92/#93 (genki/=2), #27092 P-item, multiplier OCR, and tutor."""
    from demo.agent.local_agent import LocalHybridDecisionAgent
    from demo.schemas import ScreenObservation
    from demo.sim.battle_simulator import (
        BattleConfig,
        CardInstance,
        HeadlessBattleSimulator,
        TurnMultiplierSpec,
        _build_sense_hif_p_items,
        _make_card_config,
        _make_drink_config,
    )
    from demo.sim.live_battle_planner import _parse_live_turn_multiplier
    from demo.sim.rl_env import GakumasActionCodec

    with GakumasDB("gakumas_library.db") as db:
        # 1. Exact DSL semantics: #619 (gc=0 & score+=0 -> exact +0), #153 (genki=0), #93 (genki/=2)
        def _run_one(cid: int, *, dbl: int, gc: int = 0, pc: int = 0, conc: int = 0, genki: int = 0):
            cfg = _make_card_config(db, cid)
            bcfg = BattleConfig(
                scenario_id="hif_honsen",
                stage_name="unit",
                p_idol_id=114,
                idol_name="倉本 千奈",
                plan="sense" if cid in (618, 619) else "logic",
                idol_stats={"vocal": 1500, "dance": 1500, "visual": 1500},
                total_turns=9,
                max_stamina=100,
                initial_stamina=100,
                turn_schedule=[TurnMultiplierSpec("Da", 44.82) for _ in range(9)],
                deck=[cfg],
                p_items=[],
            )
            s = HeadlessBattleSimulator(bcfg)
            inst = CardInstance(uid=1, cfg=cfg)
            s.hand = [inst]
            s.turns_remaining = 1
            s.card_uses_remaining = 1
            s.buffs["goodConditionTurns"] = gc
            s.buffs["perfectConditionTurns"] = pc
            s.buffs["concentration"] = conc
            s.genki = genki
            s.double_card_effect_cards = dbl
            s._use_card(inst)
            return s.score, int(s.buffs.get("goodConditionTurns", 0) or 0), int(s.genki)

        sc619_1x, gc619_1x, _ = _run_one(619, dbl=0, gc=49, pc=15, conc=12)
        sc619_2x, gc619_2x, _ = _run_one(619, dbl=1, gc=49, pc=15, conc=12)
        assert sc619_1x == 158_081
        assert sc619_2x == 158_081  # exact +0 on 2nd trigger (matches live W7 S222->S223!)
        assert gc619_1x == 0 and gc619_2x == 0

        sc153_1x, _, gk153_1x = _run_one(153, dbl=0, genki=100)
        sc153_2x, _, gk153_2x = _run_one(153, dbl=1, genki=100)
        assert sc153_1x > 0
        assert sc153_2x == sc153_1x  # genki zeroed on 1st trigger -> 2nd trigger adds 0
        assert gk153_1x == 0 and gk153_2x == 0

        sc93_1x, _, gk93_1x = _run_one(93, dbl=0, genki=100)
        sc93_2x, _, gk93_2x = _run_one(93, dbl=1, genki=100)
        assert gk93_1x == 50
        assert gk93_2x == 25
        assert sc93_1x < sc93_2x < sc93_1x * 2  # 2nd trigger uses halved genki (50)

        # 2. Core engine hold & anti-2x rules for #619 and #153
        sim = _new_clean_sim(db, enable_404=False, enable_509=False)
        sim.plan = "sense"
        sim.turns_remaining = 5
        sim.stamina = 40
        sim.card_uses_remaining = 1
        sim.buffs["goodConditionTurns"] = 49
        sim.hand.clear()
        sim.draw_pile.clear()
        sim.discard_pile.clear()
        for cid in (619, 163, 295, 117):
            inst = _spawn_test_card(sim, db, cid, sim.hand)
            sim.config.deck.append(inst.cfg)
        d22 = _make_drink_config(db, 22)
        d25 = _make_drink_config(db, 25)
        d12 = _make_drink_config(db, 12)
        sim.available_drinks = [d22, d25, d12]
        sim.config.drinks = [d22, d25, d12]
        codec = GakumasActionCodec(sim.config, db=db, finisher_play_max_turns_remaining=2)

        # T5/rem=5: #619 held, #22 & #12 held, #163 NOT held (no #813 in deck!), #25 not allowed when #619 is only active
        assert sim.should_hold_for_finisher(card_cfg=sim.hand[0].cfg, fin_window=2) is True  # #619
        assert sim.should_hold_for_finisher(card_cfg=sim.hand[1].cfg, fin_window=2) is False  # #163 free to double buffs
        assert sim.should_hold_for_finisher(drink_cfg=d22, fin_window=2) is True
        assert sim.should_hold_for_finisher(drink_cfg=d12, fin_window=2) is True
        assert sim.can_use_active_retrigger_drink_now(d25) is False

        # Even if double_card_effect_cards==1 on T5/rem=5, #619 AND #22/#12 MUST STILL be held!
        # With empty hand (after playing #162), engine must SKIP to carry 2x to next turn instead of burning #22 early.
        sim.double_card_effect_cards = 1
        assert sim.should_hold_for_finisher(drink_cfg=d22, fin_window=2) is True
        assert sim.should_hold_for_finisher(drink_cfg=d12, fin_window=2) is True
        _, amap_t5 = codec.legal_mask_and_map(sim, prune_dominated=True)
        assert all(
            not (k == "card" and sim.hand[idx].cfg.card_id == 619)
            for k, idx in amap_t5.values()
        )
        assert all(
            not (k == "drink" and sim.available_drinks[idx].drink_id in (22, 12))
            for k, idx in amap_t5.values()
        )
        saved_hand = list(sim.hand)
        sim.draw_pile.append(saved_hand[0])  # #619 still unexhausted in draw pile
        sim.hand.clear()
        _, amap_empty_2x = codec.legal_mask_and_map(sim, prune_dominated=True)
        assert list(amap_empty_2x.values()) == [("end_turn", -1)]
        sim.draw_pile.clear()
        sim.hand = saved_hand

        # Final turn (rem=1) with double_card_effect_cards==1 and free #117 in hand: #117 takes 2x first, #619 forbidden until 2x spent
        sim.turns_remaining = 1
        sim.card_uses_remaining = 1
        _, amap_t1_free = codec.legal_mask_and_map(sim, prune_dominated=True)
        assert all(
            not (k == "card" and sim.hand[idx].cfg.card_id == 619)
            for k, idx in amap_t1_free.values()
        )

        # Final turn (rem=1) with ONLY [#619, #295] and card_uses_remaining==1 (no free card): #619 MUST remain legal!
        sim.hand = [c for c in sim.hand if c.cfg.card_id in (619, 295)]
        _, amap_t1_last = codec.legal_mask_and_map(sim, prune_dominated=True)
        assert any(k == "card" and sim.hand[idx].cfg.card_id == 619 for k, idx in amap_t1_last.values())

        # 3. #27092 届けたい想い P-item fires at afterStartOfTurn when turns_remaining <= 3
        p27092 = next(it for it in _build_sense_hif_p_items() if it.item_id == 27092)
        c619 = _make_card_config(db, 619)
        c74 = _make_card_config(db, 74)
        bcfg_p = BattleConfig(
            scenario_id="hif_honsen",
            stage_name="p27092",
            p_idol_id=114,
            idol_name="倉本 千奈",
            plan="sense",
            idol_stats={"vocal": 1500, "dance": 1500, "visual": 1500},
            total_turns=9,
            max_stamina=50,
            initial_stamina=50,
            turn_schedule=[TurnMultiplierSpec("Da", 40.0) for _ in range(9)],
            deck=[c619, c74, c74, c74],
            p_items=[p27092],
        )
        sim_p = HeadlessBattleSimulator(bcfg_p)
        sim_p.hand = []
        sim_p.draw_pile = [CardInstance(uid=10, cfg=c619)]
        sim_p.discard_pile = []
        sim_p.turns_remaining = 3
        sim_p.buffs["goodConditionTurns"] = 50
        sim_p.buffs["concentration"] = 10
        sim_p.stamina = 30
        sim_p._fire_phase("afterStartOfTurn", is_direct=False)
        assert any(c.cfg.card_id == 619 for c in sim_p.hand)
        assert sim_p.buffs["concentration"] == 20  # +20% of 50 = +10
        assert sim_p.stamina == 29
        assert sim_p.p_item_trigger_counts[27092] == 1

        # 4. Multiplier OCR handles ろ->6 and truncated ーカル
        assert _parse_live_turn_multiplier(ScreenObservation(scene="battle_turn", week_activity="試験 ダンスろ947%")) == ("Da", 69.47)
        assert _parse_live_turn_multiplier(ScreenObservation(scene="battle_turn", week_activity="試験 ビジュアルー5ろ17%")) == ("Vi", 56.17)
        assert _parse_live_turn_multiplier(ScreenObservation(scene="battle_turn", week_activity="試験 ーカル2017%")) == ("Vo", 20.17)

        # 5. Tutor modal scoring in #619-only deck
        agent = LocalHybridDecisionAgent(db=db)
        obs_t5 = ScreenObservation(scene="card_reward", week=7, turns_remaining=5)
        obs_t1 = ScreenObservation(scene="card_reward", week=7, turns_remaining=1)
        assert agent._score_battle_tutor_card(
            card_id=619, title="世界一の勇気+", desc="", obs=obs_t5, target_effect="goodConditionTurns", guide=None, plan="sense", p_idol_id=114
        ) < 0
        assert agent._score_battle_tutor_card(
            card_id=619, title="世界一の勇気+", desc="", obs=obs_t1, target_effect="goodConditionTurns", guide=None, plan="sense", p_idol_id=114
        ) == 22_000
        assert agent._score_battle_tutor_card(
            card_id=163, title="国民的アイドル+", desc="", obs=obs_t5, target_effect="goodConditionTurns", guide=None, plan="sense", p_idol_id=114
        ) < 8_500


