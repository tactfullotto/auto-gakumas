"""Unit and integration tests for simulator cloning, Maskable PPO, and SO-ISMCTS."""

from __future__ import annotations

import random

import numpy as np

from demo.core.db import GakumasDB
from demo.sim.battle_simulator import HeadlessBattleSimulator, build_last_run_hif_preset
from demo.sim.ismcts import ISMCTSAgent, ISMCTSConfig
from demo.sim.maskable_ppo import MaskablePPOConfig, MaskablePPOTrainer, PolicyValueNet
from demo.sim.rl_env import GakumasActionCodec, GakumasBattleEnv, GakumasFeatureEncoder


def test_clone_exact_fidelity_and_isolation():
    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        sim = HeadlessBattleSimulator(cfg, db=db, seed=101)
        # Advance a few steps into the battle so triggers/buffs/zones are non-trivial
        step_rng = random.Random(77)
        for _ in range(6):
            if sim.turns_remaining <= 0:
                break
            kind, idx, _ = step_rng.choice(sim.legal_actions())
            sim.step(kind, idx)

        snap_turn = sim.turn
        snap_score = sim.score
        snap_hand = [c.uid for c in sim.hand]
        snap_draw = [c.uid for c in sim.draw_pile]

        clone = sim.clone()
        # Closing a clone must not close the shared DB handle
        clone.close()
        assert db.connection is not None

        # Running the clone forward with same RNG stream produces identical outcomes,
        # while a second independent clone preserves original trajectory
        clone2 = sim.clone()
        res_clone = clone.run_random_episode()
        assert sim.turn == snap_turn
        assert sim.score == snap_score
        assert [c.uid for c in sim.hand] == snap_hand
        assert [c.uid for c in sim.draw_pile] == snap_draw

        res_orig = sim.run_random_episode()
        assert res_clone.final_score == res_orig.final_score
        assert res_clone.turns_played == res_orig.turns_played
        assert res_clone.zone_counts == res_orig.zone_counts
        assert res_clone.cards_played == res_orig.cards_played
        assert res_clone.drinks_used == res_orig.drinks_used
        assert clone2.score == snap_score


def test_determinize_preserves_information_set():
    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        sim = HeadlessBattleSimulator(cfg, db=db, seed=2026)
        # Play 1 card so zones/triggers are populated
        acts = sim.legal_actions(include_voluntary_skip=False)
        card_act = next(a for a in acts if a[0] == "card")
        sim.step(card_act[0], card_act[1])

        orig_hand_uids = [c.uid for c in sim.hand]
        orig_discard_uids = [c.uid for c in sim.discard_pile]
        orig_exhaust_uids = [c.uid for c in sim.exhaust_pile]
        orig_draw_uids = [c.uid for c in sim.draw_pile]
        orig_drinks = [d.drink_id for d in sim.available_drinks]
        orig_buffs = dict(sim.buffs)

        seen_orders = set()
        for s in range(10):
            det = sim.determinize(random.Random(s))
            assert [c.uid for c in det.hand] == orig_hand_uids
            assert [c.uid for c in det.discard_pile] == orig_discard_uids
            assert [c.uid for c in det.exhaust_pile] == orig_exhaust_uids
            assert [d.drink_id for d in det.available_drinks] == orig_drinks
            assert det.buffs == orig_buffs
            assert sorted(c.uid for c in det.draw_pile) == sorted(orig_draw_uids)
            seen_orders.add(tuple(c.uid for c in det.draw_pile))

        assert len(orig_draw_uids) >= 5
        assert len(seen_orders) > 1


def test_action_codec_and_mask_bijection():
    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        codec = GakumasActionCodec(cfg, db=db)

        # 22 distinct card slots + 2 distinct drinks (#13, #25) + 1 END_TURN = 25
        assert codec.num_card_slots == 22
        assert codec.num_drink_slots == 2
        assert codec.num_actions == 25
        assert (137, True, False) in codec.card_key_to_action
        assert (137, False, False) in codec.card_key_to_action
        assert codec.card_key_to_action[(137, True, False)] != codec.card_key_to_action[(137, False, False)]

        # Run full random episode via canonical actions only, checking #299 slot stability & legality
        sim = HeadlessBattleSimulator(cfg, db=db, seed=303)
        slot_299 = codec.card_key_to_action[(299, False, True)]
        rng = random.Random(99)
        while sim.turns_remaining > 0:
            mask, action_map = codec.legal_mask_and_map(sim)
            legal_ids = np.flatnonzero(mask)
            assert len(legal_ids) == len(action_map)
            has_trouble = any(
                k == "card" and sim.is_trouble_card(sim.hand[idx].cfg)
                for k, idx in action_map.values()
            )
            expected_end_turn = not (has_trouble and sim.is_trouble_exile_mandatory())
            assert bool(mask[codec.end_turn_action_id]) is expected_end_turn

            for h_idx, inst in enumerate(sim.hand):
                if inst.cfg.card_id == 299:
                    assert codec.card_slot_index(inst.cfg) == slot_299

            aid = int(rng.choice(legal_ids))
            kind, idx = codec.resolve_action(sim, aid)
            assert (kind, idx) == action_map[aid]
            sim.step(kind, idx)


def test_feature_encoder_draw_pile_order_invariance():
    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        codec = GakumasActionCodec(cfg, db=db)
        encoder = GakumasFeatureEncoder(codec)
        sim = HeadlessBattleSimulator(cfg, db=db, seed=404)

        obs_orig = encoder.encode(sim)
        assert obs_orig.shape == (encoder.obs_dim,)
        assert np.all(np.isfinite(obs_orig))

        reversed_sim = sim.clone()
        reversed_sim.draw_pile.reverse()
        obs_rev = encoder.encode(reversed_sim)
        assert np.array_equal(obs_orig, obs_rev)

        for s in range(5):
            det = sim.determinize(random.Random(100 + s))
            obs_det = encoder.encode(det)
            assert np.array_equal(obs_orig, obs_det)


def test_policy_value_net_analytical_gradients():
    rng = np.random.default_rng(12345)
    obs_dim = 18
    num_actions = 7
    batch = 6
    net = PolicyValueNet(obs_dim=obs_dim, num_actions=num_actions, hidden_dim=12, seed=7)

    obs = rng.normal(0.0, 1.0, size=(batch, obs_dim))
    masks = np.zeros((batch, num_actions), dtype=bool)
    actions = np.zeros(batch, dtype=np.int64)
    for b in range(batch):
        legal = rng.choice(num_actions, size=rng.integers(2, num_actions + 1), replace=False)
        masks[b, legal] = True
        actions[b] = int(rng.choice(legal))

    # Avoid exact clip/huber kinks by perturbing old_log_probs and returns cleanly
    logits0, _, _ = net.forward_batch(obs)
    from demo.sim.maskable_ppo import masked_log_softmax

    _, logp0, _ = masked_log_softmax(logits0, masks)
    old_log_probs = logp0[np.arange(batch), actions] + rng.uniform(-0.05, 0.05, size=batch)
    advantages = rng.normal(0.0, 1.0, size=batch)
    returns = rng.normal(0.5, 0.4, size=batch)

    _, grads = net.compute_ppo_loss_and_grads(
        obs,
        masks,
        actions,
        old_log_probs,
        advantages,
        returns,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.03,
        huber_delta=1.0,
    )

    eps = 1e-6
    for pname in net.PARAM_NAMES:
        p = net.params[pname]
        g = grads[pname]
        flat_p = p.reshape(-1)
        flat_g = g.reshape(-1)
        check_indices = np.linspace(0, len(flat_p) - 1, num=min(8, len(flat_p)), dtype=int)
        for idx in check_indices:
            orig_val = float(flat_p[idx])
            flat_p[idx] = orig_val + eps
            loss_plus, _ = net.compute_ppo_loss_and_grads(
                obs, masks, actions, old_log_probs, advantages, returns, clip_eps=0.2, vf_coef=0.5, ent_coef=0.03
            )
            flat_p[idx] = orig_val - eps
            loss_minus, _ = net.compute_ppo_loss_and_grads(
                obs, masks, actions, old_log_probs, advantages, returns, clip_eps=0.2, vf_coef=0.5, ent_coef=0.03
            )
            flat_p[idx] = orig_val

            num_grad = (loss_plus["total_loss"] - loss_minus["total_loss"]) / (2.0 * eps)
            ana_grad = float(flat_g[idx])
            denom = max(1e-7, abs(num_grad) + abs(ana_grad))
            rel_err = abs(num_grad - ana_grad) / denom
            assert rel_err < 1e-4, f"Grad mismatch for {pname}[{idx}]: num={num_grad}, ana={ana_grad}"


def test_maskable_ppo_training_and_checkpoint_roundtrip(tmp_path):
    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        with GakumasBattleEnv(cfg, db=db, score_scale=1_000_000.0, record_logs=False) as env:
            ppo_cfg = MaskablePPOConfig(
                hidden_dim=64,
                total_episodes=320,
                episodes_per_update=32,
                ppo_epochs=4,
                minibatch_size=128,
                val_interval_updates=5,
                val_episodes=8,
                seed=555,
            )
            trainer = MaskablePPOTrainer(env, ppo_cfg)
            net, summary = trainer.train()

            for rec in summary["history"]:
                assert np.isfinite(rec["policy_loss"])
                assert np.isfinite(rec["value_loss"])
                assert np.isfinite(rec["entropy"])

            eval_seeds = [80000 + i for i in range(10)]
            ppo_stats = trainer.evaluate_greedy(eval_seeds)

            rand_scores = []
            for s in eval_seeds:
                sim = HeadlessBattleSimulator(cfg, db=db, seed=s, record_logs=False)
                rand_scores.append(sim.run_random_episode().final_score)
            rand_mean = float(np.mean(rand_scores))
            assert ppo_stats["mean"] > rand_mean * 1.5

            ckpt_path = tmp_path / "ppo_test.npz"
            net.save_npz(ckpt_path)
            loaded = PolicyValueNet.load_npz(ckpt_path)

            obs, mask = env.reset(seed=80000)
            p1, v1 = net.predict_single(obs, mask)
            p2, v2 = loaded.predict_single(obs, mask)
            assert np.allclose(p1, p2)
            assert abs(v1 - v2) < 1e-12


def test_hybrid_ismcts_end_to_end_and_card_conservation():
    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        with GakumasBattleEnv(cfg, db=db, score_scale=1_000_000.0, record_logs=False) as env:
            ppo_cfg = MaskablePPOConfig(
                hidden_dim=64,
                total_episodes=256,
                episodes_per_update=32,
                val_interval_updates=4,
                val_episodes=6,
                seed=777,
            )
            net, _ = MaskablePPOTrainer(env, ppo_cfg).train()

            agent = ISMCTSAgent(
                env.codec,
                env.encoder,
                net=net,
                cfg=ISMCTSConfig(num_simulations=16, c_puct=1.25, value_weight=0.35),
            )

            eval_seeds = [90000 + i for i in range(6)]
            hybrid_scores = []
            rand_scores = []
            for s in eval_seeds:
                sim = HeadlessBattleSimulator(cfg, db=db, seed=s, record_logs=True)
                # Verify select_action at root does not mutate sim
                pre_score = sim.score
                pre_turn = sim.turn
                pre_hand = [c.uid for c in sim.hand]
                _, _ = agent.select_action(sim, random.Random(s + 1))
                assert sim.score == pre_score
                assert sim.turn == pre_turn
                assert [c.uid for c in sim.hand] == pre_hand

                res, _ = agent.play_episode(sim, random.Random(s + 2))
                total_cards = sum(res.zone_counts.values())
                assert total_cards in (22, 23)
                hybrid_scores.append(res.final_score)

                rsim = HeadlessBattleSimulator(cfg, db=db, seed=s, record_logs=False)
                rand_scores.append(rsim.run_random_episode().final_score)

            assert float(np.mean(hybrid_scores)) > float(np.mean(rand_scores)) * 1.5


def test_maskable_ppo_early_stopping_and_planner_disk_cache(tmp_path):
    from demo.core.tracker import GameStateTracker
    from demo.schemas import BoundingBox, LegalAction, RecognizedCard, ScreenObservation
    from demo.sim.live_battle_planner import LiveBattlePlanner, LiveBattlePlannerConfig

    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage="round2")
        with GakumasBattleEnv(cfg, db=db, score_scale=1_000_000.0, record_logs=False) as env:
            ppo_cfg = MaskablePPOConfig(
                hidden_dim=64,
                total_episodes=640,
                episodes_per_update=32,
                ppo_epochs=2,
                minibatch_size=128,
                val_interval_updates=2,
                val_episodes=6,
                early_stop_patience=2,
                early_stop_min_episodes=128,
                early_stop_min_rel_delta=10.0,  # force stale counter to trigger right at patience=2
                seed=777,
            )
            _, summary = MaskablePPOTrainer(env, ppo_cfg).train()
            assert summary["stopped_early"] is True
            assert summary["episodes_trained"] < 640

        cache_dir = tmp_path / "battle_cache"
        obs = ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            scenario_id="hif_honsen",
            week=9,
            turn=1,
            turns_remaining=4,
            stamina=40,
            max_stamina=44,
            buffs={"goodImpressionTurns": 6, "motivation": 6},
            cards=[
                RecognizedCard(card_id=159, name="アイドル魂+", playable=True, bbox=BoundingBox(xmin=20, ymin=1184, xmax=181, ymax=1456)),
                RecognizedCard(card_id=129, name="ワクワクが止まらない+", playable=True, bbox=BoundingBox(xmin=185, ymin=1184, xmax=346, ymax=1456)),
            ],
            options=[{"choice_id": 2, "name": "SKIP", "title": "SKIP"}],
        )
        acts = [
            LegalAction(choice_id=0, action_type="play_card", description="アイドル魂+", bbox=BoundingBox(xmin=20, ymin=1184, xmax=181, ymax=1456), payload={"card_id": 159, "name": "アイドル魂+"}),
            LegalAction(choice_id=1, action_type="play_card", description="ワクワクが止まらない+", bbox=BoundingBox(xmin=185, ymin=1184, xmax=346, ymax=1456), payload={"card_id": 129, "name": "ワクワクが止まらない+"}),
            LegalAction(choice_id=2, action_type="end_turn", description="SKIP", bbox=BoundingBox(xmin=580, ymin=1100, xmax=700, ymax=1160), payload={"name": "SKIP"}),
        ]
        tr = GameStateTracker(plan="logic", p_idol_id=163)
        tr.produce_deck = [920, 922, 159, 129, 127, 137, 23, 297, 299, 427]

        p_cfg = LiveBattlePlannerConfig(
            ppo_hidden_dim=32,
            ppo_episodes=256,
            ppo_episodes_per_update=32,
            ismcts_simulations=4,
            policy_cache_dir=str(cache_dir),
        )
        planner1 = LiveBattlePlanner(db=db, cfg=p_cfg)
        planner1.decide_battle_action(obs, acts, tracker=tr)
        cached_files = list(cache_dir.glob("*.npz"))
        assert len(cached_files) == 1

        # Fresh planner instance loads cached policy from disk without calling trainer.train()
        train_calls = 0
        orig_train = MaskablePPOTrainer.train

        def _count_train(self_tr, *a, **kw):
            nonlocal train_calls
            train_calls += 1
            return orig_train(self_tr, *a, **kw)

        MaskablePPOTrainer.train = _count_train  # type: ignore[method-assign]
        try:
            planner2 = LiveBattlePlanner(db=db, cfg=p_cfg)
            planner2.decide_battle_action(obs, acts, tracker=tr)
            assert train_calls == 0
            assert planner2._net is not None
        finally:
            MaskablePPOTrainer.train = orig_train  # type: ignore[method-assign]
