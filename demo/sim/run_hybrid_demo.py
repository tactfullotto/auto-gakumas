"""CLI benchmark demo for Hybrid Maskable PPO + Online Determinized ISMCTS."""

from __future__ import annotations

import argparse
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from demo.core.db import GakumasDB
from demo.sim.battle_simulator import BattleConfig, BattleResult, HeadlessBattleSimulator, build_last_run_hif_preset
from demo.sim.ismcts import ISMCTSAgent, ISMCTSConfig
from demo.sim.maskable_ppo import MaskablePPOConfig, MaskablePPOTrainer, PolicyValueNet
from demo.sim.rl_env import GakumasActionCodec, GakumasBattleEnv, GakumasFeatureEncoder


def _eval_random(
    cfg: BattleConfig,
    db: GakumasDB,
    eval_seeds: List[int],
    cache: Dict[Any, Any],
) -> Tuple[List[BattleResult], List[float], int]:
    results: List[BattleResult] = []
    ep_times: List[float] = []
    decisions = 0
    for s in eval_seeds:
        t0 = time.perf_counter()
        sim = HeadlessBattleSimulator(cfg, db=db, seed=s, record_logs=True, card_cfg_cache=cache)
        steps = 0
        while sim.turns_remaining > 0 and steps < 300:
            steps += 1
            acts = sim.legal_actions()
            kind, idx, _ = sim.rng.choice(acts)
            decisions += 1
            sim.step(kind, idx)
        res = sim.build_result()
        ep_times.append(time.perf_counter() - t0)
        results.append(res)
    return results, ep_times, decisions


def _eval_ppo_greedy(
    cfg: BattleConfig,
    db: GakumasDB,
    codec: GakumasActionCodec,
    encoder: GakumasFeatureEncoder,
    net: PolicyValueNet,
    eval_seeds: List[int],
    cache: Dict[Any, Any],
) -> Tuple[List[BattleResult], List[float], int]:
    results: List[BattleResult] = []
    ep_times: List[float] = []
    decisions = 0
    for s in eval_seeds:
        t0 = time.perf_counter()
        sim = HeadlessBattleSimulator(cfg, db=db, seed=s, record_logs=True, card_cfg_cache=cache)
        steps = 0
        while sim.turns_remaining > 0 and steps < 300:
            steps += 1
            mask, action_map = codec.legal_mask_and_map(sim)
            if len(action_map) == 1:
                action_id = next(iter(action_map))
            else:
                obs = encoder.encode(sim)
                action_id, _, _ = net.select_action(obs, mask, greedy=True)
            decisions += 1
            kind, idx = action_map[action_id]
            sim.step(kind, idx)
        res = sim.build_result()
        ep_times.append(time.perf_counter() - t0)
        results.append(res)
    return results, ep_times, decisions


def _eval_ismcts(
    cfg: BattleConfig,
    db: GakumasDB,
    agent: ISMCTSAgent,
    eval_seeds: List[int],
    cache: Dict[Any, Any],
    *,
    rng_salt: int,
) -> Tuple[List[BattleResult], List[float], int]:
    results: List[BattleResult] = []
    ep_times: List[float] = []
    decisions = 0
    for s in eval_seeds:
        sim = HeadlessBattleSimulator(cfg, db=db, seed=s, record_logs=True, card_cfg_cache=cache)
        search_rng = random.Random(s + rng_salt)
        res, stats = agent.play_episode(sim, search_rng)
        results.append(res)
        ep_times.append(float(stats["elapsed_sec"]))
        decisions += int(stats["decisions"])
    return results, ep_times, decisions


def _summarize_method(name: str, results: List[BattleResult], ep_times: List[float], decisions: int) -> Dict[str, Any]:
    scores = np.asarray([r.final_score for r in results], dtype=np.float64)
    total_time = float(sum(ep_times))
    return {
        "name": name,
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "std": float(np.std(scores)),
        "avg_ms_decision": (total_time * 1000.0) / max(1, decisions),
        "avg_sec_ep": total_time / max(1, len(results)),
    }


def _print_comparison_table(rows: List[Dict[str, Any]]) -> None:
    print("\n=== Evaluation Comparison ===")
    header = (
        f"{'Method':<30} | {'Mean':>10} | {'Median':>10} | {'Max':>10} | "
        f"{'Min':>10} | {'Std':>9} | {'ms/step':>8} | {'sec/ep':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<30} | {r['mean']:>10,.0f} | {r['median']:>10,.0f} | {r['max']:>10,.0f} | "
            f"{r['min']:>10,.0f} | {r['std']:>9,.0f} | {r['avg_ms_decision']:>8.2f} | {r['avg_sec_ep']:>7.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Maskable PPO + Online Determinized ISMCTS local test demo")
    parser.add_argument("--stage", default="round2", choices=("round1", "round2"))
    parser.add_argument("--train-episodes", type=int, default=2400)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--value-weight", type=float, default=0.35)
    parser.add_argument("--rollout-depth", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--save-model", default=None, help="Optional .npz output path (must not be inside demo/state)")
    args = parser.parse_args()

    if args.save_model and "state" in str(args.save_model).replace("\\", "/").split("/"):
        raise ValueError("Refusing to write model artifacts inside a state directory")

    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage=args.stage)
        env = GakumasBattleEnv(cfg, db=db, score_scale=1_000_000.0, record_logs=False)

        print(f"=== Gakumas Maskable PPO + ISMCTS Demo ({cfg.stage_name}) ===")
        print(
            f"obs_dim={env.obs_dim} | num_actions={env.num_actions} | hidden={args.hidden_dim} | "
            f"train_ep={args.train_episodes} | eval_ep={args.eval_episodes} | sims={args.sims} | "
            f"c_puct={args.c_puct} | value_weight={args.value_weight} | rollout_depth={args.rollout_depth}"
        )

        ppo_cfg = MaskablePPOConfig(
            hidden_dim=args.hidden_dim,
            total_episodes=args.train_episodes,
            seed=args.seed,
        )
        trainer = MaskablePPOTrainer(env, ppo_cfg)
        train_t0 = time.perf_counter()

        def _on_upd(rec: Dict[str, Any]) -> None:
            elapsed = time.perf_counter() - train_t0
            if rec["val_mean"] is not None or rec["update"] == 1:
                val_str = f"{rec['val_mean']:,.0f}" if rec["val_mean"] is not None else "-"
                print(
                    f"[PPO upd {rec['update']:>3d} | ep {rec['episodes']:>4d}] "
                    f"train_mean={rec['train_mean']:>9,.0f} | train_max={rec['train_max']:>10,.0f} | "
                    f"val_greedy={val_str:>9} | ent={rec['entropy']:.3f} | "
                    f"vloss={rec['value_loss']:.4f} | elapsed={elapsed:.1f}s"
                )

        trained_net, train_summary = trainer.train(progress_callback=_on_upd)
        train_elapsed = time.perf_counter() - train_t0
        print(
            f"\nTraining complete in {train_elapsed:.1f}s | "
            f"best_val_mean={train_summary['best_val_mean']:,.0f} | "
            f"final_val_mean={train_summary['final_val_mean']:,.0f}"
        )

        if args.save_model:
            trained_net.save_npz(args.save_model)
            print(f"Saved trained PolicyValueNet weights to: {args.save_model}")

        eval_seeds = [args.seed + 10000 + i for i in range(args.eval_episodes)]
        shared_cache = env._card_cfg_cache

        rand_res, rand_t, rand_dec = _eval_random(cfg, db, eval_seeds, shared_cache)

        pure_ismcts_cfg = ISMCTSConfig(
            num_simulations=args.sims,
            c_puct=args.c_puct,
            value_weight=0.0,
            rollout_depth=0,
            rollout_temperature=0.0,
        )
        pure_agent = ISMCTSAgent(env.codec, env.encoder, net=None, cfg=pure_ismcts_cfg)
        pure_res, pure_t, pure_dec = _eval_ismcts(cfg, db, pure_agent, eval_seeds, shared_cache, rng_salt=900_000)

        ppo_res, ppo_t, ppo_dec = _eval_ppo_greedy(cfg, db, env.codec, env.encoder, trained_net, eval_seeds, shared_cache)

        hybrid_ismcts_cfg = ISMCTSConfig(
            num_simulations=args.sims,
            c_puct=args.c_puct,
            value_weight=args.value_weight,
            rollout_depth=args.rollout_depth,
            rollout_temperature=0.0,
        )
        hybrid_agent = ISMCTSAgent(env.codec, env.encoder, net=trained_net, cfg=hybrid_ismcts_cfg)
        hyb_res, hyb_t, hyb_dec = _eval_ismcts(cfg, db, hybrid_agent, eval_seeds, shared_cache, rng_salt=950_000)

        rows = [
            _summarize_method("1. Random", rand_res, rand_t, rand_dec),
            _summarize_method(f"2. Pure ISMCTS (N={args.sims})", pure_res, pure_t, pure_dec),
            _summarize_method("3. Maskable PPO (Greedy)", ppo_res, ppo_t, ppo_dec),
            _summarize_method(f"4. Hybrid PPO + ISMCTS (N={args.sims})", hyb_res, hyb_t, hyb_dec),
        ]
        _print_comparison_table(rows)

        wins_vs_ppo = sum(1 for h, p in zip(hyb_res, ppo_res) if h.final_score > p.final_score)
        ties_vs_ppo = sum(1 for h, p in zip(hyb_res, ppo_res) if h.final_score == p.final_score)
        wins_vs_pure = sum(1 for h, u in zip(hyb_res, pure_res) if h.final_score > u.final_score)
        print(
            f"\nHead-to-Head (n={len(eval_seeds)}): "
            f"Hybrid vs PPO Greedy = {wins_vs_ppo}/{len(eval_seeds)} wins ({ties_vs_ppo} ties) | "
            f"Hybrid vs Pure ISMCTS = {wins_vs_pure}/{len(eval_seeds)} wins"
        )

        best_res = max(hyb_res, key=lambda r: r.final_score)
        total_cards = sum(best_res.zone_counts.values())
        print(f"\n=== Best Hybrid Episode (seed={best_res.seed}) ===")
        print(
            f"得分: {best_res.final_score:,} | 回合数: {best_res.turns_played} | "
            f"剩余HP: {best_res.final_stamina}/{best_res.max_stamina} | 元气: {best_res.final_genki} | "
            f"好印象: {best_res.final_good_impression} (+{int(round(best_res.final_gi_effect_buff*100))}%) | "
            f"干劲: {best_res.final_motivation} | 除外区: {best_res.exhaust_count} 张 | 洗牌: {best_res.reshuffle_count} 次"
        )
        print(f"四区终局数量: {best_res.zone_counts} (守恒总张数={total_cards})")
        print(f"P道具触发统计: {best_res.p_item_activations}")
        print(f"メモリー触发记录 ({len(best_res.memory_triggers)}): {best_res.memory_triggers}")
        print(f"喝掉饮料 ({len(best_res.drinks_used)}): {best_res.drinks_used}")
        print(f"出牌序列 ({len(best_res.cards_played)}): {best_res.cards_played}")
        print("逐回合战况:")
        for tl in best_res.turn_logs:
            print(f"  {tl}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
