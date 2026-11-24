"""Online Single-Observer Determinized Information-Set MCTS (SO-ISMCTS)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from demo.sim.battle_simulator import BattleResult, HeadlessBattleSimulator
from demo.sim.maskable_ppo import PolicyValueNet
from demo.sim.rl_env import GakumasActionCodec, GakumasFeatureEncoder


class MinMaxStats:
    """Running value range normalizer for PUCT Q-values within one search tree."""

    def __init__(self) -> None:
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, value: float) -> None:
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return 0.0


@dataclass
class ISMCTSNode:
    """Information-set tree node storing only visit, value, prior, and availability stats."""

    children: Dict[int, "ISMCTSNode"] = field(default_factory=dict)
    edge_visits: Dict[int, int] = field(default_factory=dict)
    edge_value_sum: Dict[int, float] = field(default_factory=dict)
    edge_avail: Dict[int, int] = field(default_factory=dict)
    edge_prior_sum: Dict[int, float] = field(default_factory=dict)
    edge_prior_cnt: Dict[int, int] = field(default_factory=dict)


@dataclass
class ISMCTSConfig:
    """Configuration for online determinized PUCT ISMCTS."""

    num_simulations: int = 64
    c_puct: float = 1.25
    value_weight: float = 0.35
    rollout_depth: int = 0  # 0 means rollout to terminal state
    rollout_temperature: float = 0.0  # 0.0 uses greedy policy during rollouts
    score_scale: float = 1_000_000.0
    max_steps_per_rollout: int = 200
    prune_dominated_free_actions: bool = True
    free_action_prior_floor: float = 0.25
    memory_exile_prior_floor: float = 0.22


class ISMCTSAgent:
    """Hybrid Policy-Value guided Information-Set MCTS decision agent."""

    def __init__(
        self,
        codec: GakumasActionCodec,
        encoder: GakumasFeatureEncoder,
        net: Optional[PolicyValueNet] = None,
        cfg: Optional[ISMCTSConfig] = None,
    ):
        self.codec = codec
        self.encoder = encoder
        self.net = net
        self.cfg = cfg or ISMCTSConfig()

    def _priors_and_value(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        legal_actions: List[int],
        *,
        sim: Optional[HeadlessBattleSimulator] = None,
        action_map: Optional[Dict[int, Tuple[str, int]]] = None,
    ) -> Tuple[np.ndarray, float]:
        if self.net is not None:
            priors, val = self.net.predict_single(obs, mask)
            priors = priors.copy()
        else:
            priors = np.zeros(self.codec.num_actions, dtype=np.float64)
            u = 1.0 / float(len(legal_actions))
            for a in legal_actions:
                priors[a] = u
            val = 0.0

        if sim is not None and action_map is not None and legal_actions:
            cfg = self.cfg
            has_troubles = sim.has_active_troubles() and sim.turns_remaining >= 3
            modified = False
            for a in legal_actions:
                kind, idx = action_map.get(a, ("", -1))
                if kind == "card" and 0 <= idx < len(sim.hand):
                    inst = sim.hand[idx]
                    if sim.is_safe_dominating_free_card(inst) and float(priors[a]) < cfg.free_action_prior_floor:
                        priors[a] = cfg.free_action_prior_floor
                        modified = True
                    if has_troubles and sim.has_unused_memory_exile(inst) and float(priors[a]) < cfg.memory_exile_prior_floor:
                        priors[a] = max(float(priors[a]), cfg.memory_exile_prior_floor)
                        modified = True
            if modified:
                total = sum(max(0.0, float(priors[a])) for a in legal_actions)
                if total > 0.0:
                    for a in legal_actions:
                        priors[a] = max(0.0, float(priors[a])) / total
        return priors, val

    def _rollout_value(self, d: HeadlessBattleSimulator, rng: random.Random) -> float:
        cfg = self.cfg
        depth_limit = cfg.rollout_depth if cfg.rollout_depth > 0 else cfg.max_steps_per_rollout
        steps = 0
        np_rng = (
            np.random.default_rng(rng.randrange(1 << 63))
            if (self.net is not None and cfg.rollout_temperature > 0.0)
            else None
        )

        while d.turns_remaining > 0 and steps < depth_limit and steps < cfg.max_steps_per_rollout:
            steps += 1
            mask, action_map = self.codec.legal_mask_and_map(d, prune_dominated=cfg.prune_dominated_free_actions)
            legal = list(action_map.keys())
            if len(legal) == 1:
                chosen = legal[0]
            elif self.net is None:
                chosen = d.rng.choice(legal)
            elif cfg.rollout_temperature <= 0.0:
                obs = self.encoder.encode(d)
                probs, _ = self.net.predict_single(obs, mask)
                chosen = legal[int(np.argmax([probs[a] for a in legal]))]
            else:
                obs = self.encoder.encode(d)
                chosen, _, _ = self.net.select_action(
                    obs,
                    mask,
                    np_rng,
                    greedy=False,
                    temperature=cfg.rollout_temperature,
                )
            kind, idx = action_map[chosen]
            d.step(kind, idx)

        base_score = float(d.score) / cfg.score_scale
        if d.turns_remaining <= 0 or self.net is None:
            return base_score

        obs_end = self.encoder.encode(d)
        mask_end = self.codec.action_mask(d)
        _, v_rem = self.net.predict_single(obs_end, mask_end)
        return base_score + max(0.0, float(v_rem))

    def _evaluate_leaf(
        self,
        d: HeadlessBattleSimulator,
        rng: random.Random,
        leaf_obs: Optional[np.ndarray] = None,
        leaf_mask: Optional[np.ndarray] = None,
    ) -> float:
        cfg = self.cfg
        if d.turns_remaining <= 0:
            return float(d.score) / cfg.score_scale

        eff_vw = cfg.value_weight if self.net is not None else 0.0
        eff_vw = max(0.0, min(1.0, eff_vw))

        v_est = 0.0
        if eff_vw > 0.0 and self.net is not None:
            obs = leaf_obs if leaf_obs is not None else self.encoder.encode(d)
            mask = leaf_mask if leaf_mask is not None else self.codec.action_mask(d)
            _, v_rem = self.net.predict_single(obs, mask)
            v_est = float(d.score) / cfg.score_scale + max(0.0, float(v_rem))

        if eff_vw >= 1.0:
            return v_est

        r_est = self._rollout_value(d, rng)
        if eff_vw <= 0.0:
            return r_est
        return eff_vw * v_est + (1.0 - eff_vw) * r_est

    def select_action(
        self,
        sim: HeadlessBattleSimulator,
        rng: random.Random,
        *,
        allowed_root_actions: Optional[Set[int]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Run SO-ISMCTS from `sim` and return (best_action_id, search_stats)."""
        cfg = self.cfg
        root_mask, root_map = self.codec.legal_mask_and_map(sim, prune_dominated=False)
        root_legal = list(root_map.keys())
        if allowed_root_actions:
            filtered_legal = [a for a in root_legal if a in allowed_root_actions]
            if filtered_legal:
                root_legal = filtered_legal
                root_mask = np.zeros_like(root_mask)
                for a in root_legal:
                    root_mask[a] = True
        if cfg.prune_dominated_free_actions and len(root_legal) > 1:
            drink_filtered = [
                a
                for a in root_legal
                if not (
                    root_map.get(a, ("", -1))[0] == "drink"
                    and (
                        not sim.can_use_active_retrigger_drink_now(sim.available_drinks[root_map[a][1]])
                        or sim.should_defer_drink_for_hand_cap(sim.available_drinks[root_map[a][1]])
                    )
                )
            ]
            if drink_filtered and len(drink_filtered) != len(root_legal):
                root_legal = drink_filtered
                root_mask = np.zeros_like(root_mask)
                for a in root_legal:
                    root_mask[a] = True

            fin_window = getattr(self.codec, "finisher_play_max_turns_remaining", 2)
            if fin_window > 0 and sim.double_card_effect_cards == 0:
                has_813 = sim.has_unexhausted_card_family(813)
                hold_filtered = [
                    a
                    for a in root_legal
                    if not sim.should_hold_for_finisher(
                        card_cfg=sim.hand[root_map[a][1]].cfg if root_map.get(a, ("", -1))[0] == "card" else None,
                        drink_cfg=sim.available_drinks[root_map[a][1]] if root_map.get(a, ("", -1))[0] == "drink" else None,
                        fin_window=fin_window,
                        has_813=has_813,
                    )
                ]
                if hold_filtered and len(hold_filtered) != len(root_legal):
                    root_legal = hold_filtered
                    root_mask = np.zeros_like(root_mask)
                    for a in root_legal:
                        root_mask[a] = True

            handled_armed_active = False
            if sim.double_card_effect_cards > 0:
                fin_root = {
                    a
                    for a in root_legal
                    if root_map.get(a, ("", -1))[0] == "card"
                    and sim.is_late_scaling_finisher(sim.hand[root_map[a][1]].cfg)
                }
                if fin_root:
                    pruned_fin = [
                        a
                        for a in root_legal
                        if a in fin_root
                        or (
                            root_map.get(a, ("", -1))[0] == "drink"
                            and not sim.is_active_retrigger_drink(sim.available_drinks[root_map[a][1]])
                        )
                    ]
                    if pruned_fin:
                        root_legal = pruned_fin
                        root_mask = np.zeros_like(root_mask)
                        for a in root_legal:
                            root_mask[a] = True
                        handled_armed_active = True
                elif (
                    len(sim.hand) < sim.MAX_HAND_SIZE
                    and any(c.cfg.base_id == 813 for z in (sim.draw_pile, sim.discard_pile) for c in z)
                ):
                    tutor_root = {
                        a
                        for a in root_legal
                        if root_map.get(a, ("", -1))[0] == "drink"
                        and sim.is_tutor_drink(sim.available_drinks[root_map[a][1]])
                    }
                    if tutor_root:
                        pruned_tutor = [
                            a
                            for a in root_legal
                            if a in tutor_root
                            or (
                                root_map.get(a, ("", -1))[0] == "drink"
                                and sim.is_score_buff_drink(sim.available_drinks[root_map[a][1]])
                            )
                        ]
                        root_legal = pruned_tutor
                        root_mask = np.zeros_like(root_mask)
                        for a in root_legal:
                            root_mask[a] = True
                        handled_armed_active = True
                if (
                    (not handled_armed_active)
                    and sim.turns_remaining > 1
                    and sim.turns_remaining <= max(2, fin_window)
                    and len(sim.draw_pile) <= 3
                    and any(c.cfg.base_id == 813 for c in sim.draw_pile)
                    and self.codec.end_turn_action_id in root_legal
                ):
                    pruned_carry = [
                        a
                        for a in root_legal
                        if a == self.codec.end_turn_action_id
                        or (
                            root_map.get(a, ("", -1))[0] == "drink"
                            and not sim.is_active_retrigger_drink(sim.available_drinks[root_map[a][1]])
                        )
                    ]
                    if pruned_carry:
                        root_legal = pruned_carry
                        root_mask = np.zeros_like(root_mask)
                        for a in root_legal:
                            root_mask[a] = True
                        handled_armed_active = True

            if (not handled_armed_active) and sim.has_armed_active_retrigger(same_turn_only=True):
                active_root = {
                    a
                    for a in root_legal
                    if root_map.get(a, ("", -1))[0] == "card"
                    and sim.hand[root_map[a][1]].cfg.card_type == "active"
                    and sim.hand[root_map[a][1]].cfg.rarity != "L"
                }
                if active_root:
                    pruned_legal = [
                        a
                        for a in root_legal
                        if a in active_root
                        or (
                            root_map.get(a, ("", -1))[0] == "drink"
                            and not sim.is_active_retrigger_drink(sim.available_drinks[root_map[a][1]])
                        )
                    ]
                    if pruned_legal:
                        root_legal = pruned_legal
                        root_mask = np.zeros_like(root_mask)
                        for a in root_legal:
                            root_mask[a] = True
                        handled_armed_active = True

            if not handled_armed_active and len(root_legal) > 1:
                if sim.card_uses_remaining == 1:
                    dom_aids = {
                        a for a in root_legal
                        if root_map.get(a, ("", -1))[0] == "card"
                        and sim.is_safe_dominating_free_card(sim.hand[root_map[a][1]])
                    }
                else:
                    # Even when card_uses_remaining > 1, never ignore an unused free memory extra-use card!
                    dom_aids = {
                        a for a in root_legal
                        if root_map.get(a, ("", -1))[0] == "card"
                        and sim.has_unused_memory_extra_use(sim.hand[root_map[a][1]])
                        and sim.is_safe_dominating_free_card(sim.hand[root_map[a][1]])
                    }
                    if dom_aids:
                        # Allow choosing among all safe dominating free cards (including memory extra-use) & drinks
                        dom_aids |= {
                            a for a in root_legal
                            if root_map.get(a, ("", -1))[0] == "card"
                            and sim.is_safe_dominating_free_card(sim.hand[root_map[a][1]])
                        }
                if dom_aids:
                    pruned_legal = [
                        a for a in root_legal
                        if a in dom_aids or root_map.get(a, ("", -1))[0] == "drink"
                    ]
                    if pruned_legal:
                        root_legal = pruned_legal
                        root_mask = np.zeros_like(root_mask)
                        for a in root_legal:
                            root_mask[a] = True

            if self.codec.end_turn_action_id in root_legal and len(root_legal) > 1:
                has_non_trouble_card = any(
                    root_map.get(a, ("", -1))[0] == "card"
                    and not sim.is_trouble_card(sim.hand[root_map[a][1]].cfg)
                    for a in root_legal
                )
                has_trouble_card = any(
                    root_map.get(a, ("", -1))[0] == "card"
                    and sim.is_trouble_card(sim.hand[root_map[a][1]].cfg)
                    for a in root_legal
                )
                if sim.should_prune_blank_end_turn(
                    has_non_trouble_card=has_non_trouble_card,
                    has_trouble_card=has_trouble_card,
                ):
                    root_legal = [a for a in root_legal if a != self.codec.end_turn_action_id]
                    root_mask = np.zeros_like(root_mask)
                    for a in root_legal:
                        root_mask[a] = True
        if not root_legal:
            raise RuntimeError("No legal actions available from simulator state")

        if len(root_legal) == 1:
            only_a = root_legal[0]
            return only_a, {
                "best_q_score": float(sim.score),
                "visits": 1,
                "top_actions": [(self.codec.action_name(only_a), 1, float(sim.score))],
            }

        root = ISMCTSNode()
        min_max = MinMaxStats()
        root_obs = self.encoder.encode(sim) if self.net is not None else None

        for _ in range(max(1, cfg.num_simulations)):
            d = sim.determinize(rng, keep_logs=False)
            node = root
            path: List[Tuple[ISMCTSNode, int]] = []
            depth = 0
            leaf_obs: Optional[np.ndarray] = None
            leaf_mask: Optional[np.ndarray] = None

            while d.turns_remaining > 0:
                if depth == 0:
                    mask, action_map, legal = root_mask, root_map, root_legal
                else:
                    mask, action_map = self.codec.legal_mask_and_map(d, prune_dominated=cfg.prune_dominated_free_actions)
                    legal = list(action_map.keys())

                for a in legal:
                    node.edge_avail[a] = node.edge_avail.get(a, 0) + 1

                unexpanded = [a for a in legal if a not in node.children]
                if unexpanded:
                    if depth == 0 and root_obs is not None:
                        obs = root_obs
                    elif self.net is not None:
                        obs = self.encoder.encode(d)
                    else:
                        obs = np.empty(0, dtype=np.float64)

                    priors, _ = self._priors_and_value(obs, mask, legal, sim=d, action_map=action_map)
                    for a in legal:
                        node.edge_prior_sum[a] = node.edge_prior_sum.get(a, 0.0) + float(priors[a])
                        node.edge_prior_cnt[a] = node.edge_prior_cnt.get(a, 0) + 1

                    a_new = max(unexpanded, key=lambda a: float(priors[a]))
                    node.children[a_new] = ISMCTSNode()
                    path.append((node, a_new))
                    kind, idx = action_map[a_new]
                    d.step(kind, idx)
                    break

                best_a = legal[0]
                best_score = -float("inf")
                for a in legal:
                    visits = node.edge_visits.get(a, 0)
                    avail = node.edge_avail.get(a, 1)
                    prior = node.edge_prior_sum.get(a, 0.0) / max(1, node.edge_prior_cnt.get(a, 1))
                    if visits > 0:
                        q_mean = node.edge_value_sum[a] / float(visits)
                        q_norm = min_max.normalize(q_mean)
                    else:
                        q_norm = 0.0
                    u = q_norm + cfg.c_puct * prior * math.sqrt(float(avail)) / (1.0 + float(visits))
                    if u > best_score:
                        best_score = u
                        best_a = a

                path.append((node, best_a))
                kind, idx = action_map[best_a]
                d.step(kind, idx)
                node = node.children[best_a]
                depth += 1

            leaf_val = self._evaluate_leaf(d, rng, leaf_obs=leaf_obs, leaf_mask=leaf_mask)
            for p_node, p_a in path:
                new_v = p_node.edge_visits.get(p_a, 0) + 1
                new_sum = p_node.edge_value_sum.get(p_a, 0.0) + leaf_val
                p_node.edge_visits[p_a] = new_v
                p_node.edge_value_sum[p_a] = new_sum
                min_max.update(new_sum / float(new_v))

        def _action_key(a: int) -> Tuple[int, float]:
            vis = root.edge_visits.get(a, 0)
            q_mean = (root.edge_value_sum.get(a, 0.0) / float(vis)) if vis > 0 else -float("inf")
            return (vis, q_mean)

        best_action = max(root_legal, key=_action_key)
        best_vis = root.edge_visits.get(best_action, 0)
        best_q_score = (
            (root.edge_value_sum.get(best_action, 0.0) / float(best_vis)) * cfg.score_scale
            if best_vis > 0
            else float(sim.score)
        )
        ranked = sorted(root_legal, key=_action_key, reverse=True)
        top_actions = [
            (
                self.codec.action_name(a),
                root.edge_visits.get(a, 0),
                (root.edge_value_sum.get(a, 0.0) / float(root.edge_visits[a])) * cfg.score_scale
                if root.edge_visits.get(a, 0) > 0
                else 0.0,
            )
            for a in ranked[:5]
        ]
        return best_action, {
            "best_q_score": best_q_score,
            "visits": best_vis,
            "top_actions": top_actions,
        }

    def play_episode(
        self,
        sim: HeadlessBattleSimulator,
        rng: random.Random,
        *,
        max_steps: int = 200,
    ) -> Tuple[BattleResult, Dict[str, Any]]:
        """Play one complete battle using ISMCTS action selection."""
        t0 = time.perf_counter()
        decisions = 0
        search_decisions = 0
        steps = 0

        while sim.turns_remaining > 0 and steps < max_steps:
            steps += 1
            _, action_map = self.codec.legal_mask_and_map(sim)
            if len(action_map) > 1:
                search_decisions += 1
            best_action, _ = self.select_action(sim, rng)
            decisions += 1
            kind, idx = self.codec.resolve_action(sim, best_action)
            sim.step(kind, idx)

        elapsed_sec = time.perf_counter() - t0
        result = sim.build_result()
        stats = {
            "elapsed_sec": elapsed_sec,
            "decisions": decisions,
            "search_decisions": search_decisions,
            "avg_ms_per_decision": (elapsed_sec * 1000.0 / max(1, decisions)),
        }
        return result, stats
