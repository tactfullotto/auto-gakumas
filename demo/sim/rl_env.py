"""Canonical action codec, information-set feature encoder, and RL environment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from demo.core.db import GakumasDB
from demo.sim.battle_simulator import BattleConfig, CardConfig, HeadlessBattleSimulator


CardKey = Tuple[int, bool, bool]


@dataclass(frozen=True)
class ActionSpec:
    """Canonical discrete action descriptor."""

    action_id: int
    kind: str  # "card" | "drink" | "end_turn"
    card_key: Optional[CardKey] = None
    drink_id: Optional[int] = None
    label: str = ""


class GakumasActionCodec:
    """Bijective mapping between canonical discrete IDs and simulator actions."""

    def __init__(
        self,
        config: BattleConfig,
        db: Optional[GakumasDB] = None,
        *,
        finisher_play_max_turns_remaining: int = 2,
    ):
        self.config = config
        self.finisher_play_max_turns_remaining = max(0, int(finisher_play_max_turns_remaining))
        self.specs: List[ActionSpec] = []
        self.card_slots: List[CardKey] = []
        self.card_key_to_action: Dict[CardKey, int] = {}
        self.card_id_to_fallback_action: Dict[int, int] = {}
        self.drink_slots: List[int] = []
        self.drink_id_to_action: Dict[int, int] = {}
        self.end_turn_action_id: int = -1

        self._build_slots(config, db)

    @staticmethod
    def card_key_for_cfg(cfg: CardConfig) -> CardKey:
        return (cfg.card_id, bool(cfg.free_extra_use_first_play), bool(cfg.exiles_trouble_on_play))

    def _register_card_slot(self, key: CardKey, label: str) -> int:
        if key in self.card_key_to_action:
            return self.card_key_to_action[key]
        aid = len(self.specs)
        self.specs.append(ActionSpec(action_id=aid, kind="card", card_key=key, label=label))
        self.card_slots.append(key)
        self.card_key_to_action[key] = aid
        self.card_id_to_fallback_action.setdefault(key[0], aid)
        return aid

    def _build_slots(self, config: BattleConfig, db: Optional[GakumasDB]) -> None:
        for cfg in config.deck:
            key = self.card_key_for_cfg(cfg)
            suffix = ""
            if cfg.free_extra_use_first_play:
                suffix += "[mem:+1use]"
            if cfg.exiles_trouble_on_play:
                suffix += "[mem:exile]"
            aid = self._register_card_slot(key, f"#{cfg.card_id} {cfg.name}{suffix}")
            if cfg.base_id and cfg.base_id != cfg.card_id:
                self.card_id_to_fallback_action.setdefault(cfg.base_id, aid)

        # Ensure potential mid-battle generated #23 (眠気), core logic #170/#171 (私がスター),
        # and upgraded deck targets always have canonical slots.
        extra_keys: List[Tuple[CardKey, str]] = [((23, False, False), "#23 眠気")]
        for cfg in config.deck:
            if cfg.upgraded_id:
                ekey = (
                    cfg.upgraded_id,
                    bool(cfg.free_extra_use_first_play),
                    bool(cfg.exiles_trouble_on_play),
                )
                extra_name = f"#{cfg.upgraded_id}"
                if db is not None:
                    row = db.get_card_by_id(cfg.upgraded_id)
                    if row:
                        extra_name = f"#{cfg.upgraded_id} {row['name']}"
                extra_keys.append((ekey, extra_name))

        for key, label in extra_keys:
            if key not in self.card_key_to_action:
                self._register_card_slot(key, label)

        plan_l = str(getattr(config, "plan", "logic") or "logic").lower()
        present_bases = {cfg.base_id or cfg.card_id for cfg in config.deck} | {cfg.card_id for cfg in config.deck}
        if plan_l in ("logic", "all", "") and 170 not in present_bases and 171 not in present_bases:
            star_row = db.get_card_by_id(171) if db is not None else None
            if db is None or star_row is not None:
                star_nm = str(star_row["name"]) if star_row else "私がスター+"
                aid_star = self._register_card_slot((171, False, False), f"#171 {star_nm}")
                self.card_id_to_fallback_action.setdefault(170, aid_star)

        for dr in config.drinks:
            if dr.drink_id in self.drink_id_to_action:
                continue
            aid = len(self.specs)
            self.specs.append(
                ActionSpec(
                    action_id=aid,
                    kind="drink",
                    drink_id=dr.drink_id,
                    label=f"Drink#{dr.drink_id} {dr.name}",
                )
            )
            self.drink_slots.append(dr.drink_id)
            self.drink_id_to_action[dr.drink_id] = aid

        self.end_turn_action_id = len(self.specs)
        self.specs.append(ActionSpec(action_id=self.end_turn_action_id, kind="end_turn", label="END_TURN"))

    @property
    def num_actions(self) -> int:
        return len(self.specs)

    @property
    def num_card_slots(self) -> int:
        return len(self.card_slots)

    @property
    def num_drink_slots(self) -> int:
        return len(self.drink_slots)

    def action_name(self, action_id: int) -> str:
        return self.specs[action_id].label

    def card_slot_index(self, cfg: CardConfig) -> int:
        key = self.card_key_for_cfg(cfg)
        aid = self.card_key_to_action.get(key)
        if aid is not None:
            return aid
        aid = self.card_id_to_fallback_action.get(cfg.card_id)
        if aid is not None:
            return aid
        if cfg.base_id:
            aid = self.card_id_to_fallback_action.get(cfg.base_id)
            if aid is not None:
                return aid
        return -1

    def legal_mask_and_map(
        self,
        sim: HeadlessBattleSimulator,
        *,
        prune_dominated: bool = False,
    ) -> Tuple[np.ndarray, Dict[int, Tuple[str, int]]]:
        mask = np.zeros(self.num_actions, dtype=bool)
        action_map: Dict[int, Tuple[str, int]] = {}

        if sim.turns_remaining <= 0:
            return mask, action_map

        for h_idx, inst in enumerate(sim.hand):
            if not sim.can_play_card(inst):
                continue
            aid = self.card_slot_index(inst.cfg)
            if 0 <= aid < self.num_card_slots and aid not in action_map:
                mask[aid] = True
                action_map[aid] = ("card", h_idx)

        for d_idx, dr in enumerate(sim.available_drinks):
            # Core engine legality rule (always active, even when prune_dominated=False):
            # never allow wasting tutor (#22) or pure-draw drinks at 5-card hand cap.
            if not sim.can_use_drink(dr):
                continue
            if prune_dominated and (
                not sim.can_use_active_retrigger_drink_now(dr)
                or sim.should_defer_drink_for_hand_cap(dr)
            ):
                continue
            aid = self.drink_id_to_action.get(dr.drink_id)
            if aid is not None and aid not in action_map:
                mask[aid] = True
                action_map[aid] = ("drink", d_idx)

        # Core engine legality rule (always active, even when prune_dominated=False):
        # if a playable trouble card (#23 眠気) is in hand on a non-final turn without 2x/retrigger armed,
        # blank END_TURN (SKIP) is illegal — exile the trouble card instead of skipping!
        has_playable_trouble_card = any(
            k == "card" and sim.is_trouble_card(sim.hand[idx].cfg)
            for k, idx in action_map.values()
        )
        if not (has_playable_trouble_card and sim.is_trouble_exile_mandatory()):
            mask[self.end_turn_action_id] = True
            action_map[self.end_turn_action_id] = ("end_turn", -1)

        if prune_dominated:
            if self.finisher_play_max_turns_remaining > 0 and sim.double_card_effect_cards == 0:
                has_813 = sim.has_unexhausted_card_family(813)
                held_aids = {
                    aid
                    for aid, (kind, idx) in action_map.items()
                    if sim.should_hold_for_finisher(
                        card_cfg=sim.hand[idx].cfg if kind == "card" else None,
                        drink_cfg=sim.available_drinks[idx] if kind == "drink" else None,
                        fin_window=self.finisher_play_max_turns_remaining,
                        has_813=has_813,
                    )
                }
                if held_aids and len(held_aids) < len(action_map):
                    action_map = {aid: v for aid, v in action_map.items() if aid not in held_aids}
                    for aid in held_aids:
                        mask[aid] = False

            if sim.double_card_effect_cards > 0:
                fin_aids = {
                    aid
                    for aid, (kind, idx) in action_map.items()
                    if kind == "card" and sim.is_late_scaling_finisher(sim.hand[idx].cfg)
                }
                if fin_aids:
                    action_map = {
                        aid: v
                        for aid, v in action_map.items()
                        if aid in fin_aids
                        or (v[0] == "drink" and not sim.is_active_retrigger_drink(sim.available_drinks[v[1]]))
                    }
                    mask.fill(False)
                    for aid in action_map:
                        mask[aid] = True
                    return mask, action_map
                tutor_aids = {
                    aid
                    for aid, (kind, idx) in action_map.items()
                    if kind == "drink" and sim.is_tutor_drink(sim.available_drinks[idx])
                }
                if (
                    tutor_aids
                    and len(sim.hand) < sim.MAX_HAND_SIZE
                    and any(c.cfg.base_id == 813 for z in (sim.draw_pile, sim.discard_pile) for c in z)
                ):
                    action_map = {
                        aid: v
                        for aid, v in action_map.items()
                        if aid in tutor_aids
                        or (
                            v[0] == "drink"
                            and sim.is_score_buff_drink(sim.available_drinks[v[1]])
                        )
                    }
                    mask.fill(False)
                    for aid in action_map:
                        mask[aid] = True
                    return mask, action_map
                if (
                    sim.turns_remaining > 1
                    and sim.turns_remaining <= max(2, self.finisher_play_max_turns_remaining)
                    and len(sim.draw_pile) <= 3
                    and any(c.cfg.base_id == 813 for c in sim.draw_pile)
                    and self.end_turn_action_id in action_map
                ):
                    action_map = {
                        aid: v
                        for aid, v in action_map.items()
                        if aid == self.end_turn_action_id
                        or (v[0] == "drink" and not sim.is_active_retrigger_drink(sim.available_drinks[v[1]]))
                    }
                    mask.fill(False)
                    for aid in action_map:
                        mask[aid] = True
                    return mask, action_map

            if sim.has_armed_active_retrigger(same_turn_only=True):
                active_aids = {
                    aid
                    for aid, (kind, idx) in action_map.items()
                    if kind == "card"
                    and sim.hand[idx].cfg.card_type == "active"
                    and sim.hand[idx].cfg.rarity != "L"
                }
                if active_aids:
                    action_map = {
                        aid: v
                        for aid, v in action_map.items()
                        if aid in active_aids
                        or (v[0] == "drink" and not sim.is_active_retrigger_drink(sim.available_drinks[v[1]]))
                    }
                    mask.fill(False)
                    for aid in action_map:
                        mask[aid] = True
                    return mask, action_map

            if sim.card_uses_remaining == 1:
                dom_aids = {
                    aid
                    for aid, (kind, idx) in action_map.items()
                    if kind == "card" and sim.is_safe_dominating_free_card(sim.hand[idx])
                }
                if dom_aids:
                    action_map = {
                        aid: v
                        for aid, v in action_map.items()
                        if aid in dom_aids or v[0] == "drink"
                    }
                    mask.fill(False)
                    for aid in action_map:
                        mask[aid] = True

            if (
                self.end_turn_action_id in action_map
                and len(action_map) > 1
            ):
                has_non_trouble_card = any(
                    k == "card" and not sim.is_trouble_card(sim.hand[idx].cfg)
                    for k, idx in action_map.values()
                )
                has_trouble_card = any(
                    k == "card" and sim.is_trouble_card(sim.hand[idx].cfg)
                    for k, idx in action_map.values()
                )
                if sim.should_prune_blank_end_turn(
                    has_non_trouble_card=has_non_trouble_card,
                    has_trouble_card=has_trouble_card,
                ):
                    action_map.pop(self.end_turn_action_id, None)
                    mask[self.end_turn_action_id] = False

        return mask, action_map

    def action_mask(self, sim: HeadlessBattleSimulator, *, prune_dominated: bool = False) -> np.ndarray:
        mask, _ = self.legal_mask_and_map(sim, prune_dominated=prune_dominated)
        return mask

    def resolve_action(self, sim: HeadlessBattleSimulator, action_id: int) -> Tuple[str, int]:
        mask, action_map = self.legal_mask_and_map(sim, prune_dominated=False)
        if action_id < 0 or action_id >= self.num_actions or not mask[action_id]:
            raise ValueError(f"Illegal canonical action_id={action_id} ({self.action_name(action_id)})")
        return action_map[action_id]


class GakumasFeatureEncoder:
    """Strictly information-set-safe state encoder (invariant to draw_pile order)."""

    SCALAR_DIMS = 17 + 22 + 29 + 8

    def __init__(self, codec: GakumasActionCodec):
        self.codec = codec
        self._num_cards = codec.num_card_slots
        self._num_drinks = codec.num_drink_slots
        self._drink_idx = {did: i for i, did in enumerate(codec.drink_slots)}
        self._obs_dim = self.SCALAR_DIMS + self._num_drinks + 5 * self._num_cards

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def encode(self, sim: HeadlessBattleSimulator) -> np.ndarray:
        obs = np.zeros(self._obs_dim, dtype=np.float64)
        idx = 0

        # 1. Turn & schedule lookahead (17 dims)
        cur_spec = sim.current_turn_spec()
        attr = cur_spec.attribute.lower()
        sched = sim.turn_schedule
        sched_len = len(sched)
        cur_t_idx = sim.turn - 1
        max_mult = max((s.multiplier for s in sched), default=40.0)
        if max_mult <= 0.0:
            max_mult = 1.0

        obs[idx] = sim.turn / 13.0
        obs[idx + 1] = sim.turns_remaining / 13.0
        obs[idx + 2] = sim.turns_elapsed / 13.0
        obs[idx + 3] = 1.0 if sim.turns_remaining <= 1 else 0.0
        obs[idx + 4] = cur_spec.multiplier / max_mult
        obs[idx + 5] = 1.0 if attr.startswith("vo") else 0.0
        obs[idx + 6] = 1.0 if attr.startswith("da") else 0.0
        obs[idx + 7] = 1.0 if attr.startswith("vi") else 0.0
        for offset in range(1, 4):
            if offset < sim.turns_remaining and sched_len > 0:
                s_idx = min(sched_len - 1, max(0, cur_t_idx + offset))
                obs[idx + 7 + offset] = sched[s_idx].multiplier / max_mult
        obs[idx + 11] = sim.card_uses_remaining / 4.0
        obs[idx + 12] = sim.turn_cards_used / 6.0
        obs[idx + 13] = sim.cards_used / 30.0

        rem_vo = 0
        rem_vi = 0
        rem_da = 0
        if sched_len > 0:
            for offset in range(1, max(1, sim.turns_remaining)):
                s_idx = min(sched_len - 1, max(0, cur_t_idx + offset))
                r_attr = sched[s_idx].attribute.lower()
                if r_attr.startswith("vo"):
                    rem_vo += 1
                elif r_attr.startswith("vi"):
                    rem_vi += 1
                elif r_attr.startswith("da"):
                    rem_da += 1
        obs[idx + 14] = rem_vo / 6.0
        obs[idx + 15] = rem_vi / 3.0
        obs[idx + 16] = rem_da / 3.0
        idx += 17

        # 2. Resources, score & core buffs (22 dims)
        buffs = sim.buffs
        gi = int(buffs.get("goodImpressionTurns", 0) or 0)
        mot = int(buffs.get("motivation", 0) or 0)
        ex_cnt = len(sim.exhaust_pile)
        max_stam = max(1, sim.max_stamina)

        obs[idx] = sim.stamina / max_stam
        obs[idx + 1] = sim.genki / 60.0
        obs[idx + 2] = math.log1p(max(0, sim.score)) / 15.0
        obs[idx + 3] = sim.score / 1_000_000.0
        obs[idx + 4] = gi / 150.0
        obs[idx + 5] = 1.0 if gi >= 10 else 0.0
        obs[idx + 6] = 1.0 if gi >= 15 else 0.0
        obs[idx + 7] = mot / 40.0
        obs[idx + 8] = 1.0 if mot >= 3 else 0.0
        obs[idx + 9] = 1.0 if mot >= 4 else 0.0
        obs[idx + 10] = float(buffs.get("halfCostTurns", 0) or 0) / 5.0
        obs[idx + 11] = float(buffs.get("doubleCostTurns", 0) or 0) / 3.0
        obs[idx + 12] = float(buffs.get("nullifyDebuff", 0) or 0) / 2.0
        obs[idx + 13] = sim.total_gi_effect_buff() / 5.0
        obs[idx + 14] = sim.perm_gi_effect_buff / 3.0
        obs[idx + 15] = sim.total_gi_gain_buff()
        obs[idx + 16] = sim.total_score_buff()
        obs[idx + 17] = sim.extra_gi_eot_triggers() / 3.0
        obs[idx + 18] = sim.double_card_effect_cards / 2.0
        obs[idx + 19] = ex_cnt / 22.0
        obs[idx + 20] = 1.0 if ex_cnt >= 7 else 0.0
        obs[idx + 21] = 1.0 if ex_cnt >= 11 else 0.0
        idx += 22

        # 3. P-items, #920 再演, active triggers & memory-once state (22 dims)
        cnt_509 = sim.p_item_gi_counters.get(509, 0)
        has_404 = 404 in sim.p_item_trigger_counts
        trig_404 = sim.p_item_trigger_counts.get(404, 0)

        obs[idx] = (cnt_509 % 4) / 3.0
        obs[idx + 1] = 1.0 if (509 in sim.p_item_gi_counters and (cnt_509 % 4) == 3) else 0.0
        obs[idx + 2] = 1.0 if (has_404 and trig_404 == 0) else 0.0
        obs[idx + 3] = 1.0 if (has_404 and trig_404 == 0 and sim.turn_cards_used == 1) else 0.0

        c920_in_exhaust = 0.0
        c920_uid: Optional[int] = None
        for c in sim.exhaust_pile:
            if c.cfg.base_id == 919:
                c920_in_exhaust = 1.0
                c920_uid = c.uid
                break
        if c920_uid is not None:
            c920_eff_mod = (sim.card_effect_counters.get(c920_uid, 0) % 3) / 2.0
            c920_turn_used = 1.0 if sim.card_turn_counters.get(c920_uid, 0) > 0 else 0.0
        else:
            c920_eff_mod = 0.0
            c920_turn_used = 0.0

        c920_recasts_left = 0.0
        t127_active = 0.0
        t129_active = 0.0
        t181_active = 0.0
        t305_count = 0.0
        t778_count = 0.0
        t778_max_ttl = 0.0
        t521_left = 0.0
        t617_left = 0.0
        d25_count = 0.0
        d25_max_ttl = 0.0

        all_inst = sim._all_instances
        for tr in sim.triggers:
            if tr.limit is not None and tr.triggers_count >= tr.limit:
                continue
            if tr.ttl is not None and tr.ttl < 0:
                continue
            if tr.owner_uid == -25 and tr.phase == "activeCardUsed":
                d25_count += 1.0
                ttl_rem = float(tr.ttl + 1) if tr.ttl is not None else 1.0
                if ttl_rem > d25_max_ttl:
                    d25_max_ttl = ttl_rem
                continue

            owner = all_inst.get(tr.owner_uid)
            if owner is None:
                continue
            bid = owner.cfg.base_id
            phase = tr.phase
            if bid == 919 and phase == "goodImpressionTurnsIncreased" and tr.limit is not None:
                c920_recasts_left = max(c920_recasts_left, float(tr.limit - tr.triggers_count))
            elif bid == 126 and phase == "mentalCardUsed":
                t127_active = 1.0
            elif bid == 128 and phase == "mentalCardUsed":
                t129_active = 1.0
            elif bid == 180 and phase == "endOfTurn":
                t181_active = 1.0
            elif bid == 304 and phase == "cardUsed":
                t305_count += 1.0
            elif bid == 777 and phase == "goodImpressionTurnsIncreased":
                t778_count += 1.0
                ttl_rem = float(tr.ttl + 1) if tr.ttl is not None else 5.0
                if ttl_rem > t778_max_ttl:
                    t778_max_ttl = ttl_rem
            elif bid == 520 and phase == "startOfTurn":
                lim = tr.limit if tr.limit is not None else 4
                t521_left = max(t521_left, float(lim - tr.triggers_count))
            elif bid == 616 and phase == "afterStartOfTurn":
                lim = tr.limit if tr.limit is not None else 3
                t617_left = max(t617_left, float(lim - tr.triggers_count))

        obs[idx + 4] = c920_in_exhaust
        obs[idx + 5] = c920_eff_mod
        obs[idx + 6] = c920_turn_used
        obs[idx + 7] = c920_recasts_left / 2.0
        obs[idx + 8] = t127_active
        obs[idx + 9] = t129_active
        obs[idx + 10] = t181_active
        obs[idx + 11] = t305_count / 2.0
        obs[idx + 12] = t778_count / 2.0
        obs[idx + 13] = t778_max_ttl / 5.0
        obs[idx + 14] = t521_left / 4.0
        obs[idx + 15] = t617_left / 3.0
        obs[idx + 16] = d25_count / 3.0
        obs[idx + 17] = d25_max_ttl / 2.0
        obs[idx + 18] = sum(v * (t + (1 if f else 0)) for v, t, f in sim.timed_gi_effect_buffs) / 15.0

        seen_extra_bids: set[int] = set()
        total_rem_extra = 0
        mem_exiles_avail = 0
        troubles_in_active = 0
        for zone in (sim.hand, sim.draw_pile, sim.discard_pile):
            for c in zone:
                if sim.has_unused_memory_extra_use(c):
                    bid = c.cfg.base_id
                    if bid not in seen_extra_bids:
                        seen_extra_bids.add(bid)
                        lim = int(
                            getattr(c.cfg, "memory_extra_use_limit", 0)
                            or (1 if c.cfg.free_extra_use_first_play else 0)
                        )
                        total_rem_extra += max(0, lim - int(sim.memory_extra_use_counts.get(bid, 0) or 0))
                if sim.has_unused_memory_exile(c):
                    mem_exiles_avail += 1
                if c.cfg.card_type == "trouble" or c.cfg.base_id == 23:
                    troubles_in_active += 1

        has_dom_free_in_hand = 1.0 if any(sim.is_safe_dominating_free_card(c) for c in sim.hand) else 0.0
        has_exile_ready_in_hand = (
            1.0
            if (
                sim.has_active_troubles()
                and any(sim.can_play_card(c) and sim.has_unused_memory_exile(c) for c in sim.hand)
            )
            else 0.0
        )
        has_mem_extra_ready_in_hand = (
            1.0 if any(sim.can_play_card(c) and sim.has_unused_memory_extra_use(c) for c in sim.hand) else 0.0
        )

        obs[idx + 19] = min(1.0, total_rem_extra / 4.0)
        obs[idx + 20] = mem_exiles_avail / 3.0
        obs[idx + 21] = troubles_in_active / 2.0
        obs[idx + 22] = has_dom_free_in_hand
        obs[idx + 23] = has_exile_ready_in_hand
        obs[idx + 24] = has_mem_extra_ready_in_hand
        econ = sim.active_motivation_economy()
        obs[idx + 25] = econ["spenders_total_cost"] / 15.0
        obs[idx + 26] = econ["gainers_count"] / 6.0
        obs[idx + 27] = econ["unplayable_mot_in_hand"] / 4.0
        obs[idx + 28] = min(
            1.5,
            (mot + (sim.turns_remaining * 1.2 if t129_active > 0.5 else 0.0))
            / max(1.0, float(econ["spenders_total_cost"])),
        )
        idx += 29

        # 3b. Sense core buffs (8 dims: goodConditionTurns / perfectConditionTurns / concentration)
        gc = int(buffs.get("goodConditionTurns", 0) or 0)
        pc = int(buffs.get("perfectConditionTurns", 0) or 0)
        conc = int(buffs.get("concentration", 0) or 0)
        obs[idx] = gc / 30.0
        obs[idx + 1] = 1.0 if gc >= 4 else 0.0
        obs[idx + 2] = 1.0 if gc >= 8 else 0.0
        obs[idx + 3] = 1.0 if gc >= 12 else 0.0
        obs[idx + 4] = pc / 10.0
        obs[idx + 5] = 1.0 if pc >= 1 else 0.0
        obs[idx + 6] = conc / 40.0
        obs[idx + 7] = 1.0 if conc >= 5 else 0.0
        idx += 8

        # 4. Drink counts
        for dr in sim.available_drinks:
            d_pos = self._drink_idx.get(dr.drink_id)
            if d_pos is not None:
                obs[idx + d_pos] += 1.0 / 3.0
        idx += self._num_drinks

        # 5. Card zone multisets (hand_playable, hand_unplayable, draw, discard, exhaust)
        # Iterate sequentially without encoding draw_pile permutation order.
        nc = self._num_cards
        hp_off = idx
        hu_off = idx + nc
        dr_off = idx + 2 * nc
        di_off = idx + 3 * nc
        ex_off = idx + 4 * nc

        for c in sim.hand:
            slot = self.codec.card_slot_index(c.cfg)
            if 0 <= slot < nc:
                if sim.can_play_card(c):
                    obs[hp_off + slot] += 1.0
                else:
                    obs[hu_off + slot] += 1.0

        for c in sim.draw_pile:
            slot = self.codec.card_slot_index(c.cfg)
            if 0 <= slot < nc:
                obs[dr_off + slot] += 1.0

        for c in sim.discard_pile:
            slot = self.codec.card_slot_index(c.cfg)
            if 0 <= slot < nc:
                obs[di_off + slot] += 1.0

        for c in sim.exhaust_pile:
            slot = self.codec.card_slot_index(c.cfg)
            if 0 <= slot < nc:
                obs[ex_off + slot] += 1.0

        return obs


class GakumasBattleEnv:
    """Gym-style headless battle environment with canonical action masking."""

    def __init__(
        self,
        config: BattleConfig,
        db: Optional[GakumasDB] = None,
        *,
        score_scale: float = 1_000_000.0,
        record_logs: bool = False,
        prune_dominated_free_actions: bool = True,
        motivation_cost_weight: float = 0.0,
        finisher_play_max_turns_remaining: int = 2,
    ):
        self.config = config
        self._owns_db = db is None
        self.db = db or GakumasDB("gakumas_library.db")
        self.score_scale = float(score_scale)
        self.record_logs = record_logs
        self.prune_dominated_free_actions = bool(prune_dominated_free_actions)
        self.motivation_cost_weight = float(motivation_cost_weight)
        self.finisher_play_max_turns_remaining = max(0, int(finisher_play_max_turns_remaining))
        self._prev_mot_potential: float = 0.0
        self._card_cfg_cache: Dict[Tuple[int, Optional[bool], bool, bool], CardConfig] = {}

        self.codec = GakumasActionCodec(
            config,
            db=self.db,
            finisher_play_max_turns_remaining=self.finisher_play_max_turns_remaining,
        )
        self.encoder = GakumasFeatureEncoder(self.codec)
        self.sim: Optional[HeadlessBattleSimulator] = None

    def _motivation_potential(self, sim: HeadlessBattleSimulator) -> float:
        """Zero-terminal PBRS potential encoding Motivation economy health across turns."""
        if self.motivation_cost_weight <= 0.0 or sim.turns_remaining <= 0:
            return 0.0
        econ = sim.active_motivation_economy()
        if econ["spenders_total_cost"] <= 0:
            return 0.0
        rem_t = min(float(sim.turns_remaining), 8.0)
        target_mot = min(10.0, 2.5 + 0.75 * rem_t)
        mot_eff = float(sim.buffs.get("motivation", 0) or 0) + (4.0 if sim.has_motivation_engine() else 0.0)
        deficit_frac = min(1.0, mot_eff / max(1.0, target_mot)) - 1.0
        brick_pen = -0.25 * min(2, econ["unplayable_mot_in_hand"])
        return self.motivation_cost_weight * (rem_t / 8.0) * (deficit_frac + brick_pen)

    @property
    def obs_dim(self) -> int:
        return self.encoder.obs_dim

    @property
    def num_actions(self) -> int:
        return self.codec.num_actions

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def __enter__(self) -> "GakumasBattleEnv":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def reset(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        self.sim = HeadlessBattleSimulator(
            self.config,
            db=self.db,
            seed=seed,
            record_logs=self.record_logs,
            card_cfg_cache=self._card_cfg_cache,
        )
        self._prev_mot_potential = self._motivation_potential(self.sim)
        return self.encoder.encode(self.sim), self.codec.action_mask(
            self.sim, prune_dominated=self.prune_dominated_free_actions
        )

    def step(self, action_id: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("Environment must be reset() before step()")
        prev_score = self.sim.score
        kind, act_idx = self.codec.resolve_action(self.sim, action_id)
        self.sim.step(kind, act_idx)
        done = self.sim.turns_remaining <= 0
        new_pot = self._motivation_potential(self.sim)
        mot_shaping = new_pot - self._prev_mot_potential
        self._prev_mot_potential = new_pot
        reward = (self.sim.score - prev_score) / self.score_scale + mot_shaping
        obs = self.encoder.encode(self.sim)
        mask = self.codec.action_mask(self.sim, prune_dominated=self.prune_dominated_free_actions)
        econ = self.sim.active_motivation_economy()
        info = {
            "score": self.sim.score,
            "turn": self.sim.turn,
            "mot_shaping": mot_shaping,
            "motivation": int(self.sim.buffs.get("motivation", 0) or 0),
            "unplayable_mot_in_hand": econ["unplayable_mot_in_hand"],
        }
        return obs, mask, reward, done, info
