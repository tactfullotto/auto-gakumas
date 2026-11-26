"""Plan-independent battle heuristics and tactical constraints for Gakumas (Sense / Logic / Anomaly)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from demo.sim.battle_simulator import CardConfig, CardInstance, DrinkConfig, HeadlessBattleSimulator


class BasePlanBattleRules:
    """Common/shared drink and card classification shared by all produce plans."""

    plan_name: str = "common"

    @staticmethod
    def is_trouble_card(card_cfg: Any) -> bool:
        """Return True for trouble/interference cards (e.g. #23 眠気)."""
        return card_cfg.card_type == "trouble" or card_cfg.base_id == 23 or card_cfg.card_id == 23

    @staticmethod
    def is_score_buff_drink(drink: Any) -> bool:
        """Return True for common drinks granting timed score-gain multiplier (`setScoreBuff`, e.g. #12, #14)."""
        return drink.drink_id in (12, 14) or "setscorebuff(" in (drink.dsl_actions or "").lower()

    @staticmethod
    def score_buff_drink_duration(drink: Any) -> int:
        """Return timed turn duration for `setScoreBuff(amount, turns)` drinks (#12=3t, #14=5t)."""
        dsl_l = (drink.dsl_actions or "").lower()
        m_dur = re.search(r"setscorebuff\([^,]+,\s*(\d+)\)", dsl_l)
        if m_dur:
            return max(1, int(m_dur.group(1)))
        if drink.drink_id == 12:
            return 3
        if drink.drink_id == 14:
            return 5
        return 0

    def is_tutor_drink(self, drink: Any) -> bool:
        """Return True for plan-specific tutor drinks moving a chosen card from draw/discard to hand."""
        return False

    def is_hand_upgrade_drink(self, drink: Any) -> bool:
        """Return True for plan-specific drinks that upgrade the current hand."""
        return False

    def is_active_retrigger_drink(self, drink: Any) -> bool:
        """Return True for plan-specific drinks that grant an Active-only card effect retrigger."""
        return False

    def is_late_scaling_finisher(self, cfg: Any) -> bool:
        """Return True for plan-specific limit=1 finishers scaling with total cardsUsed."""
        return False

    def is_resource_zeroing_finisher(self, cfg: Any) -> bool:
        """Return True for plan-specific limit=1 finishers that scale on a stat and immediately zero it."""
        return False

    def is_double_effect_enabler(self, cfg: Any) -> bool:
        """Return True for plan-specific cards granting `doubleCardEffectCards+=`."""
        return False

    def is_timed_score_buff_card(self, cfg: Any) -> bool:
        """Return True for plan-specific single-use cards granting timed `setScoreBuff`."""
        return False

    def is_kimegao_burst_card(self, cfg: Any) -> bool:
        """Return True for Logic-specific `#777/#778` burst cards."""
        return False

    def has_unexhausted_kimegao_card(self, sim: "HeadlessBattleSimulator") -> bool:
        """Return True if an unexhausted `#777/#778` burst card exists in the deck (Logic only)."""
        return False

    def is_senobi_delayed_burst_card(self, cfg: Any) -> bool:
        """Return True for Logic-specific `#771/#772` delayed genki-scaling burst cards."""
        return False

    def has_unexhausted_senobi_card(self, sim: "HeadlessBattleSimulator") -> bool:
        """Return True if an unexhausted `#771/#772` burst card exists in the deck (Logic only)."""
        return False

    def drink_draw_count(self, drink: Any) -> int:
        """Count immediate cards added/drawn to hand by a drink (ignoring delayed `at:` triggers)."""
        dsl_l = (drink.dsl_actions or "").lower()
        if self.is_tutor_drink(drink) or "moveselectedtohand" in dsl_l:
            return 1
        top_part = re.split(r"\bat\s*:", re.sub(r"\{[^}]*\}", "", dsl_l), maxsplit=1)[0]
        total = sum(int(m) for m in re.findall(r"drawcard\s*\(\s*(\d+)\s*\)", top_part))
        rem_top = re.sub(r"drawcard\s*\([^)]*\)", "", top_part)
        total += len(re.findall(r"\bdrawcard\b", rem_top))
        return total

    def is_pure_extra_use_drink(self, drink: Any) -> bool:
        """Return True for pure +card-use common drinks with no side effects (e.g. #18 初星ホエイプロテイン)."""
        if drink.drink_id == 18:
            return True
        dsl_l = (drink.dsl_actions or "").lower()
        if "cardusesremaining+=" not in dsl_l:
            return False
        if (
            self.is_tutor_drink(drink)
            or self.is_score_buff_drink(drink)
            or self.is_active_retrigger_drink(drink)
            or "doublecardeffectcards" in dsl_l
            or "moveselectedtohand" in dsl_l
            or self.drink_draw_count(drink) > 0
        ):
            return False
        rem = re.sub(r"cardusesremaining\s*\+=\s*\d+", "", dsl_l)
        rem = re.sub(r"[\s;]+", "", rem)
        return rem == ""

    def should_hold_action(
        self,
        sim: "HeadlessBattleSimulator",
        *,
        card_cfg: Optional["CardConfig"] = None,
        drink_cfg: Optional["DrinkConfig"] = None,
        fin_window: int = 0,
        has_813: Optional[bool] = None,
        has_zeroing: Optional[bool] = None,
        has_kimegao: Optional[bool] = None,
        has_senobi: Optional[bool] = None,
    ) -> bool:
        """Hold plan-specific burst cards/drinks until their optimal window."""
        return False

    def should_forbid_double_on_card(
        self,
        sim: "HeadlessBattleSimulator",
        card_cfg: "CardConfig",
        *,
        has_other_double_target: bool = True,
    ) -> bool:
        """Prevent wasting armed 2x on a plan's resource-zeroing finisher."""
        return False

    def can_use_active_retrigger_drink_now(
        self,
        sim: "HeadlessBattleSimulator",
        drink: "DrinkConfig",
    ) -> bool:
        """Permit plan-owned Active-retrigger drinks only when valid; disallow Logic #25 on non-Logic plans."""
        dsl_l = (drink.dsl_actions or "").lower()
        if drink.drink_id == 25 or "doublecardeffectcards" in dsl_l:
            return False
        return True

    def can_drink25_combo_kimegao_now(
        self,
        sim: "HeadlessBattleSimulator",
        drink: "DrinkConfig",
        kimegao_inst: Optional["CardInstance"] = None,
    ) -> bool:
        """Return True when Logic #25 can immediately combo with an affordable `#777/#778`."""
        return False

    def has_ready_drink25_kimegao_combo(self, sim: "HeadlessBattleSimulator") -> bool:
        """Return True if an armed-ready Logic `#25 -> #777/#778` combo exists in hand."""
        return False

    def can_drink25_combo_senobi_now(
        self,
        sim: "HeadlessBattleSimulator",
        drink: "DrinkConfig",
        senobi_inst: Optional["CardInstance"] = None,
    ) -> bool:
        """Return True when Logic #25 can immediately combo with an affordable `#771/#772`."""
        return False

    def has_ready_drink25_senobi_combo(self, sim: "HeadlessBattleSimulator") -> bool:
        """Return True if an armed-ready Logic `#25 -> #771/#772` combo exists in hand."""
        return False

    def narrow_armed_or_combo_action_map(
        self,
        sim: "HeadlessBattleSimulator",
        action_map: Dict[int, Tuple[str, int]],
        *,
        fin_window: int = 0,
        end_turn_action_id: int = -1,
    ) -> Tuple[Optional[Dict[int, Tuple[str, int]]], bool]:
        """Return `(filtered_action_map_or_None, early_return_flag)` for plan-specific armed/combo states."""
        return None, False

    def narrow_armed_or_combo_root_legal(
        self,
        sim: "HeadlessBattleSimulator",
        root_legal: List[int],
        root_map: Dict[int, Tuple[str, int]],
        *,
        fin_window: int = 0,
        end_turn_action_id: int = -1,
    ) -> Tuple[Optional[List[int]], bool]:
        """Return `(filtered_root_legal_or_None, handled_armed_active)` for SO-ISMCTS root pruning."""
        return None, False

    def apply_ismcts_priors(
        self,
        sim: "HeadlessBattleSimulator",
        legal_actions: List[int],
        action_map: Dict[int, Tuple[str, int]],
        priors: Any,
        cfg: Any,
    ) -> bool:
        """Optionally boost plan-specific prior probabilities; return True if modified."""
        return False


def _should_forbid_zeroing_double(
    sim: "HeadlessBattleSimulator",
    rules: BasePlanBattleRules,
    card_cfg: "CardConfig",
    *,
    has_other_double_target: bool = True,
) -> bool:
    """Shared guard against wasting an armed 2x buff/retrigger on a resource-zeroing finisher."""
    if not rules.is_resource_zeroing_finisher(card_cfg) or card_cfg.rarity == "L":
        return False
    is_2x_target = sim.double_card_effect_cards > 0 or (
        sim.has_armed_active_retrigger(same_turn_only=True) and card_cfg.card_type == "active"
    )
    if not is_2x_target:
        return False
    if sim.turns_remaining > 1:
        return True
    if not has_other_double_target:
        return False
    if sim.card_uses_remaining >= 2:
        return True
    return any(
        sim.can_play_card(inst)
        and sim.is_free_action_card(inst)
        and not rules.is_trouble_card(inst.cfg)
        and not rules.is_resource_zeroing_finisher(inst.cfg)
        and inst.cfg.rarity != "L"
        and (sim.double_card_effect_cards > 0 or inst.cfg.card_type == "active")
        for inst in sim.hand
    )


class SenseBattleRules(BasePlanBattleRules):
    """Sense-plan (センス / 好調・集中) independent cards, drinks, and battle rules."""

    plan_name: str = "sense"

    def is_late_scaling_finisher(self, cfg: Any) -> bool:
        return cfg.limit_count == 1 and (
            cfg.base_id == 813 or "cardsUsed" in (cfg.dsl_actions or "")
        )

    def is_resource_zeroing_finisher(self, cfg: Any) -> bool:
        if cfg.base_id == 618:
            return True
        if cfg.limit_count != 1:
            return False
        dsl = cfg.dsl_actions or ""
        if "score+=" not in dsl:
            return False
        return bool(re.search(r"\b(goodConditionTurns|concentration)\s*=\s*0\b", dsl))

    def is_double_effect_enabler(self, cfg: Any) -> bool:
        return cfg.base_id == 162 or "doubleCardEffectCards+=" in (cfg.dsl_actions or "")

    def is_timed_score_buff_card(self, cfg: Any) -> bool:
        if cfg.base_id == 775:
            return True
        if cfg.limit_count == 1:
            return bool(re.search(r"setscorebuff\([^,]+,\s*(\d+)\)", (cfg.dsl_actions or "").lower()))
        return False

    def is_tutor_drink(self, drink: Any) -> bool:
        return drink.drink_id == 22 or "moveselectedtohand" in (drink.dsl_actions or "").lower()

    def is_hand_upgrade_drink(self, drink: Any) -> bool:
        return drink.drink_id == 21 or "upgradehand" in (drink.dsl_actions or "").lower()

    def should_forbid_double_on_card(
        self,
        sim: "HeadlessBattleSimulator",
        card_cfg: "CardConfig",
        *,
        has_other_double_target: bool = True,
    ) -> bool:
        return _should_forbid_zeroing_double(sim, self, card_cfg, has_other_double_target=has_other_double_target)

    def should_hold_action(
        self,
        sim: "HeadlessBattleSimulator",
        *,
        card_cfg: Optional["CardConfig"] = None,
        drink_cfg: Optional["DrinkConfig"] = None,
        fin_window: int = 0,
        has_813: Optional[bool] = None,
        has_zeroing: Optional[bool] = None,
        has_kimegao: Optional[bool] = None,
        has_senobi: Optional[bool] = None,
    ) -> bool:
        # Hold Sense zeroing finisher (#618/#619) until final turn (turns_remaining == 1).
        if card_cfg is not None and self.is_resource_zeroing_finisher(card_cfg) and sim.turns_remaining > 1:
            return True
        if fin_window <= 0:
            return False
        has_fin = has_813 if has_813 is not None else sim.has_unexhausted_card_family(813)
        has_zfin = has_zeroing if has_zeroing is not None else sim.has_unexhausted_zeroing_finisher()
        has_any_fin = has_fin or has_zfin
        # In a #813/#814 deck, armed 2x unlocks immediate burst (#813/#814, #22 tutor, #12/#14).
        # In a #618/#619-only deck, early #162/#163 2x must not trigger #22/#12/#14/#775 prematurely.
        if sim.double_card_effect_cards > 0 and (has_fin or not has_zfin):
            return False
        if card_cfg is not None:
            if sim.double_card_effect_cards == 0 and sim.turns_remaining > fin_window:
                if self.is_late_scaling_finisher(card_cfg):
                    return True
                if has_fin and self.is_double_effect_enabler(card_cfg):
                    return True
            if has_any_fin and self.is_timed_score_buff_card(card_cfg) and sim.turns_remaining > max(3, fin_window + 1):
                return True
        if drink_cfg is not None and has_any_fin:
            if self.is_tutor_drink(drink_cfg) and sim.turns_remaining > fin_window:
                return True
            if self.is_score_buff_drink(drink_cfg):
                dur = self.score_buff_drink_duration(drink_cfg)
                if dur > 0 and sim.turns_remaining > dur:
                    return True
        return False

    def narrow_armed_or_combo_action_map(
        self,
        sim: "HeadlessBattleSimulator",
        action_map: Dict[int, Tuple[str, int]],
        *,
        fin_window: int = 0,
        end_turn_action_id: int = -1,
    ) -> Tuple[Optional[Dict[int, Tuple[str, int]]], bool]:
        if sim.double_card_effect_cards <= 0:
            return None, False
        fin_aids = {
            aid
            for aid, (kind, idx) in action_map.items()
            if kind == "card" and self.is_late_scaling_finisher(sim.hand[idx].cfg)
        }
        if fin_aids:
            return (
                {aid: v for aid, v in action_map.items() if aid in fin_aids or v[0] == "drink"},
                True,
            )
        tutor_aids = {
            aid
            for aid, (kind, idx) in action_map.items()
            if kind == "drink" and self.is_tutor_drink(sim.available_drinks[idx])
        }
        if (
            tutor_aids
            and len(sim.hand) < sim.MAX_HAND_SIZE
            and any(c.cfg.base_id == 813 for z in (sim.draw_pile, sim.discard_pile) for c in z)
        ):
            return (
                {
                    aid: v
                    for aid, v in action_map.items()
                    if aid in tutor_aids or (v[0] == "drink" and self.is_score_buff_drink(sim.available_drinks[v[1]]))
                },
                True,
            )
        if (
            sim.turns_remaining > 1
            and sim.turns_remaining <= max(2, fin_window)
            and len(sim.draw_pile) <= 3
            and any(c.cfg.base_id == 813 for c in sim.draw_pile)
            and end_turn_action_id in action_map
        ):
            return (
                {
                    aid: v
                    for aid, v in action_map.items()
                    if aid == end_turn_action_id or v[0] == "drink"
                },
                True,
            )
        return None, False

    def narrow_armed_or_combo_root_legal(
        self,
        sim: "HeadlessBattleSimulator",
        root_legal: List[int],
        root_map: Dict[int, Tuple[str, int]],
        *,
        fin_window: int = 0,
        end_turn_action_id: int = -1,
    ) -> Tuple[Optional[List[int]], bool]:
        if sim.double_card_effect_cards <= 0:
            return None, False
        fin_root = {
            a
            for a in root_legal
            if root_map.get(a, ("", -1))[0] == "card"
            and self.is_late_scaling_finisher(sim.hand[root_map[a][1]].cfg)
        }
        if fin_root:
            pruned = [a for a in root_legal if a in fin_root or root_map.get(a, ("", -1))[0] == "drink"]
            if pruned:
                return pruned, True
        if (
            len(sim.hand) < sim.MAX_HAND_SIZE
            and any(c.cfg.base_id == 813 for z in (sim.draw_pile, sim.discard_pile) for c in z)
        ):
            tutor_root = {
                a
                for a in root_legal
                if root_map.get(a, ("", -1))[0] == "drink"
                and self.is_tutor_drink(sim.available_drinks[root_map[a][1]])
            }
            if tutor_root:
                pruned = [
                    a
                    for a in root_legal
                    if a in tutor_root
                    or (root_map.get(a, ("", -1))[0] == "drink" and self.is_score_buff_drink(sim.available_drinks[root_map[a][1]]))
                ]
                return pruned, True
        if (
            sim.turns_remaining > 1
            and sim.turns_remaining <= max(2, fin_window)
            and len(sim.draw_pile) <= 3
            and any(c.cfg.base_id == 813 for c in sim.draw_pile)
            and end_turn_action_id in root_legal
        ):
            pruned = [a for a in root_legal if a == end_turn_action_id or root_map.get(a, ("", -1))[0] == "drink"]
            if pruned:
                return pruned, True
        return None, False


class LogicBattleRules(BasePlanBattleRules):
    """Logic-plan (ロジック / 好印象・やる気) independent cards, drinks, and battle rules."""

    plan_name: str = "logic"
    KIMEGAO_PLAY_MAX_TURNS_REMAINING: int = 5
    SENOBI_PLAY_MIN_TURNS_REMAINING: int = 5
    SENOBI_PLAY_MAX_TURNS_REMAINING: int = 6

    def is_kimegao_burst_card(self, cfg: Any) -> bool:
        return cfg.base_id == 777 or cfg.card_id in (777, 778)

    def is_senobi_delayed_burst_card(self, cfg: Any) -> bool:
        return cfg.base_id == 771 or cfg.card_id in (771, 772)

    def is_active_retrigger_drink(self, drink: Any) -> bool:
        return drink.drink_id == 25 or ("doublecardeffectcards" in (drink.dsl_actions or "").lower())

    def is_resource_zeroing_finisher(self, cfg: Any) -> bool:
        if cfg.base_id == 152:
            return True
        if cfg.limit_count != 1:
            return False
        dsl = cfg.dsl_actions or ""
        if "score+=" not in dsl:
            return False
        return bool(re.search(r"\b(genki|goodImpressionTurns)\s*=\s*0\b", dsl))

    def has_unexhausted_kimegao_card(self, sim: "HeadlessBattleSimulator") -> bool:
        return sim.has_unexhausted_card_family(777)

    def has_unexhausted_senobi_card(self, sim: "HeadlessBattleSimulator") -> bool:
        return sim.has_unexhausted_card_family(771)

    @staticmethod
    def _has_consumed_genki_engine(sim: "HeadlessBattleSimulator") -> bool:
        if getattr(sim.config, "p_idol_id", None) == 136:
            return True
        if any(getattr(it, "trigger_every_n_turns_eot", 0) > 0 for it in (sim.config.p_items or ())):
            return True
        for z in (sim.hand, sim.draw_pile, sim.discard_pile):
            if any(c.cfg.base_id in (799, 782) for c in z):
                return True
        all_inst = sim._all_instances
        for tr in sim.triggers:
            if tr.limit is not None and tr.triggers_count >= tr.limit:
                continue
            if tr.ttl is not None and tr.ttl < 0:
                continue
            owner = all_inst.get(tr.owner_uid)
            if owner is not None and owner.cfg.base_id in (799, 782):
                return True
        return False

    def should_forbid_double_on_card(
        self,
        sim: "HeadlessBattleSimulator",
        card_cfg: "CardConfig",
        *,
        has_other_double_target: bool = True,
    ) -> bool:
        if _should_forbid_zeroing_double(sim, self, card_cfg, has_other_double_target=has_other_double_target):
            return True
        if (
            self.is_senobi_delayed_burst_card(card_cfg)
            and sim.turns_remaining < self.SENOBI_PLAY_MIN_TURNS_REMAINING
            and has_other_double_target
        ):
            return True
        return False

    def _can_use_active_retrigger_for_candidates(
        self,
        sim: "HeadlessBattleSimulator",
        drink: "DrinkConfig",
        candidates: List["CardInstance"],
    ) -> bool:
        if not candidates or sim.has_armed_active_retrigger():
            return False
        if any(sim.is_safe_dominating_free_card(c) and c.cfg.card_type != "active" for c in sim.hand):
            return False
        if (
            sim.card_uses_remaining <= 0
            or sim.buffs.get("noCardUseTurns", 0) > 0
            or sim.buffs.get("noActiveTurns", 0) > 0
        ):
            return False

        dsl_l = (drink.dsl_actions or "").lower()
        m_stam = re.search(r"fixedstamina-=(\d+)", dsl_l)
        stam_drop = int(m_stam.group(1)) if m_stam else (2 if drink.drink_id == 25 else 0)
        adds_double_cost = ("doublecostturns+=" in dsl_l) or (drink.drink_id == 25)
        post_stam = max(0, sim.stamina - stam_drop)

        orig_double_cost = int(sim.buffs.get("doubleCostTurns", 0) or 0)
        will_add_double = adds_double_cost and int(sim.buffs.get("nullifyDebuff", 0) or 0) <= 0
        try:
            if will_add_double:
                sim.buffs["doubleCostTurns"] = orig_double_cost + 1
            for inst in candidates:
                if inst.cfg.card_type != "active" or inst.cfg.rarity == "L":
                    continue
                if self.is_resource_zeroing_finisher(inst.cfg):
                    continue
                if not sim.can_play_card(inst):
                    continue
                cost_var, eff_amt = sim._effective_cost(inst.cfg)
                if cost_var == "cost":
                    if (sim.genki + post_stam) >= eff_amt:
                        return True
                elif cost_var == "stamina":
                    if post_stam >= eff_amt:
                        return True
                elif int(sim.buffs.get(cost_var, 0) or 0) >= eff_amt:
                    return True
        finally:
            if will_add_double:
                sim.buffs["doubleCostTurns"] = orig_double_cost
        return False

    def can_use_active_retrigger_drink_now(
        self,
        sim: "HeadlessBattleSimulator",
        drink: "DrinkConfig",
    ) -> bool:
        if not self.is_active_retrigger_drink(drink):
            return True
        return self._can_use_active_retrigger_for_candidates(sim, drink, list(sim.hand))

    def can_drink25_combo_kimegao_now(
        self,
        sim: "HeadlessBattleSimulator",
        drink: "DrinkConfig",
        kimegao_inst: Optional["CardInstance"] = None,
    ) -> bool:
        if (
            not self.is_active_retrigger_drink(drink)
            or sim.turns_remaining > self.KIMEGAO_PLAY_MAX_TURNS_REMAINING
            or sim.double_card_effect_cards != 0
        ):
            return False
        if kimegao_inst is not None:
            if not self.is_kimegao_burst_card(kimegao_inst.cfg):
                return False
            cands = [kimegao_inst]
        else:
            cands = [c for c in sim.hand if self.is_kimegao_burst_card(c.cfg)]
        return self._can_use_active_retrigger_for_candidates(sim, drink, cands)

    def has_ready_drink25_kimegao_combo(self, sim: "HeadlessBattleSimulator") -> bool:
        return any(self.can_drink25_combo_kimegao_now(sim, d) for d in sim.available_drinks)

    def can_drink25_combo_senobi_now(
        self,
        sim: "HeadlessBattleSimulator",
        drink: "DrinkConfig",
        senobi_inst: Optional["CardInstance"] = None,
    ) -> bool:
        """Return True when #25 can immediately combo with an affordable `#771/#772` inside its delay:3 landing window (rem 5..6)."""
        if (
            not self.is_active_retrigger_drink(drink)
            or sim.turns_remaining < self.SENOBI_PLAY_MIN_TURNS_REMAINING
            or sim.turns_remaining > self.SENOBI_PLAY_MAX_TURNS_REMAINING
            or sim.double_card_effect_cards != 0
        ):
            return False
        if senobi_inst is not None:
            if not self.is_senobi_delayed_burst_card(senobi_inst.cfg):
                return False
            cands = [senobi_inst]
        else:
            cands = [c for c in sim.hand if self.is_senobi_delayed_burst_card(c.cfg)]
        return self._can_use_active_retrigger_for_candidates(sim, drink, cands)

    def has_ready_drink25_senobi_combo(self, sim: "HeadlessBattleSimulator") -> bool:
        return any(self.can_drink25_combo_senobi_now(sim, d) for d in sim.available_drinks)

    def _in_window_active_retrigger_targets(
        self,
        sim: "HeadlessBattleSimulator",
        active_ids: set[int],
        get_cfg: Any,
    ) -> set[int]:
        """Prioritize `#777/#778` (rem<=5) and `#771/#772` (5<=rem<=6), and drop dead `#771/#772` at rem<=4."""
        if not active_ids:
            return active_ids
        rem = int(sim.turns_remaining)
        if rem < self.SENOBI_PLAY_MIN_TURNS_REMAINING:
            non_dead = {a for a in active_ids if not self.is_senobi_delayed_burst_card(get_cfg(a))}
            if non_dead:
                active_ids = non_dead
        burst_hits = {
            a
            for a in active_ids
            if (rem <= self.KIMEGAO_PLAY_MAX_TURNS_REMAINING and self.is_kimegao_burst_card(get_cfg(a)))
            or (
                self.SENOBI_PLAY_MIN_TURNS_REMAINING <= rem <= self.SENOBI_PLAY_MAX_TURNS_REMAINING
                and self.is_senobi_delayed_burst_card(get_cfg(a))
            )
        }
        return burst_hits or active_ids

    def should_hold_action(
        self,
        sim: "HeadlessBattleSimulator",
        *,
        card_cfg: Optional["CardConfig"] = None,
        drink_cfg: Optional["DrinkConfig"] = None,
        fin_window: int = 0,
        has_813: Optional[bool] = None,
        has_zeroing: Optional[bool] = None,
        has_kimegao: Optional[bool] = None,
        has_senobi: Optional[bool] = None,
    ) -> bool:
        # Hold Logic zeroing finisher (#152/#153) until final turn (turns_remaining == 1),
        # UNLESS the active deck/idol runs a consumedGenki engine (#136 / #799/#800 / #9136 / #782)
        # where mid-battle genki=0 intentionally feeds 80% genki rebound.
        if (
            card_cfg is not None
            and self.is_resource_zeroing_finisher(card_cfg)
            and sim.turns_remaining > 1
            and not self._has_consumed_genki_engine(sim)
        ):
            return True
        # Hold #777/#778 びしっとキメ顔(+) until the final 5 turns (turns_remaining <= 5).
        if (
            card_cfg is not None
            and self.is_kimegao_burst_card(card_cfg)
            and sim.turns_remaining > self.KIMEGAO_PLAY_MAX_TURNS_REMAINING
        ):
            return True
        # Hold #771/#772 せのびの魔法(+) (delay:3) until rem <= 6 so its big genki burst lands on final 2 turns.
        if (
            card_cfg is not None
            and self.is_senobi_delayed_burst_card(card_cfg)
            and sim.turns_remaining > self.SENOBI_PLAY_MAX_TURNS_REMAINING
        ):
            return True
        has_zfin = has_zeroing if has_zeroing is not None else sim.has_unexhausted_zeroing_finisher()
        if drink_cfg is not None:
            if self.is_active_retrigger_drink(drink_cfg):
                has_km = has_kimegao if has_kimegao is not None else self.has_unexhausted_kimegao_card(sim)
                has_sn = has_senobi if has_senobi is not None else self.has_unexhausted_senobi_card(sim)
                if has_sn:
                    if sim.turns_remaining > self.SENOBI_PLAY_MAX_TURNS_REMAINING:
                        return True
                    if (
                        sim.turns_remaining == self.SENOBI_PLAY_MAX_TURNS_REMAINING
                        and sim.count_active_retrigger_drinks() <= 1
                        and not any(self.is_senobi_delayed_burst_card(c.cfg) for c in sim.hand)
                    ):
                        return True
                if has_km:
                    if sim.turns_remaining > self.KIMEGAO_PLAY_MAX_TURNS_REMAINING:
                        if not self.can_drink25_combo_senobi_now(sim, drink_cfg):
                            return True
                    elif (
                        sim.turns_remaining >= 3
                        and sim.count_active_retrigger_drinks() <= 1
                        and not any(self.is_kimegao_burst_card(c.cfg) for c in sim.hand)
                        and not self.can_drink25_combo_senobi_now(sim, drink_cfg)
                    ):
                        return True
            if fin_window > 0 and has_zfin and self.is_score_buff_drink(drink_cfg):
                dur = self.score_buff_drink_duration(drink_cfg)
                if dur > 0 and sim.turns_remaining > dur:
                    return True
        return False

    def narrow_armed_or_combo_action_map(
        self,
        sim: "HeadlessBattleSimulator",
        action_map: Dict[int, Tuple[str, int]],
        *,
        fin_window: int = 0,
        end_turn_action_id: int = -1,
    ) -> Tuple[Optional[Dict[int, Tuple[str, int]]], bool]:
        if sim.has_armed_active_retrigger(same_turn_only=True):
            active_aids = {
                aid
                for aid, (kind, idx) in action_map.items()
                if kind == "card"
                and sim.hand[idx].cfg.card_type == "active"
                and sim.hand[idx].cfg.rarity != "L"
                and not self.is_resource_zeroing_finisher(sim.hand[idx].cfg)
            }
            if active_aids:
                active_aids = self._in_window_active_retrigger_targets(
                    sim,
                    active_aids,
                    lambda a: sim.hand[action_map[a][1]].cfg,
                )
                return (
                    {
                        aid: v
                        for aid, v in action_map.items()
                        if aid in active_aids
                        or (v[0] == "drink" and not self.is_active_retrigger_drink(sim.available_drinks[v[1]]))
                    },
                    True,
                )

        if (
            sim.turns_remaining <= max(self.KIMEGAO_PLAY_MAX_TURNS_REMAINING, self.SENOBI_PLAY_MAX_TURNS_REMAINING)
            and sim.double_card_effect_cards == 0
            and not sim.has_armed_active_retrigger(same_turn_only=True)
        ):
            d25_combo_aids = {
                aid
                for aid, (k, idx) in action_map.items()
                if k == "drink"
                and (
                    self.can_drink25_combo_kimegao_now(sim, sim.available_drinks[idx])
                    or self.can_drink25_combo_senobi_now(sim, sim.available_drinks[idx])
                )
            }
            if d25_combo_aids:
                km_ready = any(
                    self.can_drink25_combo_kimegao_now(sim, sim.available_drinks[action_map[aid][1]])
                    for aid in d25_combo_aids
                )
                sn_ready = any(
                    self.can_drink25_combo_senobi_now(sim, sim.available_drinks[action_map[aid][1]])
                    for aid in d25_combo_aids
                )
                out_map = dict(action_map)
                naked_burst = {
                    aid
                    for aid, (k, idx) in out_map.items()
                    if k == "card"
                    and (
                        (km_ready and self.is_kimegao_burst_card(sim.hand[idx].cfg))
                        or (sn_ready and self.is_senobi_delayed_burst_card(sim.hand[idx].cfg))
                    )
                }
                if naked_burst and len(naked_burst) < len(out_map):
                    for aid in naked_burst:
                        out_map.pop(aid, None)
                if sim.turns_remaining >= 2:
                    narrowed = {
                        aid: v
                        for aid, v in out_map.items()
                        if aid in d25_combo_aids
                        or (v[0] == "card" and sim.is_safe_dominating_free_card(sim.hand[v[1]], playable=True))
                        or (v[0] == "drink" and not self.is_active_retrigger_drink(sim.available_drinks[v[1]]))
                    }
                    if narrowed:
                        return narrowed, True
                return out_map, False
        return None, False

    def narrow_armed_or_combo_root_legal(
        self,
        sim: "HeadlessBattleSimulator",
        root_legal: List[int],
        root_map: Dict[int, Tuple[str, int]],
        *,
        fin_window: int = 0,
        end_turn_action_id: int = -1,
    ) -> Tuple[Optional[List[int]], bool]:
        if sim.has_armed_active_retrigger(same_turn_only=True):
            active_root = {
                a
                for a in root_legal
                if root_map.get(a, ("", -1))[0] == "card"
                and sim.hand[root_map[a][1]].cfg.card_type == "active"
                and sim.hand[root_map[a][1]].cfg.rarity != "L"
                and not self.is_resource_zeroing_finisher(sim.hand[root_map[a][1]].cfg)
            }
            if active_root:
                active_root = self._in_window_active_retrigger_targets(
                    sim,
                    active_root,
                    lambda a: sim.hand[root_map[a][1]].cfg,
                )
                pruned = [
                    a
                    for a in root_legal
                    if a in active_root
                    or (
                        root_map.get(a, ("", -1))[0] == "drink"
                        and not self.is_active_retrigger_drink(sim.available_drinks[root_map[a][1]])
                    )
                ]
                if pruned:
                    return pruned, True

        if (
            sim.turns_remaining <= max(self.KIMEGAO_PLAY_MAX_TURNS_REMAINING, self.SENOBI_PLAY_MAX_TURNS_REMAINING)
            and sim.double_card_effect_cards == 0
            and not sim.has_armed_active_retrigger(same_turn_only=True)
        ):
            d25_combo_root = {
                a
                for a in root_legal
                if root_map.get(a, ("", -1))[0] == "drink"
                and (
                    self.can_drink25_combo_kimegao_now(sim, sim.available_drinks[root_map[a][1]])
                    or self.can_drink25_combo_senobi_now(sim, sim.available_drinks[root_map[a][1]])
                )
            }
            if d25_combo_root:
                km_ready = any(
                    self.can_drink25_combo_kimegao_now(sim, sim.available_drinks[root_map[a][1]])
                    for a in d25_combo_root
                )
                sn_ready = any(
                    self.can_drink25_combo_senobi_now(sim, sim.available_drinks[root_map[a][1]])
                    for a in d25_combo_root
                )
                updated = list(root_legal)
                naked_burst = {
                    a
                    for a in updated
                    if root_map.get(a, ("", -1))[0] == "card"
                    and (
                        (km_ready and self.is_kimegao_burst_card(sim.hand[root_map[a][1]].cfg))
                        or (sn_ready and self.is_senobi_delayed_burst_card(sim.hand[root_map[a][1]].cfg))
                    )
                }
                if naked_burst and len(naked_burst) < len(updated):
                    updated = [a for a in updated if a not in naked_burst]
                if sim.turns_remaining >= 2:
                    combo_pool = [
                        a
                        for a in updated
                        if a in d25_combo_root
                        or (
                            root_map.get(a, ("", -1))[0] == "card"
                            and sim.is_safe_dominating_free_card(sim.hand[root_map[a][1]])
                        )
                        or (
                            root_map.get(a, ("", -1))[0] == "drink"
                            and not self.is_active_retrigger_drink(sim.available_drinks[root_map[a][1]])
                        )
                    ]
                    if combo_pool:
                        return combo_pool, True
                if len(updated) != len(root_legal):
                    return updated, False
        return None, False

    def apply_ismcts_priors(
        self,
        sim: "HeadlessBattleSimulator",
        legal_actions: List[int],
        action_map: Dict[int, Tuple[str, int]],
        priors: Any,
        cfg: Any,
    ) -> bool:
        armed_same_turn = sim.has_armed_active_retrigger(same_turn_only=True)
        armed_kimegao_window = armed_same_turn and sim.turns_remaining <= self.KIMEGAO_PLAY_MAX_TURNS_REMAINING
        armed_senobi_window = (
            armed_same_turn
            and self.SENOBI_PLAY_MIN_TURNS_REMAINING <= sim.turns_remaining <= self.SENOBI_PLAY_MAX_TURNS_REMAINING
        )
        km_floor = float(getattr(cfg, "kimegao_combo_prior_floor", 0.45))
        modified = False
        for a in legal_actions:
            kind, idx = action_map.get(a, ("", -1))
            if kind == "card" and 0 <= idx < len(sim.hand):
                inst = sim.hand[idx]
                is_burst_armed = (armed_kimegao_window and self.is_kimegao_burst_card(inst.cfg)) or (
                    armed_senobi_window and self.is_senobi_delayed_burst_card(inst.cfg)
                )
                if is_burst_armed and float(priors[a]) < km_floor:
                    priors[a] = max(float(priors[a]), km_floor)
                    modified = True
            elif kind == "drink" and 0 <= idx < len(sim.available_drinks):
                dr_i = sim.available_drinks[idx]
                if (
                    self.can_drink25_combo_kimegao_now(sim, dr_i)
                    or self.can_drink25_combo_senobi_now(sim, dr_i)
                ) and float(priors[a]) < km_floor:
                    priors[a] = max(float(priors[a]), km_floor)
                    modified = True
        return modified


class AnomalyBattleRules(BasePlanBattleRules):
    """Anomaly-plan (アノマリー / 強気・温存・全力) independent battle rules."""

    plan_name: str = "anomaly"


COMMON_RULES = BasePlanBattleRules()
SENSE_RULES = SenseBattleRules()
LOGIC_RULES = LogicBattleRules()
ANOMALY_RULES = AnomalyBattleRules()

_PLAN_RULES_MAP: Dict[str, BasePlanBattleRules] = {
    "sense": SENSE_RULES,
    "logic": LOGIC_RULES,
    "anomaly": ANOMALY_RULES,
}


def get_plan_rules(plan: Optional[str]) -> BasePlanBattleRules:
    """Return the independent rule implementation for the active produce plan."""
    return _PLAN_RULES_MAP.get(str(plan or "").strip().lower(), COMMON_RULES)
