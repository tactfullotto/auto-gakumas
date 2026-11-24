"""Standalone AST-driven headless battle simulator for Gakumas."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
import math
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from demo.core.db import GakumasDB


def _safe_ceil(value: float) -> int:
    """Round up with float-precision tolerance matching gakumas-engine safeCeil."""
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return int(rounded)
    return math.ceil(value)


EOT_DECREMENT_FIELDS = (
    "goodImpressionTurns",
    "halfCostTurns",
    "doubleCostTurns",
    "goodConditionTurns",
    "perfectConditionTurns",
    "poorConditionTurns",
)
DEBUFF_FIELDS = frozenset({
    "doubleCostTurns",
    "doubleBuffCostTurns",
    "costIncrease",
    "nullifyGenkiTurns",
    "noActiveTurns",
    "noMentalTurns",
    "uneaseTurns",
    "poorConditionTurns",
    "lockStanceTurns",
    "noCardUseTurns",
})
DEBUFF_SPECIAL_ACTIONS = frozenset({"setScoreDebuff", "decreaseFullPowerCharge"})
CONSUMABLE_BUFF_FIELDS = frozenset({
    "goodConditionTurns",
    "perfectConditionTurns",
    "concentration",
    "goodImpressionTurns",
    "motivation",
    "fullPowerCharge",
})
DIRECT_CHANGE_PHASES = frozenset({
    "processCard",
    "processCost",
    "cardMovedToHand",
    "cardMovedToHeld",
})


@dataclass
class TurnMultiplierSpec:
    """Single-turn stage attribute and score multiplier."""

    attribute: str
    multiplier: float


@dataclass
class CardConfig:
    """Deck card configuration with optional H.I.F memory/custom battle hooks."""

    card_id: int
    name: str
    base_id: int
    upgraded_id: Optional[int]
    card_type: str
    rarity: str
    upgraded: bool
    limit_count: Optional[int]
    force_initial_hand: bool
    p_idol_id: Optional[int]
    is_signature: bool
    dsl_conditions: str = ""
    dsl_cost: str = ""
    dsl_actions: str = ""
    dsl_effects: str = ""
    parsed_effects: Dict[str, Any] = field(default_factory=dict)
    exiles_trouble_on_play: bool = False
    free_extra_use_first_play: bool = False
    memory_extra_use_limit: int = 0
    memory_draw_on_play: int = 0
    customizations: Dict[int, int] = field(default_factory=dict)
    custom_modifiers: Dict[str, Any] = field(default_factory=dict)
    source_type: str = "produce"


@dataclass
class DrinkConfig:
    """Held P-drink configuration loaded from SQLite."""

    drink_id: int
    name: str
    rarity: str
    description: str
    dsl_actions: str = ""
    parsed_effects: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PItemConfig:
    """Equipped P-item passive configuration (stage battle or produce-phase)."""

    item_id: int
    name: str
    rarity: str = "SSR"
    mode: str = "stage"  # "stage" (战中触发) or "produce" (培育期/入场前生效)
    source_type: str = "support"
    description: str = ""
    dsl_effects: str = ""
    trigger_every_n_direct_gi: int = 0
    grant_extra_gi_eot_triggers: int = 0
    grant_extra_gi_eot_ttl: int = 0
    grant_gi_effect_buff: float = 0.0
    grant_gi_effect_ttl: Optional[int] = None
    stamina_cost_on_trigger: int = 0
    trigger_on_nth_card_in_turn: int = 0
    grant_indirect_gi: int = 0
    grant_perm_gi_effect_buff: float = 0.0
    pad_deck_to_min_cards: int = 0
    max_triggers: Optional[int] = None
    trigger_every_n_gc_cards: int = 0
    require_min_gc: int = 0
    grant_perfect_condition: int = 0
    grant_card_uses: int = 0
    grant_draw_cards: int = 0
    trigger_attr_turn: str = ""
    grant_genki: int = 0


def generate_stage_turn_schedule(
    rng: random.Random,
    *,
    attr_multipliers: Tuple[Tuple[str, float], Tuple[str, float], Tuple[str, float]] = (
        ("Vo", 40.0),
        ("Vi", 35.0),
        ("Da", 20.0),
    ),
    counts: Tuple[int, int, int] = (6, 3, 3),
    no_top_on_penultimate_two: bool = True,
) -> List[TurnMultiplierSpec]:
    """Generate a randomized stage turn schedule matching idol stat hierarchy & constraints."""
    paired = sorted(zip(attr_multipliers, counts), key=lambda item: item[0][1], reverse=True)
    (top_attr, top_mult), top_cnt = paired[0]

    non_top: List[TurnMultiplierSpec] = []
    for (attr, mult), cnt in paired[1:]:
        non_top.extend(TurnMultiplierSpec(attr, mult) for _ in range(cnt))
    rng.shuffle(non_top)

    last_turn = TurnMultiplierSpec(top_attr, top_mult)
    rem_top = [TurnMultiplierSpec(top_attr, top_mult) for _ in range(max(0, top_cnt - 1))]

    if no_top_on_penultimate_two and len(non_top) >= 2:
        penultimate = non_top[:2]
        prefix = rem_top + non_top[2:]
        rng.shuffle(prefix)
        return prefix + penultimate + [last_turn]

    prefix = rem_top + non_top
    rng.shuffle(prefix)
    return prefix + [last_turn]


def _actions_grant_double_card_effect(
    actions: Optional[List[Dict[str, Any]]],
    effects: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Return True if trigger actions or nested effect blocks add doubleCardEffectCards."""
    for act in actions or ():
        if act.get("type") == "assignment" and act.get("lhs") == "doubleCardEffectCards" and act.get("op") == "+=":
            return True
        if act.get("actions") and _actions_grant_double_card_effect(act.get("actions")):
            return True
    for eff in effects or ():
        if _actions_grant_double_card_effect(eff.get("actions"), eff.get("effects")):
            return True
    return False


@dataclass
class BattleConfig:
    """Complete pre-battle configuration for headless simulation."""

    scenario_id: str
    stage_name: str
    p_idol_id: int
    idol_name: str
    plan: str
    idol_stats: Dict[str, int]
    initial_stamina: int
    max_stamina: int
    initial_genki: int = 0
    initial_good_impression: int = 0
    initial_motivation: int = 0
    initial_good_condition: int = 0
    initial_perfect_condition: int = 0
    initial_concentration: int = 0
    initial_full_power_charge: int = 0
    initial_score_buff: float = 0.0
    total_turns: int = 12
    turn_schedule: List[TurnMultiplierSpec] = field(default_factory=list)
    randomize_turn_schedule: bool = False
    turn_schedule_attr_multipliers: Tuple[Tuple[str, float], Tuple[str, float], Tuple[str, float]] = (
        ("Vo", 40.0),
        ("Vi", 35.0),
        ("Da", 20.0),
    )
    turn_schedule_counts: Tuple[int, int, int] = (6, 3, 3)
    deck: List[CardConfig] = field(default_factory=list)
    drinks: List[DrinkConfig] = field(default_factory=list)
    p_items: List[PItemConfig] = field(default_factory=list)
    extra_notes: List[str] = field(default_factory=list)


@dataclass
class CardInstance:
    """Runtime card instance tracked across hand/draw/discard/exhaust."""

    uid: int
    cfg: CardConfig


@dataclass
class ActiveTrigger:
    """Dynamic or passive phased listener registered in battle."""

    owner_name: str
    owner_uid: int
    phase: str
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    effects: List[Dict[str, Any]] = field(default_factory=list)
    limit: Optional[int] = None
    ttl: Optional[int] = None
    delay: Optional[int] = None
    triggers_count: int = 0
    effect_counter: int = 0
    turn_counter: int = 0
    is_passive_card_effect: bool = False
    is_from_exhaust_check: bool = False
    is_card_reservation: bool = False
    grants_double_card_effect: bool = False


@dataclass
class ConditionSnapshot:
    """Shallow pre-phase/pre-action state snapshot for condition evaluation."""

    buffs: Dict[str, int]
    stamina: int
    genki: int
    score: int
    turns_elapsed: int
    turns_remaining: int
    turn_cards_used: int
    cards_used: int
    active_cards_used: int
    card_effect_counters: Dict[int, int]
    card_turn_counters: Dict[int, int]
    zone_uids: Dict[str, set[int]]
    card_rarity: str
    card_type: str
    used_card_id: int
    used_card_base_id: int
    moved_card_id: int
    moved_card_base_id: int
    last_used_card_type: str
    is_direct_effect: bool


@dataclass
class BattleResult:
    """Structured outcome and trace of a single headless simulation."""

    seed: int
    final_score: int
    turns_played: int
    final_stamina: int
    max_stamina: int
    final_genki: int
    final_good_impression: int
    final_motivation: int
    final_gi_effect_buff: float
    exhaust_count: int
    reshuffle_count: int
    cards_played: List[str]
    drinks_used: List[str]
    exhausted_names: List[str]
    turn_logs: List[str]
    zone_counts: Dict[str, int]
    p_item_activations: Dict[str, int] = field(default_factory=dict)
    memory_triggers: List[str] = field(default_factory=list)
    turn_schedule: List[str] = field(default_factory=list)


_GROWTH_VAR_TO_ACTION_LHS: Dict[str, str] = {
    "g.score": "score",
    "g.genki": "genki",
    "g.goodImpressionTurns": "goodImpressionTurns",
    "g.motivation": "motivation",
    "g.goodConditionTurns": "goodConditionTurns",
    "g.perfectConditionTurns": "perfectConditionTurns",
    "g.concentration": "concentration",
    "g.fullPowerCharge": "fullPowerCharge",
    "g.halfCostTurns": "halfCostTurns",
}

_SCALE_VAR_TO_NAME: Dict[str, str] = {
    "g.scoreByGoodImpressionTurns": "goodImpressionTurns",
    "g.scoreByMotivation": "motivation",
    "g.scoreByGenki": "genki",
}


def _reduce_cost_ast(cost_clauses: List[Dict[str, Any]], delta: float, *, cost_only: Optional[bool]) -> None:
    """Reduce matching card cost AST clause(s) in-place by `delta` (clamped >= 0)."""
    if delta <= 0:
        return
    for clause in cost_clauses:
        for act in clause.get("actions") or ():
            if not isinstance(act, dict) or act.get("type") != "assignment":
                continue
            lhs = str(act.get("lhs") or "cost")
            if cost_only is True and lhs != "cost":
                continue
            if cost_only is False and lhs == "cost":
                continue
            rhs = act.get("rhs")
            if isinstance(rhs, dict) and rhs.get("type") == "number":
                cur_val = float(rhs.get("value") or 0)
                new_val = max(0, cur_val - float(delta))
                rhs["value"] = int(new_val) if abs(new_val - round(new_val)) < 1e-9 else round(new_val, 4)
                return


def _bump_action_lhs_ast(action_clauses: List[Dict[str, Any]], target_lhs: str, delta: float) -> None:
    """Increment an existing `target_lhs += ...` AST assignment or append a direct action block."""
    if delta == 0:
        return
    candidates: List[Dict[str, Any]] = []
    # Prefer unconditional immediate blocks first, then any block
    for blk in action_clauses:
        if isinstance(blk, dict) and not blk.get("phase") and not blk.get("conditions"):
            candidates.append(blk)
    for blk in action_clauses:
        if isinstance(blk, dict) and blk not in candidates:
            candidates.append(blk)

    for blk in candidates:
        for act in blk.get("actions") or ():
            if not isinstance(act, dict) or act.get("type") != "assignment":
                continue
            if act.get("lhs") != target_lhs or act.get("op") != "+=":
                continue
            rhs = act.get("rhs")
            if isinstance(rhs, dict) and rhs.get("type") == "number":
                new_val = float(rhs.get("value") or 0) + float(delta)
                rhs["value"] = int(new_val) if abs(new_val - round(new_val)) < 1e-9 else round(new_val, 4)
                return

    num_val = int(delta) if abs(delta - round(delta)) < 1e-9 else round(float(delta), 4)
    action_clauses.append(
        {
            "actions": [
                {
                    "type": "assignment",
                    "lhs": target_lhs,
                    "op": "+=",
                    "rhs": {"type": "number", "value": num_val},
                }
            ]
        }
    )


def _bump_score_multiplier_ast(action_clauses: List[Dict[str, Any]], scale_var: str, delta: float) -> None:
    """Increase `score += <scale_var> * K` multiplier constant by `delta` (or append one)."""
    if delta == 0:
        return
    for blk in action_clauses:
        if not isinstance(blk, dict):
            continue
        for act in blk.get("actions") or ():
            if not isinstance(act, dict) or act.get("type") != "assignment" or act.get("lhs") != "score":
                continue
            rhs = act.get("rhs")
            if not isinstance(rhs, dict):
                continue
            if rhs.get("type") == "identifier" and rhs.get("name") == scale_var:
                act["rhs"] = {
                    "type": "binary",
                    "op": "*",
                    "left": {"type": "identifier", "name": scale_var},
                    "right": {"type": "number", "value": round(1.0 + float(delta), 4)},
                }
                return
            if rhs.get("type") == "binary" and rhs.get("op") == "*":
                left = rhs.get("left") or {}
                right = rhs.get("right") or {}
                if left.get("type") == "identifier" and left.get("name") == scale_var and right.get("type") == "number":
                    right["value"] = round(float(right.get("value") or 0) + float(delta), 4)
                    return
                if right.get("type") == "identifier" and right.get("name") == scale_var and left.get("type") == "number":
                    left["value"] = round(float(left.get("value") or 0) + float(delta), 4)
                    return

    action_clauses.append(
        {
            "actions": [
                {
                    "type": "assignment",
                    "lhs": "score",
                    "op": "+=",
                    "rhs": {
                        "type": "binary",
                        "op": "*",
                        "left": {"type": "identifier", "name": scale_var},
                        "right": {"type": "number", "value": round(float(delta), 4)},
                    },
                }
            ]
        }
    )


def _bump_score_times_ast(action_clauses: List[Dict[str, Any]], extra_times: int) -> None:
    """Duplicate primary `score += ...` action `extra_times` times to model `scoreTimes+=N`."""
    if extra_times <= 0:
        return
    for blk in action_clauses:
        if not isinstance(blk, dict):
            continue
        acts = blk.get("actions")
        if not isinstance(acts, list):
            continue
        for idx, act in enumerate(acts):
            if isinstance(act, dict) and act.get("type") == "assignment" and act.get("lhs") == "score":
                for _ in range(extra_times):
                    acts.insert(idx + 1, copy.deepcopy(act))
                return


def _apply_patch_entry(
    parsed: Dict[str, Any],
    sec: str,
    entry: Dict[str, Any],
    dsl_actions: str,
) -> str:
    """Apply a single customization condition/action/effect patch entry onto `parsed`."""
    op = entry.get("op")
    if op == "patch":
        anchor = entry.get("anchor")
        delta = entry.get("delta")
        if anchor and isinstance(delta, dict):
            for idx, existing in enumerate(parsed.get(sec) or ()):
                if isinstance(existing, dict) and existing.get("anchor") == anchor:
                    updated = copy.deepcopy(existing)
                    updated.update(copy.deepcopy(delta))
                    parsed[sec][idx] = updated
                    break
        return dsl_actions

    if op != "append":
        return dsl_actions
    eff = entry.get("effect")
    if not isinstance(eff, dict):
        return dsl_actions

    if sec == "effects" and eff.get("phase") == "prestage":
        for act in eff.get("actions") or ():
            if not isinstance(act, dict) or act.get("type") != "assignment":
                continue
            lhs = str(act.get("lhs") or "")
            rhs = act.get("rhs")
            if not (isinstance(rhs, dict) and rhs.get("type") == "number"):
                continue
            val = float(rhs.get("value") or 0)
            if lhs == "g.cost":
                _reduce_cost_ast(parsed.setdefault("cost", []), val, cost_only=True)
            elif lhs == "g.typedCost":
                _reduce_cost_ast(parsed.setdefault("cost", []), val, cost_only=False)
            elif lhs in _GROWTH_VAR_TO_ACTION_LHS:
                _bump_action_lhs_ast(parsed.setdefault("actions", []), _GROWTH_VAR_TO_ACTION_LHS[lhs], val)
            elif lhs in _SCALE_VAR_TO_NAME:
                _bump_score_multiplier_ast(parsed.setdefault("actions", []), _SCALE_VAR_TO_NAME[lhs], val)
            elif lhs == "g.scoreTimes":
                _bump_score_times_ast(parsed.setdefault("actions", []), int(round(val)))
        return dsl_actions

    parsed.setdefault(sec, []).append(copy.deepcopy(eff))
    if sec == "actions":
        for act in eff.get("actions") or ():
            if isinstance(act, dict) and act.get("lhs") == "cardUsesRemaining" and "cardUsesRemaining+=" not in dsl_actions:
                dsl_actions = f"{dsl_actions}; cardUsesRemaining+=1".strip("; ")
    return dsl_actions


def apply_card_customizations(
    parsed_effects: Dict[str, Any],
    *,
    force_initial_hand: bool,
    limit_count: Optional[int],
    dsl_actions: str,
    db: GakumasDB,
    customization_levels: Optional[Dict[Any, Any]] = None,
    custom_modifiers: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bool, Optional[int], str]:
    """Apply official DB customizations and manual extra enhancements onto a card's cloned AST."""
    parsed = copy.deepcopy(parsed_effects) if isinstance(parsed_effects, dict) else {}
    fih = bool(force_initial_hand)
    lim = limit_count
    eff_dsl = str(dsl_actions or "")

    for raw_cid, raw_lv in sorted((customization_levels or {}).items(), key=lambda kv: int(kv[0])):
        try:
            cust_id = int(raw_cid)
            lv = int(raw_lv or 0)
        except (TypeError, ValueError):
            continue
        if lv <= 0 or cust_id <= 0:
            continue
        crow = db.get_customization_by_id(cust_id) if hasattr(db, "get_customization_by_id") else None
        if not crow:
            continue
        if crow.get("force_initial_hand") == 1:
            fih = True
        if crow.get("limit_count") == 0:
            lim = None
        if "cardUsesRemaining+=" in str(crow.get("dsl_actions") or "") and "cardUsesRemaining+=" not in eff_dsl:
            eff_dsl = f"{eff_dsl}; cardUsesRemaining+=1".strip("; ")
        try:
            patches = json.loads(crow.get("parsed_patches_json") or "{}")
        except Exception:
            patches = {}
        for sec in ("conditions", "cost", "actions", "effects"):
            for entry in patches.get(sec) or ():
                if not isinstance(entry, dict):
                    continue
                entry_lv = entry.get("level")
                if entry_lv is not None and int(entry_lv) != lv:
                    continue
                eff_dsl = _apply_patch_entry(parsed, sec, entry, eff_dsl)

    mods = custom_modifiers if isinstance(custom_modifiers, dict) else {}
    if mods.get("force_initial_hand"):
        fih = True
    if mods.get("remove_limit"):
        lim = None
    cost_red = int(mods.get("cost_reduction") or 0)
    if cost_red > 0:
        _reduce_cost_ast(parsed.setdefault("cost", []), float(cost_red), cost_only=None)
    mod_delta_map = (
        ("score_delta", "score"),
        ("good_impression_delta", "goodImpressionTurns"),
        ("motivation_delta", "motivation"),
        ("genki_delta", "genki"),
        ("good_condition_delta", "goodConditionTurns"),
        ("concentration_delta", "concentration"),
        ("full_power_charge_delta", "fullPowerCharge"),
    )
    for mod_k, lhs_k in mod_delta_map:
        dval = float(mods.get(mod_k) or 0)
        if dval != 0:
            _bump_action_lhs_ast(parsed.setdefault("actions", []), lhs_k, dval)

    return parsed, fih, lim, eff_dsl


def _make_card_config(
    db: GakumasDB,
    card_id: int,
    *,
    force_initial_hand_override: Optional[bool] = None,
    exiles_trouble_on_play: bool = False,
    free_extra_use_first_play: bool = False,
    memory_extra_use_limit: int = 0,
    memory_draw_on_play: int = 0,
    customizations: Optional[Dict[Any, Any]] = None,
    custom_modifiers: Optional[Dict[str, Any]] = None,
) -> CardConfig:
    row = db.get_card_by_id(card_id)
    if row is None:
        raise ValueError(f"Unknown card_id in gakumas_library.db: {card_id}")
    raw_parsed = json.loads(row.get("parsed_effects_json") or "{}")
    base_fih = (
        force_initial_hand_override
        if force_initial_hand_override is not None
        else bool(row.get("force_initial_hand"))
    )
    base_lim = int(row["limit_count"]) if row.get("limit_count") is not None else None
    base_dsl = str(row.get("dsl_actions") or "")

    clean_custs: Dict[int, int] = {}
    for k, v in (customizations or {}).items():
        try:
            ik, iv = int(k), int(v or 0)
            if ik > 0 and iv > 0:
                clean_custs[ik] = iv
        except (TypeError, ValueError):
            continue
    clean_mods: Dict[str, Any] = dict(custom_modifiers) if isinstance(custom_modifiers, dict) else {}

    if clean_custs or clean_mods:
        parsed, eff_fih, eff_lim, eff_dsl = apply_card_customizations(
            raw_parsed,
            force_initial_hand=base_fih,
            limit_count=base_lim,
            dsl_actions=base_dsl,
            db=db,
            customization_levels=clean_custs,
            custom_modifiers=clean_mods,
        )
    else:
        parsed, eff_fih, eff_lim, eff_dsl = raw_parsed, base_fih, base_lim, base_dsl

    eff_extra_limit = int(memory_extra_use_limit or 0)
    eff_free_extra = bool(free_extra_use_first_play or eff_extra_limit > 0)
    if eff_free_extra and eff_extra_limit <= 0:
        eff_extra_limit = 1
    return CardConfig(
        card_id=int(row["id"]),
        name=str(row["name"]),
        base_id=int(row.get("base_id") or row["id"]),
        upgraded_id=int(row["upgraded_id"]) if row.get("upgraded_id") else None,
        card_type=str(row.get("type") or "active"),
        rarity=str(row.get("rarity") or "N"),
        upgraded=bool(row.get("upgraded")),
        limit_count=eff_lim,
        force_initial_hand=eff_fih,
        p_idol_id=int(row["p_idol_id"]) if row.get("p_idol_id") is not None else None,
        is_signature=bool(row.get("is_signature")),
        dsl_conditions=str(row.get("dsl_conditions") or ""),
        dsl_cost=str(row.get("dsl_cost") or ""),
        dsl_actions=eff_dsl,
        dsl_effects=str(row.get("dsl_effects") or ""),
        parsed_effects=parsed,
        exiles_trouble_on_play=exiles_trouble_on_play,
        free_extra_use_first_play=eff_free_extra,
        memory_extra_use_limit=eff_extra_limit,
        memory_draw_on_play=max(0, int(memory_draw_on_play or 0)),
        customizations=clean_custs,
        custom_modifiers=clean_mods,
        source_type=str(row.get("source_type") or ("pIdol" if row.get("is_signature") else "produce")),
    )


def _make_drink_config(db: GakumasDB, drink_id: int) -> DrinkConfig:
    row = db.get_drink_by_id(drink_id)
    if row is None:
        raise ValueError(f"Unknown drink_id in gakumas_library.db: {drink_id}")
    parsed = json.loads(row.get("parsed_effects_json") or "{}")
    return DrinkConfig(
        drink_id=int(row["id"]),
        name=str(row["name"]),
        rarity=str(row.get("rarity") or "R"),
        description=str(row.get("description") or "").replace("\n", " / "),
        dsl_actions=str(row.get("dsl_actions") or ""),
        parsed_effects=parsed,
    )


def _build_hif_p_items(*, include_cheering_stick: bool = True) -> List[PItemConfig]:
    """Build all equipped P-items observed on the H.I.F 本戦 loadout."""
    items: List[PItemConfig] = [
        PItemConfig(
            item_id=509,
            name="星々を統べる輝き+",
            rarity="SSR",
            mode="stage",
            source_type="pIdol",
            description="直接効果で好印象が4回増加毎: 2T 好印象追加発動+1 & 好印象強化+10% (無体力消耗)",
            dsl_effects="at:goodImpressionTurnsIncreased { if:isDirectEffect { effectCounter+=1; if:effectCounter%4==3 { setGoodImpressionTurnsTimesBuff(1,2); setGoodImpressionTurnsEffectBuff(0.1,2) } } }",
            trigger_every_n_direct_gi=4,
            grant_extra_gi_eot_triggers=1,
            grant_extra_gi_eot_ttl=2,
            grant_gi_effect_buff=0.10,
            grant_gi_effect_ttl=2,
            stamina_cost_on_trigger=0,
            max_triggers=None,
        ),
        PItemConfig(
            item_id=404,
            name="咲季の完全食・改",
            rarity="SSR",
            mode="stage",
            source_type="support",
            description="ターン内にスキルカードを2回使用時: 好印象+2(間接) & 永久好印象強化+25% (レッスン内1回)",
            dsl_effects="at:cardUsed { if:turnCardsUsed%2==1 { goodImpressionTurns+=2; setGoodImpressionTurnsEffectBuff(0.25) }; limit:1 }",
            trigger_on_nth_card_in_turn=2,
            grant_indirect_gi=2,
            grant_perm_gi_effect_buff=0.25,
            max_triggers=1,
        ),
    ]
    if include_cheering_stick:
        items.append(
            PItemConfig(
                item_id=429,
                name="H.I.F応援棒（黄）",
                rarity="SSR",
                mode="stage",
                source_type="produce",
                description="試験開始時: 山札が22枚未満なら基本スキルカードをランダム位置に補填 (開局GI=6/Mot=6固定)",
                dsl_effects="at:startOfStage { if:countCards[deck]<22 { addRandomBasicCardToDeckAtRandom(22-countCards[deck]) } }",
                pad_deck_to_min_cards=22,
                max_triggers=1,
            )
        )
    items.extend(
        [
            PItemConfig(
                item_id=436,
                name="ポーチ（黄）",
                rarity="SSR",
                mode="produce",
                source_type="produce",
                description="レッスン終了時Pドリンク獲得・次の試験開始時 好印象+2・やる気+2 (2回) [已計入固定初期 GI=6, Mot=6]",
            ),
            PItemConfig(
                item_id=447,
                name="H.I.Fワッペン",
                rarity="SSR",
                mode="produce",
                source_type="produce",
                description="スキルカード獲得時 スター性+10 (プロデュース中20回) [育成期専用・戦中不触発]",
            ),
            PItemConfig(
                item_id=61,
                name="美鈴からの贈り物",
                rarity="SR",
                mode="produce",
                source_type="support",
                description="Vo SPレッスン終了時(Vo>=400) ランダムなスキルカードを強化 [育成期専用・戦中不触発]",
            ),
            PItemConfig(
                item_id=510,
                name="過激め？チョイス",
                rarity="SSR",
                mode="produce",
                source_type="support",
                description="ロジックカード獲得時(Vo>=900) Vo+25・ランダムなスキルカードを強化 (2回) [育成期専用・戦中不触発]",
            ),
            PItemConfig(
                item_id=487,
                name="素敵なお仕事",
                rarity="SSR",
                mode="produce",
                source_type="support",
                description="スキルカードチェンジ時(Vi>=400) Vi+25・体力回復2 (3回) [育成期専用・戦中不触発]",
            ),
        ]
    )
    return items


def _build_sense_hif_p_items() -> List[PItemConfig]:
    """Build Sense H.I.F P-items ([ガラクタロード] 姫崎 莉波 & common Sense support items)."""
    return [
        PItemConfig(
            item_id=9139,
            name="憧れ続けた輝き",
            rarity="SSR",
            mode="stage",
            source_type="pIdol",
            description="好調8T以上時、好調効果スキルカード4回使用毎: 絶好調1T & 使用数+1 & 1ドロー & 固定体力-1",
            trigger_every_n_gc_cards=4,
            require_min_gc=8,
            grant_perfect_condition=1,
            grant_card_uses=1,
            grant_draw_cards=1,
            stamina_cost_on_trigger=1,
            max_triggers=None,
        ),
        PItemConfig(
            item_id=9140,
            name="包まれるやさしさ",
            rarity="SSR",
            mode="stage",
            source_type="support",
            description="【Viターン開始時】好調10T以上の場合: 絶好調4T & 元気+10 (レッスン内1回)",
            trigger_attr_turn="Vi",
            require_min_gc=10,
            grant_perfect_condition=4,
            grant_genki=10,
            max_triggers=1,
        ),
        PItemConfig(
            item_id=9141,
            name="ひとりだちシューズ",
            rarity="SSR",
            mode="stage",
            source_type="support",
            description="【Daターン】体力50%以下時発動（サポートPアイテム）",
        ),
        PItemConfig(
            item_id=9142,
            name="ほっこりまんぶく",
            rarity="SSR",
            mode="produce",
            source_type="support",
            description="好調効果スキルカード獲得時ランダムPドリンク獲得 [育成期専用・戦中不触発]",
        ),
    ]


def build_last_run_hif_preset(
    db: Optional[GakumasDB] = None,
    stage: str = "round2",
) -> BattleConfig:
    """
    Build H.I.F 本戦 battle configuration from the previous live run
    ([ガラクタロード] 秦谷 美鈴, p_idol_id=163) under alignment rules:
    - All skill cards upgraded (+); no customizations except #127/#129/#159 opening hand
    - All 8 equipped P-items loaded (stage + produce-phase)
    - Initial GI=6, Mot=6 fixed
    - Once-per-memory trigger limits per card instance
    """
    own_db = db is None
    active_db = db or GakumasDB("gakumas_library.db")
    try:
        stage_l = (stage or "round2").lower()

        # Tuple: (card_id, force_initial_hand, exiles_trouble_once, free_extra_use_once)
        if stage_l == "round1":
            deck_specs: List[Tuple[int, bool, bool, bool]] = [
                (920, False, False, False),  # ありのままのわたし+
                (922, False, False, False),  # 認めてくれるひと+
                (159, True, False, False),   # アイドル魂+ (固有開局手札)
                (129, True, True, False),    # ワクワクが止まらない+ (開局手札 + メモリー除外眠気 1回)
                (127, True, False, False),   # ファンシーチャーム+ (開局手札)
                (137, False, False, True),   # やる気は満点+ (メモリー再動 +1使用数 1回)
                (181, False, False, False),  # 虹色ドリーマー+
                (713, False, False, False),  # 気まぐれハート+
                (23, False, False, False),   # 眠気 (トラブル)
            ]
            drink_ids = [14, 13, 14, 25]
            schedule = [
                TurnMultiplierSpec("Vo", 44.10),
                TurnMultiplierSpec("Vo", 44.10),
                TurnMultiplierSpec("Da", 25.88),
                TurnMultiplierSpec("Vi", 50.85),
                TurnMultiplierSpec("Vi", 50.85),
                TurnMultiplierSpec("Vo", 44.10),
                TurnMultiplierSpec("Da", 25.88),
                TurnMultiplierSpec("Vi", 50.85),
                TurnMultiplierSpec("Vo", 44.10),
            ]
            stamina = 44
            init_gi = 6
            init_mot = 6
            stage_name = "H.I.F 本戦 ラウンド1 (9回合)"
            total_turns = 9
            idol_stats = {"vocal": 2444, "dance": 1468, "visual": 2463}
            p_items = _build_hif_p_items(include_cheering_stick=False)
        else:
            # Round 2 22-card aligned deck: 21 upgraded (+) skill cards + 1 #23 眠気
            deck_specs = [
                (920, False, False, False),  # ありのままのわたし+ (固有1)
                (922, False, False, False),  # 認めてくれるひと+ (固有2)
                (159, True, False, False),   # アイドル魂+ (固有開始時手札)
                (129, True, True, False),    # ワクワクが止まらない+ (開始時手札 + メモリー無料除外眠気 1回)
                (127, True, False, False),   # ファンシーチャーム+ (開始時手札)
                (299, False, True, False),   # オトメゴコロ+ (可重複, メモリー無料除外眠気 僅第1次発動)
                (297, False, True, False),   # 止められない想い+ (メモリー無料除外眠気 1回)
                (137, False, False, True),   # やる気は満点+ (#1 メモリー首次+1使用数 1回)
                (137, False, False, False),  # やる気は満点+ (#2 授業追加分・無メモリー再動)
                (778, False, False, False),  # びしっとキメ顔+
                (305, False, False, False),  # 輝くキミへ+
                (617, False, False, False),  # 対象、ロックオン+
                (521, False, False, False),  # 夢色リップ+
                (181, False, False, False),  # 虹色ドリーマー+
                (713, False, False, False),  # 気まぐれハート+ (戦中生成 #23 眠気)
                (161, False, False, False),  # 仕切り直し+
                (171, False, False, False),  # 私がスター+
                (99, False, False, False),   # アイドル宣言+
                (173, False, False, False),  # 星屑センセーション+
                (179, False, False, False),  # トキメキ+
                (175, False, False, False),  # ノートの端の決意+
                (23, False, False, False),   # 眠気 (初期編成トラブル)
            ]
            drink_ids = [13, 25, 25, 25]
            randomize_sched = True
            sched_attr_mults = (("Vo", 40.0), ("Vi", 35.0), ("Da", 20.0))
            sched_counts = (6, 3, 3)
            # Sample schedule obeying rule: Vo=4000%(6T), Vi=3500%(3T), Da=2000%(3T);
            # T12 fixed Vo 4000%, T10/T11 never Vo, T1..T9 randomized per battle seed.
            schedule = generate_stage_turn_schedule(
                random.Random(0),
                attr_multipliers=sched_attr_mults,
                counts=sched_counts,
            )
            stamina = 33
            init_gi = 6
            init_mot = 6
            stage_name = "H.I.F 本戦 ラウンド2 (12回合)"
            total_turns = 12
            idol_stats = {"vocal": 2730, "dance": 1572, "visual": 1849}
            p_items = _build_hif_p_items(include_cheering_stick=True)

        if stage_l == "round1":
            randomize_sched = False
            sched_attr_mults = (("Vi", 50.85), ("Vo", 44.10), ("Da", 25.88))
            sched_counts = (3, 4, 2)

        deck: List[CardConfig] = [
            _make_card_config(
                active_db,
                cid,
                force_initial_hand_override=force_init,
                exiles_trouble_on_play=exile_once,
                free_extra_use_first_play=extra_once,
            )
            for cid, force_init, exile_once, extra_once in deck_specs
        ]

        drinks = [_make_drink_config(active_db, did) for did in drink_ids]
        notes = [
            "全技能卡按全升级(+)載入，不額外計算カスタマイズ，僅保留 #127 ファンシーチャーム+ / #129 ワクワクが止まらない+ (及固有 #159) 開局入手法効果。",
            "初期パネル固定: 好印象 GI=6, やる気 Mot=6 (戰前サポ/H.I.F効果已固定折算於此)。",
            "每個回憶(メモリー)の『眠気除外 / 免費+1使用數』全場僅觸發1次: 即使重複卡 #299 オトメゴコロ+ 也僅第1次打出觸發除外; 僅編入槽1張 #137 やる気は満点+ 帶1次免費+1使用數。",
            "裝備全8個P道具已齊全載入: 戰中觸發3件 (#509 星々を統べる輝き+, #404 咲季の完全食・改, #429 H.I.F応援棒（黄）) + 育成期/入場合計5件 (#436 ポーチ（黄）, #447 H.I.Fワッペン, #61 美鈴からの贈り物, #510 過激め？チョイス, #487 素敵なお仕事)。",
            "場地倍率按默認三維排序配置: Vo 4000%(6T) / Vi 3500%(3T) / Da 2000%(3T); 第12回合固定 Vo 4000% (加賽T13沿用), 倒數第2/3回合(T10/T11)不出現 Vo, 其餘回合隨機分配。",
        ]
        return BattleConfig(
            scenario_id="hif_honsen",
            stage_name=stage_name,
            p_idol_id=163,
            idol_name="[ガラクタロード] 秦谷 美鈴",
            plan="logic",
            idol_stats=idol_stats,
            initial_stamina=stamina,
            max_stamina=44,
            initial_genki=0,
            initial_good_impression=init_gi,
            initial_motivation=init_mot,
            initial_score_buff=0.0,
            total_turns=total_turns,
            turn_schedule=schedule,
            randomize_turn_schedule=randomize_sched,
            turn_schedule_attr_multipliers=sched_attr_mults,
            turn_schedule_counts=sched_counts,
            deck=deck,
            drinks=drinks,
            p_items=p_items,
            extra_notes=notes,
        )
    finally:
        if own_db:
            active_db.close()


class HeadlessBattleSimulator:
    """Fast headless simulator executing gakumas-tools JSON ASTs and zone rules."""

    MAX_HAND_SIZE = 5
    OPENING_DRAW = 3
    TURN_DRAW = 3

    def __init__(
        self,
        config: BattleConfig,
        db: Optional[GakumasDB] = None,
        seed: int = 0,
        *,
        record_logs: bool = True,
        card_cfg_cache: Optional[Dict[Tuple[Any, ...], CardConfig]] = None,
    ):
        self.config = config
        self._owns_db = db is None
        self.db = db or GakumasDB("gakumas_library.db")
        self._record_logs = record_logs
        self._card_cfg_cache: Dict[Tuple[Any, ...], CardConfig] = (
            card_cfg_cache if card_cfg_cache is not None else {}
        )
        for cfg in config.deck:
            key = (
                cfg.card_id,
                cfg.force_initial_hand,
                cfg.exiles_trouble_on_play,
                cfg.free_extra_use_first_play,
                cfg.memory_extra_use_limit,
                cfg.memory_draw_on_play,
                tuple(sorted(cfg.customizations.items())),
                tuple(sorted((str(k), v) for k, v in cfg.custom_modifiers.items())),
            )
            self._card_cfg_cache.setdefault(key, cfg)
        self.rng = random.Random(seed)
        self.seed = seed

        self.turn = 1
        self.turns_elapsed = 0
        self.turns_remaining = config.total_turns
        self.card_uses_remaining = 1
        self.stamina = config.initial_stamina
        self.max_stamina = config.max_stamina
        self.genki = config.initial_genki
        self.score = 0
        self.buffs: Dict[str, int] = {
            "goodImpressionTurns": config.initial_good_impression,
            "motivation": config.initial_motivation,
            "halfCostTurns": 0,
            "doubleCostTurns": 0,
            "nullifyDebuff": 0,
            "goodConditionTurns": config.initial_good_condition,
            "perfectConditionTurns": config.initial_perfect_condition,
            "poorConditionTurns": 0,
            "concentration": config.initial_concentration,
            "nullifyCostCards": 0,
            "nullifyCostActiveCards": 0,
        }
        if config.initial_full_power_charge > 0:
            self.buffs["fullPowerCharge"] = int(config.initial_full_power_charge)
        self.perm_gi_effect_buff = 0.0
        self.perm_score_buff = config.initial_score_buff
        self.concentration_multiplier: float = 1.0
        self.good_condition_multiplier: float = 1.0
        self.perm_conc_effect_buff: float = 0.0
        self.perm_gc_gain_buff: float = 0.0
        self.perm_conc_gain_buff: float = 0.0
        # Timed buffs store (amount, remaining_turns, fresh) matching BuffManager.js
        self.timed_gi_effect_buffs: List[Tuple[float, int, bool]] = []
        self.timed_gi_gain_buffs: List[Tuple[float, int, bool]] = []
        self.timed_score_buffs: List[Tuple[float, int, bool]] = []
        self.timed_extra_gi_eot: List[Tuple[int, int, bool]] = []
        self.timed_conc_effect_buffs: List[Tuple[float, int, bool]] = []
        self.timed_gc_gain_buffs: List[Tuple[float, int, bool]] = []
        self.timed_gc_addition_buffs: List[Tuple[int, int, bool]] = []
        self.timed_conc_gain_buffs: List[Tuple[float, int, bool]] = []
        self.fresh_turn_stat_buffs: set[str] = set()
        self.in_unfresh_phase: bool = True

        self.double_card_effect_cards = 0
        self.turn_cards_used = 0
        self.cards_used = 0
        self.active_cards_used = 0
        self.phase: Optional[str] = None
        self.parent_phase: Optional[str] = None
        self._current_change_is_direct: bool = False
        self._active_reservation_is_card: bool = False
        self._free_card_uses: int = 0
        self._buff_cost_consumed: bool = False

        self.p_item_gi_counters: Dict[int, int] = {it.item_id: 0 for it in config.p_items}
        self.p_item_sense_gc_counters: Dict[int, int] = {it.item_id: 0 for it in config.p_items}
        self.p_item_trigger_counts: Dict[int, int] = {it.item_id: 0 for it in config.p_items}
        # Once-per-memory-card-instance (uid) and stage-wide base_id tracking
        self.memory_extra_used_uids: set[int] = set()
        self.memory_exile_used_uids: set[int] = set()
        self.memory_extra_used_base_ids: set[int] = set()
        self.memory_extra_use_counts: Dict[int, int] = {}
        self.memory_exile_used_base_ids: set[int] = set()
        self.memory_triggers_log: List[str] = []
        self.card_effect_counters: Dict[int, int] = {}
        self.card_turn_counters: Dict[int, int] = {}

        self.next_uid = 1
        self._all_instances: Dict[int, CardInstance] = {}
        self.hand: List[CardInstance] = []
        self.draw_pile: List[CardInstance] = []
        self.discard_pile: List[CardInstance] = []
        self.exhaust_pile: List[CardInstance] = []
        self.in_play_card: Optional[CardInstance] = None
        self.used_card: Optional[CardInstance] = None
        self.last_used_card: Optional[CardInstance] = None
        self.moved_card: Optional[CardInstance] = None
        self.available_drinks: List[DrinkConfig] = list(config.drinks)

        self.triggers: List[ActiveTrigger] = []
        self.reshuffle_count = 0
        self.cards_played_log: List[str] = []
        self.drinks_used_log: List[str] = []
        self.turn_logs: List[str] = []

        if config.randomize_turn_schedule:
            self.turn_schedule: List[TurnMultiplierSpec] = generate_stage_turn_schedule(
                self.rng,
                attr_multipliers=config.turn_schedule_attr_multipliers,
                counts=config.turn_schedule_counts,
            )
        else:
            self.turn_schedule = list(config.turn_schedule)

        self._setup_initial_zones_and_passives()

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def _get_card_config_cached(
        self,
        card_id: int,
        *,
        force_initial_hand_override: Optional[bool] = None,
        exiles_trouble_on_play: bool = False,
        free_extra_use_first_play: bool = False,
        memory_extra_use_limit: int = 0,
        memory_draw_on_play: int = 0,
        customizations: Optional[Dict[int, int]] = None,
        custom_modifiers: Optional[Dict[str, Any]] = None,
    ) -> CardConfig:
        key = (
            card_id,
            force_initial_hand_override,
            exiles_trouble_on_play,
            free_extra_use_first_play,
            memory_extra_use_limit,
            memory_draw_on_play,
            tuple(sorted((customizations or {}).items())),
            tuple(sorted((str(k), v) for k, v in (custom_modifiers or {}).items())),
        )
        cfg = self._card_cfg_cache.get(key)
        if cfg is None:
            cfg = _make_card_config(
                self.db,
                card_id,
                force_initial_hand_override=force_initial_hand_override,
                exiles_trouble_on_play=exiles_trouble_on_play,
                free_extra_use_first_play=free_extra_use_first_play,
                memory_extra_use_limit=memory_extra_use_limit,
                memory_draw_on_play=memory_draw_on_play,
                customizations=customizations,
                custom_modifiers=custom_modifiers,
            )
            self._card_cfg_cache[key] = cfg
        return cfg

    def _new_instance(self, cfg: CardConfig) -> CardInstance:
        inst = CardInstance(uid=self.next_uid, cfg=cfg)
        self.next_uid += 1
        self._all_instances[inst.uid] = inst
        return inst

    def _query_basic_padding_card_ids(self) -> List[int]:
        try:
            rows = self.db.connection.execute(
                "SELECT id FROM cards WHERE rarity='N' AND upgraded=0 AND plan IN (?, 'free') AND name LIKE '%基本%' ORDER BY id",
                (self.config.plan.lower(),),
            ).fetchall()
            ids = [int(r["id"]) for r in rows]
            if ids:
                return ids
        except Exception:
            pass
        return [1, 3, 13, 19, 21]

    def _register_passive_effects_for_card(self, inst: CardInstance) -> None:
        effects = inst.cfg.parsed_effects.get("effects") or []
        if effects or inst.cfg.base_id == 919:
            self.card_effect_counters.setdefault(inst.uid, 0)
            self.card_turn_counters.setdefault(inst.uid, 0)
        for eff in effects:
            phase = eff.get("phase")
            if not phase:
                continue
            eff_actions = list(eff.get("actions") or [])
            eff_nested = list(eff.get("effects") or [])
            self.triggers.append(
                ActiveTrigger(
                    owner_name=inst.cfg.name,
                    owner_uid=inst.uid,
                    phase=str(phase),
                    conditions=list(eff.get("conditions") or []),
                    actions=eff_actions,
                    effects=eff_nested,
                    limit=int(eff["limit"]) if eff.get("limit") is not None else None,
                    ttl=int(eff["ttl"]) if eff.get("ttl") is not None else None,
                    delay=int(eff["delay"]) if eff.get("delay") is not None else None,
                    is_passive_card_effect=True,
                    grants_double_card_effect=_actions_grant_double_card_effect(eff_actions, eff_nested),
                )
            )

    def _setup_initial_zones_and_passives(self) -> None:
        self.in_unfresh_phase = True
        init_cards: List[CardInstance] = []
        rest_cards: List[CardInstance] = []
        for cfg in self.config.deck:
            inst = self._new_instance(cfg)
            self._register_passive_effects_for_card(inst)
            if cfg.force_initial_hand:
                init_cards.append(inst)
            else:
                rest_cards.append(inst)

        # startOfStage P-item (#429 H.I.F応援棒（黄）): pad basic cards up to min deck size if < 22
        total_cards = len(init_cards) + len(rest_cards)
        for it in self.config.p_items:
            if it.mode != "stage" or it.pad_deck_to_min_cards <= 0:
                continue
            if it.max_triggers is not None and self.p_item_trigger_counts[it.item_id] >= it.max_triggers:
                continue
            if total_cards < it.pad_deck_to_min_cards:
                needed = it.pad_deck_to_min_cards - total_cards
                basic_ids = self._query_basic_padding_card_ids()
                for _ in range(needed):
                    bcid = self.rng.choice(basic_ids)
                    bcfg = self._get_card_config_cached(bcid)
                    binst = self._new_instance(bcfg)
                    self._register_passive_effects_for_card(binst)
                    rest_cards.append(binst)
                self.p_item_trigger_counts[it.item_id] += 1
                total_cards += needed

        self.rng.shuffle(rest_cards)
        self.draw_pile = init_cards + rest_cards
        self._fire_phase("startOfStage", is_direct=False)
        self._fire_phase("afterStartOfStage", is_direct=False)
        self._start_turn(is_first_turn=True)

    def _start_turn(self, *, is_first_turn: bool = False) -> None:
        self.card_uses_remaining = 1
        self.turn_cards_used = 0
        self.in_unfresh_phase = True
        effects_before_turn_start = len(self.triggers)
        self._fire_phase("beforeStartOfTurn", is_direct=False)
        self._fire_phase("startOfTurn", is_direct=False)
        self.draw_cards(self.TURN_DRAW)
        if is_first_turn:
            for _ in range(2):
                if len(self.hand) >= self.MAX_HAND_SIZE or not self.draw_pile:
                    break
                if self.draw_pile[0].cfg.force_initial_hand:
                    self.draw_cards(1)
                else:
                    break
        self._fire_phase("afterStartOfTurn", is_direct=False)
        self._fire_phase("turn", registered_before=effects_before_turn_start, is_direct=False)
        self._fire_phase("everyTurn", is_direct=False)
        self.in_unfresh_phase = False

    def current_turn_spec(self) -> TurnMultiplierSpec:
        idx = self.turn - 1
        if 0 <= idx < len(self.turn_schedule):
            return self.turn_schedule[idx]
        if self.turn_schedule:
            return self.turn_schedule[-1]
        return TurnMultiplierSpec("Vo", 1.0)

    def total_gi_effect_buff(self) -> float:
        return self.perm_gi_effect_buff + sum(v for v, _, _ in self.timed_gi_effect_buffs)

    def total_gi_gain_buff(self) -> float:
        return sum(v for v, _, _ in self.timed_gi_gain_buffs)

    def total_score_buff(self) -> float:
        return self.perm_score_buff + sum(v for v, _, _ in self.timed_score_buffs)

    def extra_gi_eot_triggers(self) -> int:
        return sum(v for v, _, _ in self.timed_extra_gi_eot)

    def total_conc_effect_buff(self) -> float:
        return self.perm_conc_effect_buff + sum(v for v, _, _ in self.timed_conc_effect_buffs)

    def total_gc_gain_buff(self) -> float:
        return self.perm_gc_gain_buff + sum(v for v, _, _ in self.timed_gc_gain_buffs)

    def total_gc_addition_buff(self) -> int:
        return sum(int(v) for v, _, _ in self.timed_gc_addition_buffs)

    def total_conc_gain_buff(self) -> float:
        return self.perm_conc_gain_buff + sum(v for v, _, _ in self.timed_conc_gain_buffs)

    def _set_buff_value(self, key: str, new_val: int) -> None:
        prev = int(self.buffs.get(key, 0) or 0)
        clamped = max(0, int(new_val))
        self.buffs[key] = clamped
        if key in EOT_DECREMENT_FIELDS:
            if prev == 0 and clamped > 0 and not self.in_unfresh_phase:
                self.fresh_turn_stat_buffs.add(key)

    @staticmethod
    def _step_timed_buffs(entries: List[Tuple[Any, int, bool]]) -> List[Tuple[Any, int, bool]]:
        out: List[Tuple[Any, int, bool]] = []
        for val, turns, fresh in entries:
            if fresh:
                out.append((val, turns, False))
            elif turns > 1:
                out.append((val, turns - 1, False))
        return out

    def _compute_final_score_gain(self, raw_score: float, *, apply_sense_mods: bool = False) -> int:
        if raw_score <= 0 and not apply_sense_mods:
            return 0
        spec = self.current_turn_spec()
        if not apply_sense_mods:
            step1 = _safe_ceil(raw_score)
            step2 = _safe_ceil(step1 * (1.0 + max(0.0, self.total_score_buff())))
            return _safe_ceil(step2 * spec.multiplier)

        conc = max(0, int(self.buffs.get("concentration", 0) or 0))
        conc_eff = 1.0 + max(0.0, self.total_conc_effect_buff())
        base = float(raw_score) + float(conc) * float(self.concentration_multiplier) * conc_eff
        if base <= 0:
            return 0
        step1 = _safe_ceil(base)
        gc = max(0, int(self.buffs.get("goodConditionTurns", 0) or 0))
        pc = max(0, int(self.buffs.get("perfectConditionTurns", 0) or 0))
        poor = max(0, int(self.buffs.get("poorConditionTurns", 0) or 0))
        cond_bonus = 0.0
        if gc > 0:
            cond_bonus += 0.5
            if pc > 0:
                cond_bonus += 0.1 * gc
        cond_mult = 1.0 + cond_bonus * float(self.good_condition_multiplier)
        if poor > 0:
            cond_mult = max(0.1, cond_mult - 0.33)
        step_cond = _safe_ceil(step1 * cond_mult)
        step2 = _safe_ceil(step_cond * (1.0 + max(0.0, self.total_score_buff())))
        return _safe_ceil(step2 * spec.multiplier)

    def _active_instances(self) -> Dict[int, CardInstance]:
        active: Dict[int, CardInstance] = {}
        for z in (self.hand, self.draw_pile, self.discard_pile, self.exhaust_pile):
            for c in z:
                active[c.uid] = c
        if self.used_card is not None:
            active.setdefault(self.used_card.uid, self.used_card)
        return active

    def _snapshot_condition_state(
        self,
        *,
        card: Optional[CardInstance] = None,
        is_direct: Optional[bool] = None,
    ) -> ConditionSnapshot:
        ref_card = card if card is not None else self.used_card
        active = self._active_instances()
        direct_flag = (
            bool(is_direct)
            if is_direct is not None
            else ((self.phase in DIRECT_CHANGE_PHASES) or self._active_reservation_is_card)
        )
        return ConditionSnapshot(
            buffs=dict(self.buffs),
            stamina=self.stamina,
            genki=self.genki,
            score=self.score,
            turns_elapsed=self.turns_elapsed,
            turns_remaining=self.turns_remaining,
            turn_cards_used=self.turn_cards_used,
            cards_used=self.cards_used,
            active_cards_used=self.active_cards_used,
            card_effect_counters=dict(self.card_effect_counters),
            card_turn_counters=dict(self.card_turn_counters),
            zone_uids={
                "hand": {c.uid for c in self.hand},
                "deck": {c.uid for c in self.draw_pile},
                "discarded": {c.uid for c in self.discard_pile},
                "removed": {c.uid for c in self.exhaust_pile},
                "held": set(),
                "all": set(active.keys()),
            },
            card_rarity=ref_card.cfg.rarity if ref_card is not None else "",
            card_type=ref_card.cfg.card_type if ref_card is not None else "",
            used_card_id=ref_card.cfg.card_id if ref_card is not None else 0,
            used_card_base_id=ref_card.cfg.base_id if ref_card is not None else 0,
            moved_card_id=self.moved_card.cfg.card_id if self.moved_card is not None else 0,
            moved_card_base_id=self.moved_card.cfg.base_id if self.moved_card is not None else 0,
            last_used_card_type=self.last_used_card.cfg.card_type if self.last_used_card is not None else "",
            is_direct_effect=direct_flag,
        )

    def draw_cards(self, count: int = 1) -> int:
        drawn = 0
        for _ in range(max(0, count)):
            if len(self.hand) >= self.MAX_HAND_SIZE:
                break
            if not self.draw_pile:
                if not self.discard_pile:
                    break
                self.draw_pile = list(self.discard_pile)
                self.discard_pile.clear()
                self.rng.shuffle(self.draw_pile)
                self.reshuffle_count += 1
            if not self.draw_pile:
                break
            inst = self.draw_pile.pop(0)
            self.hand.append(inst)
            drawn += 1
            self.moved_card = inst
            self._fire_phase("cardMovedToHand", is_direct=True)
        return drawn

    def _effective_cost_with_nullify(self, cfg: CardConfig) -> Tuple[str, int, Optional[str]]:
        cost_clauses = cfg.parsed_effects.get("cost") or []
        if not cost_clauses:
            return ("cost", 0, None)
        act_list = cost_clauses[0].get("actions") or []
        if not act_list:
            return ("cost", 0, None)
        act = act_list[0]
        lhs = str(act.get("lhs") or "cost")
        amt = int(float(self._eval_expr(act.get("rhs", 0))))
        if lhs in ("cost", "stamina"):
            if amt > 0:
                if cfg.card_type == "active" and int(self.buffs.get("nullifyCostActiveCards", 0) or 0) > 0:
                    return (lhs, 0, "nullifyCostActiveCards")
                if int(self.buffs.get("nullifyCostCards", 0) or 0) > 0:
                    return (lhs, 0, "nullifyCostCards")
            eff_float = -float(amt)
            if self.buffs.get("halfCostTurns", 0) > 0:
                eff_float *= 0.5
            if self.buffs.get("doubleCostTurns", 0) > 0:
                eff_float *= 2.0
            eff = -math.floor(eff_float)
            if lhs == "cost":
                eff -= int(self.buffs.get("costReduction", 0) or 0)
                eff += int(self.buffs.get("costIncrease", 0) or 0)
            return (lhs, max(0, eff), None)
        if self.buffs.get("doubleBuffCostTurns", 0) > 0 and lhs in CONSUMABLE_BUFF_FIELDS:
            amt *= 2
        return (lhs, max(0, amt), None)

    def _effective_cost(self, cfg: CardConfig) -> Tuple[str, int]:
        lhs, eff_amt, _ = self._effective_cost_with_nullify(cfg)
        return (lhs, eff_amt)

    def can_play_card(self, inst: CardInstance) -> bool:
        if self.card_uses_remaining <= 0:
            return False
        if self.buffs.get("noCardUseTurns", 0) > 0:
            return False
        if self.buffs.get("noActiveTurns", 0) > 0 and inst.cfg.card_type == "active":
            return False
        if self.buffs.get("noMentalTurns", 0) > 0 and inst.cfg.card_type == "mental":
            return False
        snap = self._snapshot_condition_state(card=inst, is_direct=False)
        for cond_node in inst.cfg.parsed_effects.get("conditions") or []:
            if not self._eval_condition(cond_node, snap=snap):
                return False
        cost_var, eff_amt = self._effective_cost(inst.cfg)
        if cost_var == "cost":
            return (self.genki + self.stamina) >= eff_amt
        if cost_var == "stamina":
            return self.stamina >= eff_amt
        return int(self.buffs.get(cost_var, 0) or 0) >= eff_amt

    def has_unused_memory_extra_use(self, inst: CardInstance) -> bool:
        effective_limit = int(
            getattr(inst.cfg, "memory_extra_use_limit", 0)
            or (1 if inst.cfg.free_extra_use_first_play else 0)
        )
        if effective_limit <= 0:
            return False
        if inst.uid in self.memory_extra_used_uids or inst.cfg.base_id in self.memory_extra_used_base_ids:
            return False
        return self.memory_extra_use_counts.get(inst.cfg.base_id, 0) < effective_limit

    def has_unused_memory_exile(self, inst: CardInstance) -> bool:
        return bool(
            inst.cfg.exiles_trouble_on_play
            and inst.uid not in self.memory_exile_used_uids
            and inst.cfg.base_id not in self.memory_exile_used_base_ids
        )

    def is_free_action_card(self, inst: CardInstance) -> bool:
        return ("cardUsesRemaining+=" in (inst.cfg.dsl_actions or "")) or self.has_unused_memory_extra_use(inst)

    def has_active_troubles(self) -> bool:
        return any(
            (c.cfg.card_type == "trouble" or c.cfg.base_id == 23)
            for z in (self.hand, self.draw_pile, self.discard_pile)
            for c in z
        )

    def has_motivation_engine(self) -> bool:
        """Return True if an active trigger grants Motivation on Mental card plays (e.g. #128/#129)."""
        all_inst = self._all_instances
        for tr in self.triggers:
            if tr.limit is not None and tr.triggers_count >= tr.limit:
                continue
            if tr.ttl is not None and tr.ttl < 0:
                continue
            if tr.phase != "mentalCardUsed":
                continue
            owner = all_inst.get(tr.owner_uid)
            if owner is not None and owner.cfg.base_id == 128:
                return True
            for blk in (tr.actions or ()):
                for act in (blk.get("actions") or ()):
                    if act.get("lhs") == "motivation" and act.get("op") in ("+=", "="):
                        return True
        return False

    def active_motivation_economy(self) -> Dict[str, int]:
        """Return order-invariant summary of Motivation spenders and gainers across active deck zones."""
        spenders_count = 0
        spenders_total_cost = 0
        gainers_count = 0
        unplayable_mot_in_hand = 0
        cur_mot = int(self.buffs.get("motivation", 0) or 0)
        for z_idx, zone in enumerate((self.hand, self.draw_pile, self.discard_pile)):
            for inst in zone:
                c_var, c_amt = self._effective_cost(inst.cfg)
                if c_var == "motivation" and c_amt > 0:
                    spenders_count += 1
                    spenders_total_cost += int(c_amt)
                    if z_idx == 0 and cur_mot < c_amt:
                        unplayable_mot_in_hand += 1
                if "motivation+=" in (inst.cfg.dsl_actions or ""):
                    gainers_count += 1
        return {
            "spenders_count": spenders_count,
            "spenders_total_cost": spenders_total_cost,
            "gainers_count": gainers_count,
            "unplayable_mot_in_hand": unplayable_mot_in_hand,
        }

    def has_armed_active_retrigger(self, *, same_turn_only: bool = False) -> bool:
        """Return True if an Active-card double-effect trigger is armed (optionally same-turn ttl==1 only)."""
        if not same_turn_only and self.double_card_effect_cards > 0:
            return True
        for tr in self.triggers:
            if not tr.grants_double_card_effect or tr.phase != "activeCardUsed":
                continue
            if tr.limit is not None and tr.triggers_count >= tr.limit:
                continue
            if tr.ttl is not None and tr.ttl < 0:
                continue
            if same_turn_only and tr.ttl != 1:
                continue
            return True
        return False

    @staticmethod
    def is_active_retrigger_drink(drink: DrinkConfig) -> bool:
        """Return True if `drink` grants an Active-only card effect retrigger (e.g. #25 特製ハツボシエキス)."""
        return drink.drink_id == 25 or ("doublecardeffectcards" in (drink.dsl_actions or "").lower())

    def can_use_active_retrigger_drink_now(self, drink: DrinkConfig) -> bool:
        """Permit #25 only after dominating free non-Active cards are played and an Active card remains playable post-drink."""
        if not self.is_active_retrigger_drink(drink):
            return True
        if self.has_armed_active_retrigger():
            return False
        if any(self.is_safe_dominating_free_card(c) and c.cfg.card_type != "active" for c in self.hand):
            return False
        if (
            self.card_uses_remaining <= 0
            or self.buffs.get("noCardUseTurns", 0) > 0
            or self.buffs.get("noActiveTurns", 0) > 0
        ):
            return False

        dsl_l = (drink.dsl_actions or "").lower()
        m_stam = re.search(r"fixedstamina-=(\d+)", dsl_l)
        stam_drop = int(m_stam.group(1)) if m_stam else (2 if drink.drink_id == 25 else 0)
        adds_double_cost = ("doublecostturns+=" in dsl_l) or (drink.drink_id == 25)
        post_stam = max(0, self.stamina - stam_drop)

        orig_double_cost = int(self.buffs.get("doubleCostTurns", 0) or 0)
        will_add_double = adds_double_cost and int(self.buffs.get("nullifyDebuff", 0) or 0) <= 0
        try:
            if will_add_double:
                self.buffs["doubleCostTurns"] = orig_double_cost + 1
            for inst in self.hand:
                if inst.cfg.card_type != "active" or inst.cfg.rarity == "L":
                    continue
                if not self.can_play_card(inst):
                    continue
                cost_var, eff_amt = self._effective_cost(inst.cfg)
                if cost_var == "cost":
                    if (self.genki + post_stam) >= eff_amt:
                        return True
                elif cost_var == "stamina":
                    if post_stam >= eff_amt:
                        return True
                elif int(self.buffs.get(cost_var, 0) or 0) >= eff_amt:
                    return True
        finally:
            if will_add_double:
                self.buffs["doubleCostTurns"] = orig_double_cost
        return False

    @staticmethod
    def is_late_scaling_finisher(cfg: CardConfig) -> bool:
        """Return True for limit=1 finishers whose score scales with total cardsUsed (e.g. #813/#814)."""
        return cfg.limit_count == 1 and (
            cfg.base_id == 813 or "cardsUsed" in (cfg.dsl_actions or "")
        )

    @staticmethod
    def is_double_effect_enabler(cfg: CardConfig) -> bool:
        """Return True for cards that grant doubleCardEffectCards+= (e.g. #162/#163)."""
        return cfg.base_id == 162 or "doubleCardEffectCards+=" in (cfg.dsl_actions or "")

    @staticmethod
    def is_score_buff_drink(drink: DrinkConfig) -> bool:
        """Return True for drinks granting independent score-gain multiplier (`setScoreBuff`, e.g. #12, #14)."""
        return drink.drink_id in (12, 14) or "setscorebuff(" in (drink.dsl_actions or "").lower()

    @staticmethod
    def score_buff_drink_duration(drink: DrinkConfig) -> int:
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

    @staticmethod
    def is_timed_score_buff_card(cfg: CardConfig) -> bool:
        """Return True for single-use burst cards granting timed `setScoreBuff` (e.g. #775/#776 話題沸騰)."""
        if cfg.base_id == 775:
            return True
        if cfg.limit_count == 1:
            return bool(re.search(r"setscorebuff\([^,]+,\s*(\d+)\)", (cfg.dsl_actions or "").lower()))
        return False

    @staticmethod
    def is_tutor_drink(drink: DrinkConfig) -> bool:
        """Return True for drinks that move a chosen card from draw/discard to hand (e.g. #22 初星黒酢)."""
        return drink.drink_id == 22 or "moveselectedtohand" in (drink.dsl_actions or "").lower()

    @staticmethod
    def is_trouble_card(card_cfg: CardConfig) -> bool:
        """Return True for trouble/interference cards (e.g. #23 眠気)."""
        return card_cfg.card_type == "trouble" or card_cfg.base_id == 23 or card_cfg.card_id == 23

    @staticmethod
    def drink_draw_count(drink: DrinkConfig) -> int:
        """Count immediate cards added/drawn to hand by a drink (ignoring delayed `at:` triggers)."""
        dsl_l = (drink.dsl_actions or "").lower()
        if HeadlessBattleSimulator.is_tutor_drink(drink) or "moveselectedtohand" in dsl_l:
            return 1
        top_part = re.split(r"\bat\s*:", re.sub(r"\{[^}]*\}", "", dsl_l), maxsplit=1)[0]
        total = sum(int(m) for m in re.findall(r"drawcard\s*\(\s*(\d+)\s*\)", top_part))
        rem_top = re.sub(r"drawcard\s*\([^)]*\)", "", top_part)
        total += len(re.findall(r"\bdrawcard\b", rem_top))
        return total

    def can_use_drink(self, drink_cfg: DrinkConfig, *, hand_size: Optional[int] = None) -> bool:
        """Core engine legality rule: forbid wasting tutor (#22) or pure-draw drinks at 5-card hand cap."""
        h_sz = len(self.hand) if hand_size is None else int(hand_size)
        if h_sz >= self.MAX_HAND_SIZE:
            if self.is_tutor_drink(drink_cfg):
                return False
            if self.drink_draw_count(drink_cfg) > 0 and not self.is_score_buff_drink(drink_cfg):
                return False
        return True

    def is_trouble_exile_mandatory(self, *, turns_remaining: Optional[int] = None) -> bool:
        """Core engine rule: on non-final turns without 2x / active retrigger armed, exiling #23 眠気 strictly beats blank SKIP."""
        if self.double_card_effect_cards != 0 or self.has_armed_active_retrigger(same_turn_only=True):
            return False
        rem = self.turns_remaining if turns_remaining is None else int(turns_remaining)
        return rem >= 2

    def should_defer_drink_for_hand_cap(
        self,
        drink_cfg: DrinkConfig,
        *,
        hand_size: Optional[int] = None,
        has_playable_non_finisher: Optional[bool] = None,
    ) -> bool:
        """Defer draw/tutor drinks that would overflow the 5-card hand cap when cards can be played first."""
        h_sz = len(self.hand) if hand_size is None else int(hand_size)
        if not self.can_use_drink(drink_cfg, hand_size=h_sz):
            return True
        need = self.drink_draw_count(drink_cfg)
        if need <= 0:
            return False
        if self.is_tutor_drink(drink_cfg):
            return h_sz >= self.MAX_HAND_SIZE
        if h_sz + need <= self.MAX_HAND_SIZE:
            return False
        can_play_first = (
            bool(has_playable_non_finisher)
            if has_playable_non_finisher is not None
            else any(
                self.can_play_card(inst)
                and not self.is_late_scaling_finisher(inst.cfg)
                and not self.is_trouble_card(inst.cfg)
                for inst in self.hand
            )
        )
        if self.is_score_buff_drink(drink_cfg) and self.turns_remaining <= 1:
            return bool(can_play_first)
        return bool(h_sz >= self.MAX_HAND_SIZE or can_play_first)

    def should_prune_blank_end_turn(
        self,
        *,
        has_non_trouble_card: bool,
        has_trouble_card: bool,
        stamina: Optional[int] = None,
        turns_remaining: Optional[int] = None,
    ) -> bool:
        """Return True when blank SKIP/end_turn must be pruned because useful plays or trouble exile exist."""
        if self.double_card_effect_cards != 0 or self.has_armed_active_retrigger(same_turn_only=True):
            return False
        st = self.stamina if stamina is None else int(stamina)
        rem = self.turns_remaining if turns_remaining is None else int(turns_remaining)
        if has_trouble_card and self.is_trouble_exile_mandatory(turns_remaining=rem):
            return True
        if has_non_trouble_card and (st >= 6 or rem <= 1):
            return True
        return False

    def should_hold_for_finisher(
        self,
        *,
        card_cfg: Optional[CardConfig] = None,
        drink_cfg: Optional[DrinkConfig] = None,
        fin_window: int,
        has_813: Optional[bool] = None,
    ) -> bool:
        """Hold finishers, 2x enablers, tutor drinks, and timed score-buff multipliers until the late burst window."""
        if fin_window <= 0 or self.double_card_effect_cards > 0:
            return False
        has_fin = has_813 if has_813 is not None else self.has_unexhausted_card_family(813)
        if card_cfg is not None:
            if self.turns_remaining > fin_window:
                if self.is_late_scaling_finisher(card_cfg):
                    return True
                if has_fin and self.is_double_effect_enabler(card_cfg):
                    return True
            if has_fin and self.is_timed_score_buff_card(card_cfg) and self.turns_remaining > max(3, fin_window + 1):
                return True
        if drink_cfg is not None and has_fin:
            if self.is_tutor_drink(drink_cfg) and self.turns_remaining > fin_window:
                return True
            if self.is_score_buff_drink(drink_cfg):
                dur = self.score_buff_drink_duration(drink_cfg)
                if dur > 0 and self.turns_remaining > dur:
                    return True
        return False

    def has_unexhausted_card_family(self, base_id: int) -> bool:
        """Return True if any card with `base_id` remains in hand, draw pile, or discard pile."""
        bid = int(base_id)
        for zone in (self.hand, self.draw_pile, self.discard_pile):
            if any(c.cfg.base_id == bid for c in zone):
                return True
        return False

    def is_safe_dominating_free_card(self, inst: CardInstance) -> bool:
        """Return True when playing `inst` now strictly dominates spending the final action on a turn-ending card."""
        if not self.can_play_card(inst) or not self.is_free_action_card(inst):
            return False
        cfg_c = inst.cfg
        # Any non-L card (Mental or Active) consumes double_card_effect_cards in _use_card
        if self.double_card_effect_cards > 0 and cfg_c.rarity != "L":
            return False
        # Do not force a lower-value Active card to consume #25 Active-only retrigger
        if self.has_armed_active_retrigger() and cfg_c.card_type == "active" and cfg_c.rarity != "L":
            return False
        # Never auto-force #162/#163 double-effect enablers prematurely
        if self.is_double_effect_enabler(cfg_c):
            return False
        # Do not force a limit_count==1 card early when one of its conditional bonus blocks is currently False
        if cfg_c.limit_count == 1 and self.turns_remaining > 2:
            snap = self._snapshot_condition_state(card=inst, is_direct=False)
            for blk in (cfg_c.parsed_effects.get("actions") or []):
                if blk.get("conditions") and not all(self._eval_condition(c, snap=snap) for c in blk["conditions"]):
                    return False
        c_var, c_amt = self._effective_cost(cfg_c)
        if c_var == "motivation" and c_amt > 0 and self.turns_remaining > 4:
            mot_now = int(self.buffs.get("motivation", 0) or 0)
            rem_mot = mot_now - c_amt
            has_eng = self.has_motivation_engine()
            is_repeatable_pure_active = (
                cfg_c.card_type == "active"
                and cfg_c.limit_count is None
                and "goodImpressionTurns+=" not in (cfg_c.dsl_actions or "")
            )
            if has_eng:
                if is_repeatable_pure_active and rem_mot < 2:
                    return False
            else:
                if is_repeatable_pure_active:
                    if rem_mot < 5:
                        return False
                elif rem_mot < 1:
                    return False
        if c_var == "stamina" and self.turns_remaining > 2 and (self.stamina - c_amt) <= 2:
            return False
        non_free_playable = [
            o for o in self.hand
            if o.uid != inst.uid and self.can_play_card(o) and not self.is_free_action_card(o)
        ]
        if non_free_playable and c_amt > 0:
            if c_var in ("motivation", "goodImpressionTurns", "concentration", "goodConditionTurns"):
                rem_val = int(self.buffs.get(c_var, 0) or 0) - c_amt
                if not any(
                    (ov := self._effective_cost(o.cfg))[0] != c_var or rem_val >= ov[1]
                    for o in non_free_playable
                ):
                    return False
                if c_var == "goodConditionTurns":
                    for o in non_free_playable:
                        m_gc = re.search(r"goodConditionTurns\s*>=\s*(\d+)", o.cfg.dsl_conditions or "")
                        if m_gc and rem_val < int(m_gc.group(1)):
                            return False
            elif c_var == "stamina":
                rem_hp = self.stamina - c_amt
                if not any(
                    (ov := self._effective_cost(o.cfg))[0] != "stamina" or rem_hp >= ov[1]
                    for o in non_free_playable
                ):
                    return False
        return True

    def legal_actions(self, *, include_voluntary_skip: bool = True) -> List[Tuple[str, int, str]]:
        """Return list of (kind, index, label): kind in {'card', 'drink', 'end_turn'}."""
        acts: List[Tuple[str, int, str]] = []
        has_playable_trouble = False
        for idx, inst in enumerate(self.hand):
            if self.can_play_card(inst):
                c_var, c_amt = self._effective_cost(inst.cfg)
                acts.append(("card", idx, f"{inst.cfg.name}({c_var}-{c_amt})"))
                if self.is_trouble_card(inst.cfg):
                    has_playable_trouble = True
        for didx, dr in enumerate(self.available_drinks):
            if not self.can_use_drink(dr):
                continue
            acts.append(("drink", didx, f"{dr.name}(#{dr.drink_id})"))
        forbid_skip_for_trouble = has_playable_trouble and self.is_trouble_exile_mandatory()
        if (not forbid_skip_for_trouble and include_voluntary_skip) or not any(a[0] == "card" for a in acts):
            acts.append(("end_turn", -1, "END_TURN"))
        return acts

    def play_card(self, hand_idx: int) -> None:
        inst = self.hand.pop(hand_idx)
        self._use_card(inst, using_free=False)

    def _use_card(self, inst: CardInstance, *, using_free: bool = False) -> None:
        cfg = inst.cfg
        prev_used = self.used_card
        self.used_card = inst
        self.in_play_card = inst
        if self._record_logs:
            self.cards_played_log.append(f"T{self.turn}:{cfg.name}(再演)" if using_free else f"T{self.turn}:{cfg.name}")

        try:
            self._fire_phase("beforeCardUsed", is_direct=False)
            pre_cost_snap = self._snapshot_condition_state(card=inst, is_direct=False)

            skip_actions = False
            self._buff_cost_consumed = False
            memory_draw_count = 0
            if using_free:
                live_snap = self._snapshot_condition_state(card=inst, is_direct=False)
                for cond_node in cfg.parsed_effects.get("conditions") or []:
                    if not self._eval_condition(cond_node, snap=live_snap):
                        skip_actions = True
                        break
            else:
                outer_phase = self.phase
                self.phase = "processCost"
                try:
                    cost_var, eff_amt, nullify_key = self._effective_cost_with_nullify(cfg)
                    if nullify_key is not None:
                        self._set_buff_value(nullify_key, int(self.buffs.get(nullify_key, 0) or 0) - 1)
                    if cost_var == "cost" and eff_amt > 0:
                        shield_used = min(self.genki, eff_amt)
                        self.genki -= shield_used
                        hp_used = eff_amt - shield_used
                        self.stamina = max(0, self.stamina - hp_used)
                    elif cost_var == "stamina" and eff_amt > 0:
                        self.stamina = max(0, self.stamina - eff_amt)
                    elif cost_var != "cost" and eff_amt > 0:
                        self._set_buff_value(cost_var, int(self.buffs.get(cost_var, 0) or 0) - eff_amt)
                        if cost_var in CONSUMABLE_BUFF_FIELDS:
                            self._buff_cost_consumed = True
                finally:
                    self.phase = outer_phase

                first_free = False
                if self.has_unused_memory_extra_use(inst):
                    bid = inst.cfg.base_id
                    new_cnt = self.memory_extra_use_counts.get(bid, 0) + 1
                    self.memory_extra_use_counts[bid] = new_cnt
                    effective_limit = int(
                        getattr(inst.cfg, "memory_extra_use_limit", 0)
                        or (1 if inst.cfg.free_extra_use_first_play else 0)
                    )
                    if new_cnt >= effective_limit:
                        self.memory_extra_used_uids.add(inst.uid)
                        self.memory_extra_used_base_ids.add(bid)
                    first_free = True
                    memory_draw_count = int(getattr(inst.cfg, "memory_draw_on_play", 0) or 0)
                    if self._record_logs:
                        draw_note = f"+抽{memory_draw_count}" if memory_draw_count > 0 else ""
                        self.memory_triggers_log.append(
                            f"T{self.turn}:{cfg.name}[uid={inst.uid}]->再動+1使用数{draw_note}({new_cnt}/{effective_limit})"
                        )
                if not first_free:
                    self.card_uses_remaining = max(0, self.card_uses_remaining - 1)

                if self.has_unused_memory_exile(inst):
                    self.memory_exile_used_uids.add(inst.uid)
                    self.memory_exile_used_base_ids.add(inst.cfg.base_id)
                    exiled_name = self._exile_one_trouble()
                    if self._record_logs:
                        self.memory_triggers_log.append(
                            f"T{self.turn}:{cfg.name}[uid={inst.uid}]->除外:{exiled_name or '無トラブル'}"
                        )

            # cardUsed / activeCardUsed / mentalCardUsed (CardManager.js L561-578)
            self._fire_phase("cardUsed", condition_snapshot=pre_cost_snap, is_direct=False)
            if cfg.card_type == "active":
                self._fire_phase("activeCardUsed", condition_snapshot=pre_cost_snap, is_direct=False)
            elif cfg.card_type == "mental":
                self._fire_phase("mentalCardUsed", condition_snapshot=pre_cost_snap, is_direct=False)

            if not skip_actions:
                outer_phase = self.phase
                self.phase = "processCard"
                try:
                    actions = cfg.parsed_effects.get("actions") or []
                    if self.double_card_effect_cards > 0 and cfg.rarity != "L":
                        self.double_card_effect_cards -= 1
                        self._trigger_action_blocks(actions, owner=inst, is_from_card=True)
                    self._trigger_action_blocks(actions, owner=inst, is_from_card=True)
                finally:
                    self.concentration_multiplier = 1.0
                    self.good_condition_multiplier = 1.0
                    self.phase = outer_phase

            self.cards_used += 1
            self.turn_cards_used += 1
            if cfg.card_type == "active":
                self.active_cards_used += 1

            post_action_snap = self._snapshot_condition_state(card=inst, is_direct=False)

            # Route played card AFTER actions complete (CardManager.js L602-615)
            if cfg.limit_count == 1:
                self.exhaust_pile.append(inst)
                self._fire_phase("cardRemoved", condition_snapshot=post_action_snap, is_direct=False)
            else:
                self.discard_pile.append(inst)

            if memory_draw_count > 0:
                self.draw_cards(memory_draw_count)

            if self._buff_cost_consumed:
                self._fire_phase("buffCostConsumed", condition_snapshot=pre_cost_snap, is_direct=False)
                self._buff_cost_consumed = False

            self._fire_phase("afterCardUsed", condition_snapshot=post_action_snap, is_direct=False)
            if cfg.card_type == "active":
                self._fire_phase("afterActiveCardUsed", condition_snapshot=post_action_snap, is_direct=False)
            elif cfg.card_type == "mental":
                self._fire_phase("afterMentalCardUsed", condition_snapshot=post_action_snap, is_direct=False)

            self.last_used_card = inst
        finally:
            self.used_card = prev_used
            self.in_play_card = prev_used

        if not using_free and self.card_uses_remaining <= 0:
            self.end_turn()

    def use_drink(self, drink_idx: int) -> None:
        dr = self.available_drinks.pop(drink_idx)
        if self._record_logs:
            self.drinks_used_log.append(f"T{self.turn}:{dr.name}")
        fake_owner = CardInstance(
            uid=-dr.drink_id,
            cfg=CardConfig(
                card_id=-dr.drink_id,
                name=dr.name,
                base_id=-dr.drink_id,
                upgraded_id=None,
                card_type="drink",
                rarity=dr.rarity,
                upgraded=False,
                limit_count=1,
                force_initial_hand=False,
                p_idol_id=None,
                is_signature=False,
                source_type="drink",
            ),
        )
        outer_phase = self.phase
        self.phase = "useDrink"
        try:
            self._trigger_action_blocks(
                dr.parsed_effects.get("actions") or [],
                owner=fake_owner,
                is_from_card=False,
            )
        finally:
            self.phase = outer_phase

    def end_turn(self) -> None:
        # 0. Voluntary SKIP (remaining actions > 0) recovers +2 HP and fires turnSkipped
        if self.card_uses_remaining > 0:
            self.stamina = min(self.max_stamina, self.stamina + 2)
            self._fire_phase("turnSkipped", is_direct=False)

        # 1. Fire endOfTurn listeners BEFORE discarding hand (TurnManager.js L138 vs L167)
        self._fire_phase("endOfTurn", is_direct=False)

        # 2. Settle Good Impression score at end of turn
        gi = int(self.buffs.get("goodImpressionTurns", 0) or 0)
        if gi > 0:
            n_ticks = 1 + self.extra_gi_eot_triggers()
            raw_tick = gi * (1.0 + max(0.0, self.total_gi_effect_buff()))
            for _ in range(n_ticks):
                self.score += self._compute_final_score_gain(raw_tick)

        # 3. Decrement turn-based stat durations (respecting EOT fresh flags) & timed buffs
        for bk in EOT_DECREMENT_FIELDS:
            if bk in self.fresh_turn_stat_buffs:
                self.fresh_turn_stat_buffs.discard(bk)
            elif self.buffs.get(bk, 0) > 0:
                self.buffs[bk] -= 1

        self.timed_gi_effect_buffs = self._step_timed_buffs(self.timed_gi_effect_buffs)
        self.timed_gi_gain_buffs = self._step_timed_buffs(self.timed_gi_gain_buffs)
        self.timed_score_buffs = self._step_timed_buffs(self.timed_score_buffs)
        self.timed_extra_gi_eot = self._step_timed_buffs(self.timed_extra_gi_eot)
        self.timed_conc_effect_buffs = self._step_timed_buffs(self.timed_conc_effect_buffs)
        self.timed_gc_gain_buffs = self._step_timed_buffs(self.timed_gc_gain_buffs)
        self.timed_gc_addition_buffs = self._step_timed_buffs(self.timed_gc_addition_buffs)
        self.timed_conc_gain_buffs = self._step_timed_buffs(self.timed_conc_gain_buffs)

        # 4. Reset one-turn fields & update trigger ttl/delay & prune expired triggers (ttl < 0)
        self.card_uses_remaining = 0
        self.turn_cards_used = 0
        surviving: List[ActiveTrigger] = []
        for tr in self.triggers:
            if tr.ttl is not None and tr.ttl != -1:
                tr.ttl -= 1
            if tr.delay is not None and tr.delay != -1:
                tr.delay -= 1
            if tr.limit is not None and tr.triggers_count >= tr.limit:
                continue
            if tr.ttl is not None and tr.ttl < 0:
                continue
            surviving.append(tr)
        self.triggers = surviving

        # 5. Discard remaining hand (TurnManager.js L167)
        while self.hand:
            self.discard_pile.append(self.hand.pop(0))

        if self._record_logs:
            spec = self.current_turn_spec()
            self.turn_logs.append(
                f"T{self.turn}({spec.attribute} {spec.multiplier*100:.0f}%) end: "
                f"S={self.score}, HP={self.stamina}/{self.max_stamina}, G={self.genki}, "
                f"GI={self.buffs['goodImpressionTurns']}(+{int(round(self.total_gi_effect_buff()*100))}%), "
                f"Mot={self.buffs['motivation']}, Ex={len(self.exhaust_pile)}, Dis={len(self.discard_pile)}, Draw={len(self.draw_pile)}"
            )

        # 6. Advance turn if remaining
        self.turns_elapsed += 1
        self.turns_remaining -= 1
        if self.turns_remaining <= 0:
            return
        self.turn += 1
        self._start_turn(is_first_turn=False)

    def _exile_one_trouble(self) -> Optional[str]:
        candidates: List[Tuple[List[CardInstance], int]] = []
        for zone in (self.hand, self.draw_pile, self.discard_pile):
            for idx, item in enumerate(zone):
                if item.cfg.card_type == "trouble" or item.cfg.card_id == 23 or item.cfg.base_id == 23:
                    candidates.append((zone, idx))
        if not candidates:
            return None
        zone, idx = self.rng.choice(candidates)
        exiled = zone.pop(idx)
        self.exhaust_pile.append(exiled)
        self._fire_phase("cardRemoved", is_direct=False)
        return exiled.cfg.name

    def _trigger_action_blocks(
        self,
        blocks: List[Dict[str, Any]],
        *,
        owner: CardInstance,
        is_from_card: bool,
        cond_snap: Optional[ConditionSnapshot] = None,
    ) -> None:
        if not blocks:
            return
        snap = cond_snap or self._snapshot_condition_state(
            card=owner if is_from_card else None,
        )
        for blk in blocks:
            if blk.get("phase"):
                phase_name = str(blk["phase"])
                limit_val = int(blk["limit"]) if blk.get("limit") is not None else None
                blk_actions = list(blk.get("actions") or [])
                blk_effects = list(blk.get("effects") or [])
                self.triggers.append(
                    ActiveTrigger(
                        owner_name=owner.cfg.name,
                        owner_uid=owner.uid,
                        phase=phase_name,
                        conditions=list(blk.get("conditions") or []),
                        actions=blk_actions,
                        effects=blk_effects,
                        limit=limit_val,
                        ttl=int(blk["ttl"]) if blk.get("ttl") is not None else None,
                        delay=int(blk["delay"]) if blk.get("delay") is not None else None,
                        is_card_reservation=(is_from_card and phase_name == "turn" and limit_val == 1),
                        grants_double_card_effect=_actions_grant_double_card_effect(blk_actions, blk_effects),
                    )
                )
                continue

            if blk.get("conditions"):
                if not all(self._eval_condition(c, snap=snap) for c in blk["conditions"]):
                    continue

            if blk.get("actions"):
                for act in blk["actions"]:
                    self._exec_single_action(act, owner=owner)

            if blk.get("effects"):
                self._trigger_action_blocks(
                    blk["effects"],
                    owner=owner,
                    is_from_card=is_from_card,
                    cond_snap=snap,
                )

            if "type" in blk:
                self._exec_single_action(blk, owner=owner)

    def _fire_phase(
        self,
        phase: str,
        *,
        condition_snapshot: Optional[ConditionSnapshot] = None,
        registered_before: Optional[int] = None,
        is_direct: Optional[bool] = None,
    ) -> None:
        outer_phase = self.phase
        outer_parent = self.parent_phase
        outer_direct = self._current_change_is_direct
        self.parent_phase = outer_phase
        self.phase = phase
        if is_direct is not None:
            self._current_change_is_direct = is_direct
        else:
            self._current_change_is_direct = (outer_phase in DIRECT_CHANGE_PHASES) or self._active_reservation_is_card

        try:
            snapshot_triggers = (
                list(self.triggers[:registered_before])
                if registered_before is not None
                else list(self.triggers)
            )
            cond_snap = condition_snapshot or self._snapshot_condition_state(
                card=self.used_card,
                is_direct=self._current_change_is_direct,
            )

            # Stage P-items firing at cardUsed (#404 咲季の完全食・改)
            if phase == "cardUsed":
                for it in self.config.p_items:
                    if it.mode != "stage" or it.trigger_on_nth_card_in_turn <= 0:
                        continue
                    n_mod = it.trigger_on_nth_card_in_turn
                    if (self.turn_cards_used % n_mod) != (n_mod - 1):
                        continue
                    if it.max_triggers is not None and self.p_item_trigger_counts[it.item_id] >= it.max_triggers:
                        continue
                    self.p_item_trigger_counts[it.item_id] += 1
                    if it.grant_perm_gi_effect_buff > 0:
                        self.perm_gi_effect_buff += it.grant_perm_gi_effect_buff
                    if it.grant_indirect_gi > 0:
                        self._add_good_impression_indirect(it.grant_indirect_gi)

            # Stage P-items firing at direct goodImpressionTurnsIncreased (#509 星々を統べる輝き+)
            if phase == "goodImpressionTurnsIncreased" and self._current_change_is_direct:
                is_fresh = not self.in_unfresh_phase
                for it in self.config.p_items:
                    if it.mode != "stage" or it.trigger_every_n_direct_gi <= 0:
                        continue
                    if it.max_triggers is not None and self.p_item_trigger_counts[it.item_id] >= it.max_triggers:
                        continue
                    self.p_item_gi_counters[it.item_id] += 1
                    if self.p_item_gi_counters[it.item_id] % it.trigger_every_n_direct_gi == 0:
                        self.p_item_trigger_counts[it.item_id] += 1
                        if it.stamina_cost_on_trigger > 0:
                            self.stamina = max(0, self.stamina - it.stamina_cost_on_trigger)
                        if it.grant_extra_gi_eot_triggers > 0:
                            self.timed_extra_gi_eot.append(
                                (it.grant_extra_gi_eot_triggers, max(1, it.grant_extra_gi_eot_ttl), is_fresh)
                            )
                        if it.grant_gi_effect_buff > 0:
                            if it.grant_gi_effect_ttl is None:
                                self.perm_gi_effect_buff += it.grant_gi_effect_buff
                            else:
                                self.timed_gi_effect_buffs.append(
                                    (it.grant_gi_effect_buff, max(1, it.grant_gi_effect_ttl), is_fresh)
                                )

            # Stage P-items firing at startOfTurn (e.g. 包まれるやさしさ: Vi turn & 好調>=10)
            if phase == "startOfTurn":
                for it in self.config.p_items:
                    if it.mode != "stage" or not it.trigger_attr_turn:
                        continue
                    if not self.current_turn_spec().attribute.lower().startswith(it.trigger_attr_turn.lower()):
                        continue
                    if int(self.buffs.get("goodConditionTurns", 0) or 0) < it.require_min_gc:
                        continue
                    if it.max_triggers is not None and self.p_item_trigger_counts[it.item_id] >= it.max_triggers:
                        continue
                    self.p_item_trigger_counts[it.item_id] += 1
                    if it.grant_perfect_condition > 0:
                        self._set_buff_value(
                            "perfectConditionTurns",
                            int(self.buffs.get("perfectConditionTurns", 0) or 0) + it.grant_perfect_condition,
                        )
                    if it.grant_genki > 0:
                        self.genki += it.grant_genki

            # Stage P-items firing at afterCardUsed (e.g. 憧れ続けた輝き: 好調>=8, every 4th 好調 card)
            if phase == "afterCardUsed" and self.used_card is not None:
                used_dsl = self.used_card.cfg.dsl_actions or ""
                if "goodConditionTurns" in used_dsl:
                    cur_gc = int(self.buffs.get("goodConditionTurns", 0) or 0)
                    for it in self.config.p_items:
                        if it.mode != "stage" or it.trigger_every_n_gc_cards <= 0:
                            continue
                        if cur_gc < it.require_min_gc:
                            continue
                        if it.max_triggers is not None and self.p_item_trigger_counts[it.item_id] >= it.max_triggers:
                            continue
                        self.p_item_sense_gc_counters[it.item_id] = self.p_item_sense_gc_counters.get(it.item_id, 0) + 1
                        if (
                            self.p_item_sense_gc_counters[it.item_id] % it.trigger_every_n_gc_cards == 0
                            and self.stamina >= it.stamina_cost_on_trigger
                        ):
                            self.p_item_trigger_counts[it.item_id] += 1
                            if it.stamina_cost_on_trigger > 0:
                                self.stamina = max(0, self.stamina - it.stamina_cost_on_trigger)
                            if it.grant_perfect_condition > 0:
                                self._set_buff_value(
                                    "perfectConditionTurns",
                                    int(self.buffs.get("perfectConditionTurns", 0) or 0) + it.grant_perfect_condition,
                                )
                            if it.grant_card_uses > 0:
                                self.card_uses_remaining += it.grant_card_uses
                            if it.grant_draw_cards > 0:
                                self.draw_cards(it.grant_draw_cards)

            for tr in snapshot_triggers:
                if tr.phase != phase:
                    continue
                if tr.limit is not None and tr.triggers_count >= tr.limit:
                    continue
                if tr.ttl is not None and tr.ttl < 0:
                    continue
                if tr.delay is not None and tr.delay >= 0:
                    continue
                if tr.grants_double_card_effect and self.double_card_effect_cards > 0:
                    continue
                if tr.conditions and not all(self._eval_condition(c, snap=cond_snap, trig=tr) for c in tr.conditions):
                    continue
                tr.triggers_count += 1
                prev_res = self._active_reservation_is_card
                self._active_reservation_is_card = tr.is_card_reservation
                try:
                    for act in tr.actions:
                        self._exec_single_action(act, trig=tr)
                    if tr.effects:
                        owner_inst = self._active_instances().get(tr.owner_uid) or self.used_card
                        if owner_inst is not None:
                            self._trigger_action_blocks(
                                tr.effects,
                                owner=owner_inst,
                                is_from_card=tr.is_card_reservation,
                                cond_snap=cond_snap,
                            )
                finally:
                    self.concentration_multiplier = 1.0
                    self.good_condition_multiplier = 1.0
                    self._active_reservation_is_card = prev_res
        finally:
            self.phase = outer_phase
            self.parent_phase = outer_parent
            self._current_change_is_direct = outer_direct

    def _add_good_impression_direct(self, base_amt: int) -> None:
        if base_amt <= 0:
            return
        gain_mult = 1.0 + max(0.0, self.total_gi_gain_buff())
        gained = _safe_ceil(base_amt * gain_mult)
        cur = int(self.buffs.get("goodImpressionTurns", 0) or 0)
        self._set_buff_value("goodImpressionTurns", cur + gained)
        self._fire_phase("goodImpressionTurnsIncreased", is_direct=True)

    def _add_good_impression_indirect(self, base_amt: int) -> None:
        if base_amt <= 0:
            return
        gain_mult = 1.0 + max(0.0, self.total_gi_gain_buff())
        gained = _safe_ceil(base_amt * gain_mult)
        cur = int(self.buffs.get("goodImpressionTurns", 0) or 0)
        self._set_buff_value("goodImpressionTurns", cur + gained)

    def _exec_single_action(
        self,
        act: Dict[str, Any],
        owner: Optional[CardInstance] = None,
        trig: Optional[ActiveTrigger] = None,
    ) -> None:
        atype = act.get("type")
        if atype == "assignment":
            lhs = str(act.get("lhs") or "")
            lhs_name = act.get("lhsName")
            op = act.get("op")
            val = float(self._eval_expr(act.get("rhs", 0), trig=trig))

            if lhs == "effectCounter":
                if trig is not None:
                    ouid = trig.owner_uid
                    if lhs_name == "turn":
                        cur = self.card_turn_counters.get(ouid, 0)
                        self.card_turn_counters[ouid] = int(val) if op == "=" else (cur + int(val))
                    else:
                        cur = self.card_effect_counters.get(ouid, 0)
                        self.card_effect_counters[ouid] = int(val) if op == "=" else (cur + int(val))
                return

            if lhs in DEBUFF_FIELDS and ((op == "+=" and val > 0) or (op == "=" and val > 0)):
                if self.buffs.get("nullifyDebuff", 0) > 0:
                    self.buffs["nullifyDebuff"] -= 1
                    return

            if lhs == "score" and op == "+=":
                self.score += self._compute_final_score_gain(val, apply_sense_mods=True)
                return

            if lhs == "concentrationMultiplier":
                if op == "=":
                    self.concentration_multiplier = max(0.0, float(val))
                elif op == "+=":
                    self.concentration_multiplier = max(0.0, self.concentration_multiplier + float(val))
                elif op == "*=":
                    self.concentration_multiplier = max(0.0, self.concentration_multiplier * float(val))
                return

            if lhs == "goodConditionTurnsMultiplier":
                if op == "=":
                    self.good_condition_multiplier = max(0.0, float(val))
                elif op == "+=":
                    self.good_condition_multiplier = max(0.0, self.good_condition_multiplier + float(val))
                elif op == "*=":
                    self.good_condition_multiplier = max(0.0, self.good_condition_multiplier * float(val))
                return

            if lhs == "fixedGenki" and op == "+=":
                self.genki = max(0, self.genki + int(val))
                return

            if lhs == "genki" and op == "+=":
                if self.buffs.get("nullifyGenkiTurns", 0) > 0:
                    return
                add_g = int(val)
                if add_g > 0:
                    raw_g = add_g + max(0, int(self.buffs.get("motivation", 0) or 0))
                    if self.buffs.get("uneaseTurns", 0) > 0:
                        raw_g = _safe_ceil(raw_g * 0.67)
                    self.genki += raw_g
                return

            if lhs == "goodImpressionTurns":
                if op == "+=":
                    is_direct = (self.phase in DIRECT_CHANGE_PHASES) or self._active_reservation_is_card
                    if is_direct:
                        self._add_good_impression_direct(int(val))
                    else:
                        self._add_good_impression_indirect(int(val))
                elif op == "*=":
                    cur = int(self.buffs.get("goodImpressionTurns", 0) or 0)
                    self._set_buff_value("goodImpressionTurns", _safe_ceil(cur * val))
                elif op == "-=":
                    cur = int(self.buffs.get("goodImpressionTurns", 0) or 0)
                    self._set_buff_value("goodImpressionTurns", cur - int(val))
                return

            if lhs == "fixedStamina":
                if op == "-=":
                    self.stamina = max(0, self.stamina - int(val))
                elif op == "+=":
                    self.stamina = min(self.max_stamina, self.stamina + int(val))
                return

            if lhs == "stamina":
                if op == "-=":
                    eff_float = -float(val)
                    if self.buffs.get("halfCostTurns", 0) > 0:
                        eff_float *= 0.5
                    if self.buffs.get("doubleCostTurns", 0) > 0:
                        eff_float *= 2.0
                    cost_amt = max(0, -math.floor(eff_float))
                    self.stamina = max(0, self.stamina - cost_amt)
                elif op == "+=":
                    self.stamina = min(self.max_stamina, self.stamina + int(val))
                return

            if lhs == "cardUsesRemaining" and op == "+=":
                self.card_uses_remaining += int(val)
                return
            if lhs == "turnsRemaining" and op == "+=":
                self.turns_remaining += int(val)
                return
            if lhs == "doubleCardEffectCards" and op == "+=":
                self.double_card_effect_cards += int(val)
                return

            if lhs == "goodConditionTurns":
                cur = int(self.buffs.get("goodConditionTurns", 0) or 0)
                is_direct = (self.phase in DIRECT_CHANGE_PHASES) or self._active_reservation_is_card
                if op == "+=":
                    if int(val) > 0:
                        gain_mult = 1.0 + max(0.0, self.total_gc_gain_buff())
                        add_b = self.total_gc_addition_buff() if is_direct else 0
                        gained = _safe_ceil((int(val) + add_b) * gain_mult)
                        self._set_buff_value("goodConditionTurns", cur + gained)
                        if is_direct:
                            self._fire_phase("goodConditionTurnsIncreased", is_direct=True)
                elif op == "-=":
                    self._set_buff_value("goodConditionTurns", cur - int(val))
                elif op == "=":
                    self._set_buff_value("goodConditionTurns", int(val))
                elif op == "*=":
                    self._set_buff_value("goodConditionTurns", _safe_ceil(cur * val))
                    if val > 1.0 and is_direct:
                        self._fire_phase("goodConditionTurnsIncreased", is_direct=True)
                return

            if lhs == "concentration":
                cur = int(self.buffs.get("concentration", 0) or 0)
                is_direct = (self.phase in DIRECT_CHANGE_PHASES) or self._active_reservation_is_card
                if op == "+=":
                    if int(val) > 0:
                        gain_mult = 1.0 + max(0.0, self.total_conc_gain_buff())
                        gained = _safe_ceil(int(val) * gain_mult)
                        self._set_buff_value("concentration", cur + gained)
                        if is_direct:
                            self._fire_phase("concentrationIncreased", is_direct=True)
                elif op == "-=":
                    self._set_buff_value("concentration", cur - int(val))
                elif op == "=":
                    self._set_buff_value("concentration", int(val))
                elif op == "*=":
                    self._set_buff_value("concentration", _safe_ceil(cur * val))
                    if val > 1.0 and is_direct:
                        self._fire_phase("concentrationIncreased", is_direct=True)
                return

            if (
                lhs in ("motivation", "nullifyDebuff", "costReduction", "costIncrease", "nullifyCostCards", "nullifyCostActiveCards")
                or lhs in EOT_DECREMENT_FIELDS
                or lhs in DEBUFF_FIELDS
                or lhs in CONSUMABLE_BUFF_FIELDS
            ):
                cur = int(self.buffs.get(lhs, 0) or 0)
                if op == "+=":
                    self._set_buff_value(lhs, cur + int(val))
                elif op == "-=":
                    self._set_buff_value(lhs, cur - int(val))
                elif op == "=":
                    self._set_buff_value(lhs, int(val))
                elif op == "*=":
                    self._set_buff_value(lhs, _safe_ceil(cur * val))
                return

        elif atype == "call":
            cname = act.get("name")
            if cname in DEBUFF_SPECIAL_ACTIONS and self.buffs.get("nullifyDebuff", 0) > 0:
                self.buffs["nullifyDebuff"] -= 1
                return
            args = [self._eval_expr(a, trig=trig) for a in (act.get("args") or [])]
            is_fresh = not self.in_unfresh_phase
            if cname == "setGoodImpressionTurnsEffectBuff":
                b_val = float(args[0])
                ttl = int(args[1]) if len(args) >= 2 else None
                if ttl is None:
                    self.perm_gi_effect_buff += b_val
                else:
                    self.timed_gi_effect_buffs.append((b_val, ttl, is_fresh))
            elif cname == "setGoodImpressionTurnsBuff":
                b_val = float(args[0])
                ttl = int(args[1]) if len(args) >= 2 else 999
                self.timed_gi_gain_buffs.append((b_val, ttl, is_fresh))
            elif cname == "setScoreBuff":
                b_val = float(args[0])
                ttl = int(args[1]) if len(args) >= 2 else None
                if ttl is None:
                    self.perm_score_buff += b_val
                else:
                    self.timed_score_buffs.append((b_val, ttl, is_fresh))
            elif cname == "setConcentrationEffectBuff":
                b_val = float(args[0])
                ttl = int(args[1]) if len(args) >= 2 else None
                if ttl is None:
                    self.perm_conc_effect_buff += b_val
                else:
                    self.timed_conc_effect_buffs.append((b_val, ttl, is_fresh))
            elif cname == "setConcentrationBuff":
                b_val = float(args[0])
                ttl = int(args[1]) if len(args) >= 2 else None
                if ttl is None:
                    self.perm_conc_gain_buff += b_val
                else:
                    self.timed_conc_gain_buffs.append((b_val, ttl, is_fresh))
            elif cname == "setGoodConditionTurnsBuff":
                b_val = float(args[0])
                ttl = int(args[1]) if len(args) >= 2 else None
                if ttl is None:
                    self.perm_gc_gain_buff += b_val
                else:
                    self.timed_gc_gain_buffs.append((b_val, ttl, is_fresh))
            elif cname == "setGoodConditionTurnsAdditionBuff":
                b_val = int(round(float(args[0])))
                ttl = int(args[1]) if len(args) >= 2 else 999
                self.timed_gc_addition_buffs.append((b_val, ttl, is_fresh))
            elif cname == "drawCard":
                n_draw = int(args[0]) if args else 1
                self.draw_cards(n_draw)
            elif cname == "addCardToDeck":
                cid = int(args[0])
                cfg = self._get_card_config_cached(cid)
                new_inst = self._new_instance(cfg)
                self._register_passive_effects_for_card(new_inst)
                pos = self.rng.randint(0, len(self.draw_pile))
                self.draw_pile.insert(pos, new_inst)
            elif cname == "moveRandomToHand":
                n_move = int(args[0]) if args else 1
                tgt = act.get("target")
                owner_uid = owner.uid if owner else (trig.owner_uid if trig else None)
                matching_uids = self._eval_target_ast(tgt, owner_uid=owner_uid)
                candidates: List[Tuple[List[CardInstance], CardInstance]] = []
                for zlist in (self.draw_pile, self.discard_pile, self.exhaust_pile, self.hand):
                    for inst in zlist:
                        if inst.uid in matching_uids:
                            candidates.append((zlist, inst))
                for _ in range(max(0, n_move)):
                    if not candidates or len(self.hand) >= self.MAX_HAND_SIZE:
                        break
                    pick_idx = self.rng.randrange(len(candidates))
                    zlist, moved_inst = candidates.pop(pick_idx)
                    zlist.remove(moved_inst)
                    self.hand.append(moved_inst)
                    self.moved_card = moved_inst
                    self._fire_phase("cardMovedToHand", is_direct=True)
            elif cname == "moveSelectedToHand":
                n_move = int(args[0]) if args else 1
                tgt = act.get("target")
                owner_uid = owner.uid if owner else (trig.owner_uid if trig else None)
                matching_uids = self._eval_target_ast(tgt, owner_uid=owner_uid)
                candidates = []
                for zlist in (self.draw_pile, self.discard_pile):
                    for inst in zlist:
                        if inst.uid in matching_uids:
                            candidates.append((zlist, inst))
                rarity_rank = {"SSR": 5, "SR": 4, "L": 3, "R": 2, "N": 1, "T": 0}
                has_fin_in_hand = any(self.is_late_scaling_finisher(c.cfg) for c in self.hand)
                has_162_in_hand = any(self.is_double_effect_enabler(c.cfg) for c in self.hand)
                fin_unexhausted = self.has_unexhausted_card_family(813)
                in_fin_window = (self.turns_remaining <= 2) or (self.double_card_effect_cards > 0)

                def _tutor_key(item: Tuple[List[CardInstance], CardInstance]) -> Tuple[Any, ...]:
                    c = item[1].cfg
                    if c.card_type == "trouble":
                        return (-1, 0, 0, 0, 0, 0)
                    tier = 2
                    if in_fin_window:
                        if self.is_late_scaling_finisher(c) and not has_fin_in_hand:
                            tier = 6
                        elif (
                            self.is_double_effect_enabler(c)
                            and self.double_card_effect_cards == 0
                            and not has_162_in_hand
                            and fin_unexhausted
                        ):
                            tier = 5
                        elif self.is_timed_score_buff_card(c):
                            tier = 4
                        elif c.base_id == 811:
                            tier = 3
                    else:
                        if self.is_late_scaling_finisher(c):
                            tier = 0
                        elif c.base_id == 811:
                            tier = 5
                        elif "cardUsesRemaining+=" in (c.dsl_actions or ""):
                            tier = 4
                        elif self.is_timed_score_buff_card(c):
                            tier = 3
                    return (
                        1,
                        tier,
                        1 if c.is_signature else 0,
                        rarity_rank.get(c.rarity, 1),
                        1 if ("score+=" in c.dsl_actions or "cardUsesRemaining+=" in c.dsl_actions) else 0,
                        1 if c.upgraded else 0,
                        c.card_id,
                    )

                for _ in range(max(0, n_move)):
                    if not candidates or len(self.hand) >= self.MAX_HAND_SIZE:
                        break
                    candidates.sort(key=_tutor_key, reverse=True)
                    zlist, moved_inst = candidates.pop(0)
                    zlist.remove(moved_inst)
                    self.hand.append(moved_inst)
                    self.moved_card = moved_inst
                    if moved_inst.cfg.base_id == 813:
                        has_fin_in_hand = True
                    elif moved_inst.cfg.base_id == 162:
                        has_162_in_hand = True
                    self._fire_phase("cardMovedToHand", is_direct=True)
            elif cname == "removeRandom":
                n_rem = int(args[0]) if args else 1
                tgt = act.get("target")
                owner_uid = owner.uid if owner else (trig.owner_uid if trig else None)
                matching_uids = self._eval_target_ast(tgt, owner_uid=owner_uid)
                candidates = []
                for zlist in (self.hand, self.draw_pile, self.discard_pile):
                    for inst in zlist:
                        if inst.uid in matching_uids:
                            candidates.append((zlist, inst))
                for _ in range(max(0, n_rem)):
                    if not candidates:
                        break
                    pick_idx = self.rng.randrange(len(candidates))
                    zlist, rem_inst = candidates.pop(pick_idx)
                    zlist.remove(rem_inst)
                    self.exhaust_pile.append(rem_inst)
                    self._fire_phase("cardRemoved", is_direct=False)
            elif cname == "removeDebuffs":
                n_clr = int(args[0]) if args else 1
                active_debuffs = [dk for dk in sorted(DEBUFF_FIELDS) if int(self.buffs.get(dk, 0) or 0) > 0]
                for dk in active_debuffs[: max(0, n_clr)]:
                    self.buffs[dk] = 0
            elif cname == "upgradeRandomCardInHand":
                n_up = int(args[0]) if args else 1
                elig = [inst for inst in self.hand if not inst.cfg.upgraded and inst.cfg.upgraded_id]
                self.rng.shuffle(elig)
                for inst in elig[: max(0, n_up)]:
                    inst.cfg = self._get_card_config_cached(
                        inst.cfg.upgraded_id,
                        force_initial_hand_override=inst.cfg.force_initial_hand,
                        exiles_trouble_on_play=inst.cfg.exiles_trouble_on_play,
                        free_extra_use_first_play=inst.cfg.free_extra_use_first_play,
                        memory_extra_use_limit=inst.cfg.memory_extra_use_limit,
                        memory_draw_on_play=inst.cfg.memory_draw_on_play,
                        customizations=inst.cfg.customizations,
                        custom_modifiers=inst.cfg.custom_modifiers,
                    )
            elif cname in ("useRandomCardFree", "useRandomFree"):
                owner_uid = owner.uid if owner else (trig.owner_uid if trig else None)
                matching_uids = self._eval_target_ast(act.get("target"), owner_uid=owner_uid)
                if matching_uids:
                    chosen_uid = self.rng.choice(sorted(matching_uids))
                    for zlist in (self.draw_pile, self.hand, self.discard_pile, self.exhaust_pile):
                        idx = next((i for i, c in enumerate(zlist) if c.uid == chosen_uid), None)
                        if idx is not None:
                            chosen = zlist.pop(idx)
                            self._free_card_uses += 1
                            try:
                                self._use_card(chosen, using_free=True)
                            finally:
                                self._free_card_uses -= 1
                            break
            return

        elif atype == "identifier":
            name = act.get("name")
            if name in DEBUFF_SPECIAL_ACTIONS and self.buffs.get("nullifyDebuff", 0) > 0:
                self.buffs["nullifyDebuff"] -= 1
                return
            if name == "drawCard":
                self.draw_cards(1)
            elif name == "exchangeHand":
                n_old = len(self.hand)
                while self.hand:
                    self.discard_pile.append(self.hand.pop(0))
                self.draw_cards(n_old)
            elif name == "upgradeHand":
                for inst in self.hand:
                    if not inst.cfg.upgraded and inst.cfg.upgraded_id:
                        inst.cfg = self._get_card_config_cached(
                            inst.cfg.upgraded_id,
                            force_initial_hand_override=inst.cfg.force_initial_hand,
                            exiles_trouble_on_play=inst.cfg.exiles_trouble_on_play,
                            free_extra_use_first_play=inst.cfg.free_extra_use_first_play,
                            memory_extra_use_limit=inst.cfg.memory_extra_use_limit,
                            memory_draw_on_play=inst.cfg.memory_draw_on_play,
                            customizations=inst.cfg.customizations,
                            custom_modifiers=inst.cfg.custom_modifiers,
                        )

    def _eval_target_ast(
        self,
        node: Optional[Dict[str, Any]],
        *,
        snap: Optional[ConditionSnapshot] = None,
        owner_uid: Optional[int] = None,
    ) -> set[int]:
        if not isinstance(node, dict):
            return set()
        active = self._active_instances()
        all_uids = snap.zone_uids["all"] if snap is not None else set(active.keys())
        ntype = node.get("type")

        if ntype == "binary":
            op = node.get("op")
            left = self._eval_target_ast(node.get("left"), snap=snap, owner_uid=owner_uid)
            right = self._eval_target_ast(node.get("right"), snap=snap, owner_uid=owner_uid)
            if op == "&":
                return left & right
            if op == "|":
                return left | right
            return set()

        if ntype == "unary" and node.get("op") == "!":
            operand = self._eval_target_ast(node.get("operand"), snap=snap, owner_uid=owner_uid)
            return all_uids - operand

        if ntype == "call":
            cname = node.get("name")
            args = node.get("args") or []
            if cname == "baseId" and args:
                a0 = args[0]
                want = int(a0.get("value") if a0.get("type") == "number" else a0.get("name", 0))
                return {uid for uid in all_uids if uid in active and active[uid].cfg.base_id == want}
            if cname == "id" and args:
                a0 = args[0]
                want = int(a0.get("value") if a0.get("type") == "number" else a0.get("name", 0))
                return {uid for uid in all_uids if uid in active and active[uid].cfg.card_id == want}
            return set()

        if ntype in ("identifier", "number"):
            target = str(node.get("name") if ntype == "identifier" else node.get("value"))
            if target == "this":
                return {owner_uid} if owner_uid is not None and owner_uid in all_uids else set()
            if target in ("hand", "deck", "discarded", "held", "removed"):
                if snap is not None:
                    return set(snap.zone_uids.get(target, set()))
                zone_map = {
                    "hand": self.hand,
                    "deck": self.draw_pile,
                    "discarded": self.discard_pile,
                    "removed": self.exhaust_pile,
                    "held": [],
                }
                return {c.uid for c in zone_map[target]}
            if target == "all":
                return set(all_uids)
            if target in ("active", "mental", "trouble"):
                return {uid for uid in all_uids if uid in active and active[uid].cfg.card_type == target}
            if target == "basic":
                return {uid for uid in all_uids if uid in active and "基本" in active[uid].cfg.name}
            if target == "pIdol":
                return {
                    uid
                    for uid in all_uids
                    if uid in active
                    and (
                        active[uid].cfg.source_type == "pIdol"
                        or (active[uid].cfg.is_signature and active[uid].cfg.rarity != "L")
                    )
                }
            if target in ("T", "N", "R", "SR", "SSR", "L"):
                return {uid for uid in all_uids if uid in active and active[uid].cfg.rarity == target}
            if target.isdigit():
                want = int(target)
                return {uid for uid in all_uids if uid in active and active[uid].cfg.base_id == want}

        return set()

    def _eval_expr(
        self,
        expr: Any,
        *,
        snap: Optional[ConditionSnapshot] = None,
        trig: Optional[ActiveTrigger] = None,
    ) -> float:
        if isinstance(expr, (int, float)):
            return float(expr)
        if not isinstance(expr, dict):
            return 0.0
        etype = expr.get("type")
        if etype == "number":
            return float(expr.get("value", 0))
        if etype == "identifier":
            name = str(expr.get("name") or "")
            if snap is not None:
                if name in snap.buffs:
                    return float(snap.buffs.get(name, 0) or 0)
                if name == "genki":
                    return float(snap.genki)
                if name == "stamina":
                    return float(snap.stamina)
                if name == "score":
                    return float(snap.score)
                if name == "turnsElapsed":
                    return float(snap.turns_elapsed)
                if name == "turnsRemaining":
                    return float(snap.turns_remaining)
                if name == "turnCardsUsed":
                    return float(snap.turn_cards_used)
                if name == "cardsUsed":
                    return float(snap.cards_used)
                if name == "activeCardsUsed":
                    return float(snap.active_cards_used)
                if name == "usedCardId":
                    return float(snap.used_card_id)
                if name == "usedCardBaseId":
                    return float(snap.used_card_base_id)
                if name == "movedCardId":
                    return float(snap.moved_card_id)
                if name == "movedCardBaseId":
                    return float(snap.moved_card_base_id)
                if name == "isDirectEffect":
                    return 1.0 if snap.is_direct_effect else 0.0
                if name == "effectCounter":
                    return float(snap.card_effect_counters.get(trig.owner_uid, 0) if trig is not None else 0)
            else:
                if name in self.buffs:
                    return float(self.buffs.get(name, 0) or 0)
                if name == "genki":
                    return float(self.genki)
                if name == "stamina":
                    return float(self.stamina)
                if name == "score":
                    return float(self.score)
                if name == "turnsElapsed":
                    return float(self.turns_elapsed)
                if name == "turnsRemaining":
                    return float(self.turns_remaining)
                if name == "turnCardsUsed":
                    return float(self.turn_cards_used)
                if name == "cardsUsed":
                    return float(self.cards_used)
                if name == "activeCardsUsed":
                    return float(self.active_cards_used)
                if name == "usedCardId":
                    return float(self.used_card.cfg.card_id if self.used_card is not None else 0)
                if name == "usedCardBaseId":
                    return float(self.used_card.cfg.base_id if self.used_card is not None else 0)
                if name == "movedCardId":
                    return float(self.moved_card.cfg.card_id if self.moved_card is not None else 0)
                if name == "movedCardBaseId":
                    return float(self.moved_card.cfg.base_id if self.moved_card is not None else 0)
                if name == "isDirectEffect":
                    return 1.0 if self._current_change_is_direct else 0.0
                if name == "effectCounter":
                    return float(self.card_effect_counters.get(trig.owner_uid, 0) if trig is not None else 0)

            if name == "maxStamina":
                return float(self.max_stamina)
            if name == "isVocalTurn":
                return 1.0 if self.current_turn_spec().attribute.lower().startswith("vo") else 0.0
            if name == "isDanceTurn":
                return 1.0 if self.current_turn_spec().attribute.lower().startswith("da") else 0.0
            if name == "isVisualTurn":
                return 1.0 if self.current_turn_spec().attribute.lower().startswith("vi") else 0.0
            return 0.0

        if etype == "call":
            cname = expr.get("name")
            if cname == "countCards":
                owner_uid = trig.owner_uid if trig is not None else None
                return float(len(self._eval_target_ast(expr.get("target"), snap=snap, owner_uid=owner_uid)))
            if cname == "effectCounter":
                counters = snap.card_turn_counters if snap is not None else self.card_turn_counters
                return float(counters.get(trig.owner_uid, 0) if trig is not None else 0)
            if cname == "floor":
                args = expr.get("args") or []
                return float(math.floor(self._eval_expr(args[0], snap=snap, trig=trig))) if args else 0.0
            return 0.0

        if etype == "unary":
            op = expr.get("op")
            ov = self._eval_expr(expr.get("operand"), snap=snap, trig=trig)
            if op == "!":
                return 0.0 if bool(ov) else 1.0
            if op == "-":
                return -ov
            return 0.0

        if etype == "binary":
            op = expr.get("op")
            lv = self._eval_expr(expr.get("left"), snap=snap, trig=trig)
            rv = self._eval_expr(expr.get("right"), snap=snap, trig=trig)
            if op == "*":
                return lv * rv
            if op == "/":
                return lv / rv if rv != 0 else 0.0
            if op == "+":
                return lv + rv
            if op == "-":
                return lv - rv
            if op == "%":
                return lv % rv if rv != 0 else 0.0
            if op == "&":
                return 1.0 if (bool(lv) and bool(rv)) else 0.0
            if op == "|":
                return 1.0 if (bool(lv) or bool(rv)) else 0.0
        return 0.0

    def _resolve_str_operand(self, node: Dict[str, Any], snap: ConditionSnapshot) -> Optional[str]:
        if node.get("type") != "identifier":
            return None
        name = str(node.get("name") or "")
        if name == "cardRarity":
            return snap.card_rarity
        if name == "cardType":
            return snap.card_type
        if name == "lastUsedCardType":
            return snap.last_used_card_type
        if name in ("T", "N", "R", "SR", "SSR", "L", "active", "mental", "trouble", "drink"):
            return name
        return None

    def _eval_condition(
        self,
        node: Dict[str, Any],
        *,
        snap: Optional[ConditionSnapshot] = None,
        trig: Optional[ActiveTrigger] = None,
    ) -> bool:
        cond_snap = snap or self._snapshot_condition_state()
        if "conditions" in node and isinstance(node["conditions"], list):
            return all(self._eval_condition(c, snap=cond_snap, trig=trig) for c in node["conditions"])
        ntype = node.get("type")
        if ntype == "unary" and node.get("op") == "!":
            return not self._eval_condition(node["operand"], snap=cond_snap, trig=trig)
        if ntype == "binary":
            op = node.get("op")
            if op == "&":
                return self._eval_condition(node["left"], snap=cond_snap, trig=trig) and self._eval_condition(
                    node["right"], snap=cond_snap, trig=trig
                )
            if op == "|":
                return self._eval_condition(node["left"], snap=cond_snap, trig=trig) or self._eval_condition(
                    node["right"], snap=cond_snap, trig=trig
                )
        if ntype == "comparison":
            op = node.get("op")
            l_node = node.get("left") or {}
            r_node = node.get("right") or {}
            l_str = self._resolve_str_operand(l_node, cond_snap)
            r_str = self._resolve_str_operand(r_node, cond_snap)
            if l_str is not None and r_str is not None:
                return (l_str != r_str) if op == "!=" else (l_str == r_str)
            lv = self._eval_expr(l_node, snap=cond_snap, trig=trig)
            rv = self._eval_expr(r_node, snap=cond_snap, trig=trig)
            if op == ">=":
                return lv >= rv
            if op == "<=":
                return lv <= rv
            if op == ">":
                return lv > rv
            if op == "<":
                return lv < rv
            if op == "==":
                return abs(lv - rv) < 1e-6
            if op == "!=":
                return abs(lv - rv) >= 1e-6
            return False
        if ntype == "identifier":
            name = str(node.get("name") or "")
            if name == "isDirectEffect":
                return cond_snap.is_direct_effect
            return bool(self._eval_expr(node, snap=cond_snap, trig=trig))
        return bool(self._eval_expr(node, snap=cond_snap, trig=trig))

    def step(self, kind: str, idx: int) -> None:
        if kind == "card":
            self.play_card(idx)
        elif kind == "drink":
            self.use_drink(idx)
        elif kind == "end_turn":
            self.end_turn()
        else:
            raise ValueError(f"Unknown action kind: {kind!r}")

    def build_result(self) -> BattleResult:
        return BattleResult(
            seed=self.seed,
            final_score=self.score,
            turns_played=self.turn,
            final_stamina=self.stamina,
            max_stamina=self.max_stamina,
            final_genki=self.genki,
            final_good_impression=int(self.buffs.get("goodImpressionTurns", 0) or 0),
            final_motivation=int(self.buffs.get("motivation", 0) or 0),
            final_gi_effect_buff=self.total_gi_effect_buff(),
            exhaust_count=len(self.exhaust_pile),
            reshuffle_count=self.reshuffle_count,
            cards_played=list(self.cards_played_log),
            drinks_used=list(self.drinks_used_log),
            exhausted_names=[c.cfg.name for c in self.exhaust_pile],
            turn_logs=list(self.turn_logs),
            zone_counts={
                "hand": len(self.hand),
                "draw": len(self.draw_pile),
                "discard": len(self.discard_pile),
                "exhaust": len(self.exhaust_pile),
            },
            p_item_activations={it.name: self.p_item_trigger_counts[it.item_id] for it in self.config.p_items},
            memory_triggers=list(self.memory_triggers_log),
            turn_schedule=[f"T{i+1}:{s.attribute} {s.multiplier*100:.0f}%" for i, s in enumerate(self.turn_schedule)],
        )

    def clone(self, *, seed: Optional[int] = None, keep_logs: bool = True) -> "HeadlessBattleSimulator":
        """Create a fast independent copy of the simulator state for MCTS rollouts."""
        cls = type(self)
        cloned = cls.__new__(cls)
        cloned.config = self.config
        cloned._owns_db = False
        cloned.db = self.db
        cloned._record_logs = keep_logs
        cloned._card_cfg_cache = self._card_cfg_cache
        cloned.turn_schedule = list(self.turn_schedule)

        if seed is not None:
            cloned.seed = seed
            cloned.rng = random.Random(seed)
        else:
            cloned.seed = self.seed
            cloned.rng = random.Random()
            cloned.rng.setstate(self.rng.getstate())

        cloned.turn = self.turn
        cloned.turns_elapsed = self.turns_elapsed
        cloned.turns_remaining = self.turns_remaining
        cloned.card_uses_remaining = self.card_uses_remaining
        cloned.stamina = self.stamina
        cloned.max_stamina = self.max_stamina
        cloned.genki = self.genki
        cloned.score = self.score
        cloned.buffs = dict(self.buffs)
        cloned.perm_gi_effect_buff = self.perm_gi_effect_buff
        cloned.perm_score_buff = self.perm_score_buff
        cloned.concentration_multiplier = self.concentration_multiplier
        cloned.good_condition_multiplier = self.good_condition_multiplier
        cloned.perm_conc_effect_buff = self.perm_conc_effect_buff
        cloned.perm_gc_gain_buff = self.perm_gc_gain_buff
        cloned.perm_conc_gain_buff = self.perm_conc_gain_buff
        cloned.timed_gi_effect_buffs = list(self.timed_gi_effect_buffs)
        cloned.timed_gi_gain_buffs = list(self.timed_gi_gain_buffs)
        cloned.timed_score_buffs = list(self.timed_score_buffs)
        cloned.timed_extra_gi_eot = list(self.timed_extra_gi_eot)
        cloned.timed_conc_effect_buffs = list(self.timed_conc_effect_buffs)
        cloned.timed_gc_gain_buffs = list(self.timed_gc_gain_buffs)
        cloned.timed_gc_addition_buffs = list(self.timed_gc_addition_buffs)
        cloned.timed_conc_gain_buffs = list(self.timed_conc_gain_buffs)
        cloned.fresh_turn_stat_buffs = set(self.fresh_turn_stat_buffs)
        cloned.in_unfresh_phase = self.in_unfresh_phase

        cloned.double_card_effect_cards = self.double_card_effect_cards
        cloned.turn_cards_used = self.turn_cards_used
        cloned.cards_used = self.cards_used
        cloned.active_cards_used = self.active_cards_used
        cloned.phase = self.phase
        cloned.parent_phase = self.parent_phase
        cloned._current_change_is_direct = self._current_change_is_direct
        cloned._active_reservation_is_card = self._active_reservation_is_card
        cloned._free_card_uses = self._free_card_uses
        cloned._buff_cost_consumed = self._buff_cost_consumed

        cloned.p_item_gi_counters = dict(self.p_item_gi_counters)
        cloned.p_item_sense_gc_counters = dict(self.p_item_sense_gc_counters)
        cloned.p_item_trigger_counts = dict(self.p_item_trigger_counts)
        cloned.memory_extra_used_uids = set(self.memory_extra_used_uids)
        cloned.memory_exile_used_uids = set(self.memory_exile_used_uids)
        cloned.memory_extra_used_base_ids = set(self.memory_extra_used_base_ids)
        cloned.memory_extra_use_counts = dict(self.memory_extra_use_counts)
        cloned.memory_exile_used_base_ids = set(self.memory_exile_used_base_ids)
        cloned.memory_triggers_log = list(self.memory_triggers_log) if keep_logs else []
        cloned.card_effect_counters = dict(self.card_effect_counters)
        cloned.card_turn_counters = dict(self.card_turn_counters)

        cloned.next_uid = self.next_uid
        cloned_instances: Dict[int, CardInstance] = {
            uid: CardInstance(uid=c.uid, cfg=c.cfg) for uid, c in self._all_instances.items()
        }
        cloned._all_instances = cloned_instances

        def _map_inst(c: Optional[CardInstance]) -> Optional[CardInstance]:
            if c is None:
                return None
            existing = cloned_instances.get(c.uid)
            if existing is None:
                existing = CardInstance(uid=c.uid, cfg=c.cfg)
                cloned_instances[c.uid] = existing
            return existing

        cloned.hand = [cloned_instances[c.uid] for c in self.hand]
        cloned.draw_pile = [cloned_instances[c.uid] for c in self.draw_pile]
        cloned.discard_pile = [cloned_instances[c.uid] for c in self.discard_pile]
        cloned.exhaust_pile = [cloned_instances[c.uid] for c in self.exhaust_pile]
        cloned.in_play_card = _map_inst(self.in_play_card)
        cloned.used_card = _map_inst(self.used_card)
        cloned.last_used_card = _map_inst(self.last_used_card)
        cloned.moved_card = _map_inst(self.moved_card)
        cloned.available_drinks = list(self.available_drinks)

        cloned.triggers = [
            ActiveTrigger(
                owner_name=tr.owner_name,
                owner_uid=tr.owner_uid,
                phase=tr.phase,
                conditions=tr.conditions,
                actions=tr.actions,
                effects=tr.effects,
                limit=tr.limit,
                ttl=tr.ttl,
                delay=tr.delay,
                triggers_count=tr.triggers_count,
                effect_counter=tr.effect_counter,
                turn_counter=tr.turn_counter,
                is_passive_card_effect=tr.is_passive_card_effect,
                is_from_exhaust_check=tr.is_from_exhaust_check,
                is_card_reservation=tr.is_card_reservation,
                grants_double_card_effect=tr.grants_double_card_effect,
            )
            for tr in self.triggers
        ]
        cloned.reshuffle_count = self.reshuffle_count
        cloned.cards_played_log = list(self.cards_played_log) if keep_logs else []
        cloned.drinks_used_log = list(self.drinks_used_log) if keep_logs else []
        cloned.turn_logs = list(self.turn_logs) if keep_logs else []
        return cloned

    def determinize(self, rng: random.Random, *, keep_logs: bool = False) -> "HeadlessBattleSimulator":
        """Sample a determinized world compatible with the player's current information set."""
        cloned = self.clone(seed=rng.randrange(1 << 63), keep_logs=keep_logs)
        cloned.rng.shuffle(cloned.draw_pile)
        return cloned

    def run_random_episode(self, max_steps: int = 300) -> BattleResult:
        steps = 0
        while self.turns_remaining > 0 and steps < max_steps:
            steps += 1
            acts = self.legal_actions()
            kind, idx, _label = self.rng.choice(acts)
            self.step(kind, idx)
        return self.build_result()


def describe_loaded_config(cfg: BattleConfig) -> str:
    """Format full loaded simulator parameters and deck table for human alignment."""
    lines: List[str] = []
    lines.append(f"=== 无头模拟器已加载配置 ({cfg.stage_name}) ===")
    lines.append(
        f"偶像: {cfg.idol_name} (#{cfg.p_idol_id}) | 流派: {cfg.plan} | "
        f"三维: Vo={cfg.idol_stats['vocal']} / Da={cfg.idol_stats['dance']} / Vi={cfg.idol_stats['visual']}"
    )
    lines.append(
        f"体力: {cfg.initial_stamina}/{cfg.max_stamina} | 初始元气: {cfg.initial_genki} | "
        f"初始好印象: {cfg.initial_good_impression} | 初始干劲: {cfg.initial_motivation} | 总回合: {cfg.total_turns}"
    )
    sched_str = ", ".join(f"T{i+1}:{s.attribute} {s.multiplier*100:.0f}%" for i, s in enumerate(cfg.turn_schedule))
    if cfg.randomize_turn_schedule:
        comp_str = " / ".join(
            f"{attr} {mult*100:.0f}%({cnt}T)"
            for (attr, mult), cnt in zip(cfg.turn_schedule_attr_multipliers, cfg.turn_schedule_counts)
        )
        lines.append(f"场地倍率规则: 随机分布 [{comp_str}; 末回固定最高倍率, 倒数第2/3回不出最高倍率] | 示例: {sched_str}")
    else:
        lines.append(f"逐回合场地倍率: {sched_str}")
    lines.append(f"装备 P道具 ({len(cfg.p_items)} 件):")
    for it in cfg.p_items:
        mode_tag = "战中" if it.mode == "stage" else "培育期"
        lines.append(f"  - [#{it.item_id:<3} {it.rarity}/{it.source_type}/{mode_tag}] {it.name}: {it.description}")
    lines.append(f"持有 P饮料 ({len(cfg.drinks)}):")
    for i, d in enumerate(cfg.drinks, 1):
        lines.append(f"  {i}. [#{d.drink_id}] {d.name} ({d.rarity}) | DSL={d.dsl_actions}")
    lines.append(f"卡库明细 ({len(cfg.deck)} 张):")
    for i, c in enumerate(cfg.deck, 1):
        tags: List[str] = []
        if c.force_initial_hand:
            tags.append("开局手牌")
        if c.limit_count == 1:
            tags.append("限1")
        if c.exiles_trouble_on_play:
            tags.append("メモリー:免费除外眠気(限1次)")
        if c.free_extra_use_first_play:
            lim = c.memory_extra_use_limit or 1
            draw_s = f"+抽{c.memory_draw_on_play}" if c.memory_draw_on_play > 0 else ""
            tags.append(f"メモリー:+1使用数{draw_s}(限{lim}次)")
        tag_txt = ", ".join(tags) if tags else "可重复"
        lines.append(
            f"  {i:2d}. [#{c.card_id:<3}] {c.name:<18} ({c.card_type}/{c.rarity}) | {c.dsl_cost or '0'} | {tag_txt} | DSL={c.dsl_actions}"
        )
    if cfg.extra_notes:
        lines.append("配置对齐说明:")
        for n in cfg.extra_notes:
            lines.append(f"  * {n}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run headless Gakumas battle simulator with random play")
    parser.add_argument("--stage", default="round2", choices=("round1", "round2"))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    with GakumasDB("gakumas_library.db") as db:
        cfg = build_last_run_hif_preset(db=db, stage=args.stage)
        print(describe_loaded_config(cfg))
        for r_idx in range(args.runs):
            s_val = args.seed + r_idx
            sim = HeadlessBattleSimulator(cfg, db=db, seed=s_val)
            res = sim.run_random_episode()
            print(f"\n=== 随机打牌测试 #{r_idx + 1} (seed={res.seed}, 无策略权重纯随机) ===")
            print(
                f"最终得分: {res.final_score:,} | 回合数: {res.turns_played} | "
                f"剩余HP: {res.final_stamina}/{res.max_stamina} | 元气: {res.final_genki} | "
                f"好印象: {res.final_good_impression} (+{int(round(res.final_gi_effect_buff*100))}%) | "
                f"干劲: {res.final_motivation} | 除外区: {res.exhaust_count} 张 | 洗牌: {res.reshuffle_count} 次"
            )
            print(f"四区终局数量: {res.zone_counts}")
            print(f"P道具触发统计: {res.p_item_activations}")
            print(f"メモリー触发记录 ({len(res.memory_triggers)}): {res.memory_triggers}")
            print(f"喝掉饮料 ({len(res.drinks_used)}): {res.drinks_used}")
            print(f"出牌序列 ({len(res.cards_played)}): {res.cards_played}")
            print(f"除外区卡牌 ({len(res.exhausted_names)}): {res.exhausted_names}")
            print("逐回合战况:")
            for tl in res.turn_logs:
                print(f"  {tl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
