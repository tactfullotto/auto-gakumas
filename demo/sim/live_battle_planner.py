"""Live battle planner wrapping Maskable PPO + SO-ISMCTS for H.I.F Round 1 & Round 2."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from demo.core.db import GakumasDB
from demo.core.idol_guides import resolve_card_family_meta
from demo.schemas import LegalAction, RecognizedCard, RecognizedDrink, ScreenObservation
from demo.sim.battle_simulator import (
    BattleConfig,
    CardConfig,
    CardInstance,
    DrinkConfig,
    HeadlessBattleSimulator,
    PItemConfig,
    TurnMultiplierSpec,
    _build_hif_p_items,
    _build_sense_hif_p_items,
    _make_card_config,
    _make_drink_config,
    generate_stage_turn_schedule,
)
from demo.sim.ismcts import ISMCTSAgent, ISMCTSConfig
from demo.sim.maskable_ppo import MaskablePPOConfig, MaskablePPOTrainer, PolicyValueNet
from demo.sim.rl_env import GakumasActionCodec, GakumasBattleEnv, GakumasFeatureEncoder


_KNOWN_STAGE_P_ITEMS: Dict[int, PItemConfig] = {
    it.item_id: it
    for it in (_build_hif_p_items(include_cheering_stick=True) + _build_sense_hif_p_items())
}
_BANNED_DRINK_IDS = frozenset({19})  # #19 初星スペシャル青汁 (banned)


def _is_banned_drink(
    drink_id: Optional[int],
    name: str = "",
    desc: str = "",
    banned_ids: Optional[Sequence[int]] = None,
) -> bool:
    eff_banned = _BANNED_DRINK_IDS if banned_ids is None else {int(x) for x in banned_ids}
    if drink_id is not None and int(drink_id) in eff_banned:
        return True
    if 19 in eff_banned:
        combined = f"{name or ''} {desc or ''}"
        if (
            "青汁" in combined
            or "ランダムな強化済みスキルカード" in combined
            or "addrandomupgradedssrcardtohand" in combined.lower()
        ):
            return True
    return False

_ATTR_NAME_TO_CODE: Dict[str, str] = {
    "ボーカル": "Vo",
    "ポーカル": "Vo",
    "ホーカル": "Vo",
    "オーカル": "Vo",
    "ダンス": "Da",
    "タンス": "Da",
    "ビジュアル": "Vi",
    "ビシュアル": "Vi",
    "ジュアル": "Vi",
}


def _parse_live_turn_multiplier(obs: ScreenObservation) -> Optional[Tuple[str, float]]:
    """Parse live turn attribute and percentage multiplier (e.g. ポーカル4887 -> ('Vo', 48.87))."""
    text = f"{obs.week_activity or ''} {obs.raw_text or ''}"
    m = re.search(
        r"(ボーカル|ポーカル|ホーカル|オーカル|ダンス|タンス|ビジュアル|ビシュアル|ジュアル)[^\d]{0,5}(\d{3,4})\s*%?",
        text,
    )
    if not m:
        return None
    attr = _ATTR_NAME_TO_CODE.get(m.group(1))
    if not attr:
        return None
    mult = float(int(m.group(2))) / 100.0
    if 10.0 <= mult <= 80.0:
        return (attr, round(mult, 2))
    return None


def compute_dynamic_turn_counts(total_turns: int) -> Tuple[int, int, int]:
    """Split arbitrary battle turn count T into (top_attr, mid_attr, low_attr) counts."""
    t = max(1, int(total_turns))
    if t == 12:
        return (6, 3, 3)
    if t == 9:
        return (5, 2, 2)
    top_c = max(1, int(round(t * 0.5)))
    rem = max(0, t - top_c)
    mid_c = (rem + 1) // 2
    low_c = max(0, rem - mid_c)
    return (top_c, mid_c, low_c)


def build_p_items_from_loadout(
    equipped_p_items: Optional[Sequence[Dict[str, Any]]],
    *,
    scenario_id: str = "hif_honsen",
    total_turns: int = 12,
    week: int = 0,
    plan: str = "logic",
) -> List[PItemConfig]:
    """Convert scanned produce library P-items into battle simulator PItemConfig objects."""
    is_r2 = total_turns >= 11 or int(week or 0) >= 8
    plan_l = (plan or "logic").lower()
    if not equipped_p_items:
        if scenario_id in ("hif_honsen", "hif"):
            if plan_l == "logic":
                return _build_hif_p_items(include_cheering_stick=is_r2)
            return [_KNOWN_STAGE_P_ITEMS[429]] if is_r2 else []
        return []

    out: List[PItemConfig] = []
    seen_ids: set[int] = set()
    for raw in equipped_p_items:
        iid = int(raw.get("item_id") or 0)
        cname = str(raw.get("canonical_name") or raw.get("name") or "")
        if iid in _KNOWN_STAGE_P_ITEMS:
            base = _KNOWN_STAGE_P_ITEMS[iid]
            out.append(base)
            seen_ids.add(base.item_id)
            continue

        matched_known = next(
            (it for it in _KNOWN_STAGE_P_ITEMS.values() if it.name == cname or (cname and cname in it.name)),
            None,
        )
        if matched_known is not None:
            if matched_known.item_id not in seen_ids:
                out.append(matched_known)
                seen_ids.add(matched_known.item_id)
            continue

        out.append(
            PItemConfig(
                item_id=iid or (abs(hash(cname)) % 90000 + 10000),
                name=cname or "未知Pアイテム",
                rarity=str(raw.get("rarity") or "SSR"),
                mode=str(raw.get("mode") or "produce"),
                source_type=str(raw.get("source_type") or "support"),
                description=str(raw.get("description") or ""),
            )
        )
        seen_ids.add(out[-1].item_id)

    # In H.I.F 本戦 Round 2 (when 応援棒 has joined after R1 or >=11T), always ensure #429 is modeled
    if scenario_id in ("hif_honsen", "hif") and is_r2 and 429 not in seen_ids:
        out.append(_KNOWN_STAGE_P_ITEMS[429])
    return out


def _build_card_config_with_hooks(
    db: GakumasDB,
    card_id: int,
    *,
    hooks: Optional[Dict[Any, Dict[str, Any]]] = None,
    custs: Optional[Dict[Any, Dict[str, Any]]] = None,
    use_misuzu_legacy: bool = False,
) -> Optional[CardConfig]:
    """Construct a CardConfig with family-resolved memory hooks and customizations."""
    row = db.get_card_by_id(int(card_id))
    if row is None:
        return None
    cid_int = int(card_id)
    base_id = int(row.get("base_id") or cid_int)
    h_meta = resolve_card_family_meta(hooks, cid_int, db=db, row=row)
    if "force_initial_hand" in h_meta:
        force_init: Optional[bool] = bool(h_meta.get("force_initial_hand"))
    elif use_misuzu_legacy and cid_int in (127, 129, 159):
        force_init = True
    else:
        force_init = None

    # Only genuine trouble-exile memories trigger exiles_trouble_on_play (ignore #426/#427 キセキの魔法 condition text).
    if base_id == 426:
        exile_once = False
    elif "exiles_trouble_on_play" in h_meta:
        exile_once = bool(h_meta.get("exiles_trouble_on_play"))
    else:
        exile_once = bool(use_misuzu_legacy and base_id in (128, 296, 298))

    if "free_extra_use_first_play" in h_meta:
        extra_once = bool(h_meta.get("free_extra_use_first_play"))
    elif "extra_use_limit" in h_meta:
        extra_once = int(h_meta.get("extra_use_limit") or 0) > 0
    else:
        extra_once = bool(use_misuzu_legacy and base_id == 136)

    extra_limit = int(h_meta.get("extra_use_limit") or (1 if extra_once else 0))
    if extra_limit > 0:
        extra_once = True
    draw_on_play = max(0, int(h_meta.get("draw_on_play") or 0))

    c_meta = resolve_card_family_meta(custs, cid_int, db=db, row=row)
    cust_levels = c_meta.get("levels") if isinstance(c_meta, dict) else None
    cust_mods = c_meta.get("modifiers") if isinstance(c_meta, dict) else None

    return _make_card_config(
        db,
        cid_int,
        force_initial_hand_override=force_init,
        exiles_trouble_on_play=exile_once,
        free_extra_use_first_play=extra_once,
        memory_extra_use_limit=extra_limit,
        memory_draw_on_play=draw_on_play,
        customizations=cust_levels,
        custom_modifiers=cust_mods,
    )


def build_live_battle_config(
    *,
    db: GakumasDB,
    deck_card_ids: Sequence[int],
    equipped_p_items: Optional[Sequence[Dict[str, Any]]] = None,
    drinks: Optional[Sequence[Any]] = None,
    idol_stats: Optional[Dict[str, int]] = None,
    stamina: int = 33,
    max_stamina: int = 44,
    total_turns: int = 12,
    week: int = 0,
    buffs: Optional[Dict[str, int]] = None,
    p_idol_id: Optional[int] = None,
    idol_name: str = "[ガラクタロード] 秦谷 美鈴",
    plan: str = "logic",
    card_memory_hooks: Optional[Dict[Any, Dict[str, Any]]] = None,
    card_customizations: Optional[Dict[Any, Dict[str, Any]]] = None,
    memory_passives: Optional[Sequence[Dict[str, Any]]] = None,
    stage_name: str = "H.I.F 本戦",
    scenario_id: str = "hif_honsen",
    initial_genki: int = 0,
    live_top_multiplier: Optional[float] = None,
) -> BattleConfig:
    """Construct a BattleConfig aligned to current produce deck, P-items, stats, and live turn count."""
    raw_stats = dict(idol_stats or {"vocal": 2400, "dance": 1400, "visual": 1800})
    stats = {
        k: min(3200, max(0, int(v or 0)))
        for k, v in raw_stats.items()
    }
    for req_k, dflt in (("vocal", 2400), ("dance", 1400), ("visual", 1800)):
        if req_k not in stats or stats[req_k] <= 0:
            stats[req_k] = dflt
    hooks = card_memory_hooks or {}
    custs = card_customizations or {}

    use_misuzu_legacy = p_idol_id in (None, 163) and str(plan or "").lower() == "logic"
    deck_cfgs: List[CardConfig] = []
    for cid in deck_card_ids:
        cfg_item = _build_card_config_with_hooks(
            db,
            int(cid),
            hooks=hooks,
            custs=custs,
            use_misuzu_legacy=use_misuzu_legacy,
        )
        if cfg_item is not None:
            deck_cfgs.append(cfg_item)

    plan_l = (plan or "logic").lower()
    if not deck_cfgs:
        # Fallback starter deck if scan wasn't available
        fallback_ids = (812, 814, 159, 105, 113, 75, 23) if plan_l == "sense" else (920, 922, 159, 129, 127, 137, 23)
        for fallback_cid in fallback_ids:
            if db.get_card_by_id(fallback_cid) is not None:
                deck_cfgs.append(_make_card_config(db, fallback_cid))

    drink_cfgs: List[DrinkConfig] = []
    for d in drinks or ():
        did = int(getattr(d, "drink_id", d) or 0)
        if did > 0 and db.get_drink_by_id(did) is not None:
            drink_cfgs.append(_make_drink_config(db, did))

    ranked = sorted(
        (
            ("Vo", int(stats.get("vocal", 180) or 180)),
            ("Da", int(stats.get("dance", 160) or 160)),
            ("Vi", int(stats.get("visual", 150) or 150)),
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    top_m = float(live_top_multiplier) if (live_top_multiplier is not None and 25.0 <= float(live_top_multiplier) <= 80.0) else 40.0
    if ranked[0][1] > 500:
        r1 = max(0.50, min(0.92, ranked[1][1] / max(1, ranked[0][1])))
        r2 = max(0.38, min(0.75, ranked[2][1] / max(1, ranked[0][1])))
    else:
        r1, r2 = 0.875, 0.50
    sched_attr_mults = (
        (ranked[0][0], round(top_m, 2)),
        (ranked[1][0], round(top_m * r1, 2)),
        (ranked[2][0], round(top_m * r2, 2)),
    )
    t_total = max(1, int(total_turns))
    sched_counts = compute_dynamic_turn_counts(t_total)
    schedule = generate_stage_turn_schedule(
        random.Random(0),
        attr_multipliers=sched_attr_mults,
        counts=sched_counts,
    )

    live_buffs = buffs or {}
    init_gi = max(0, int(live_buffs.get("goodImpressionTurns", 0) or 0))
    init_mot = max(0, int(live_buffs.get("motivation", 0) or 0))
    init_gc = max(0, int(live_buffs.get("goodConditionTurns", 0) or 0))
    init_pc = max(0, int(live_buffs.get("perfectConditionTurns", 0) or 0))
    init_conc = max(0, int(live_buffs.get("concentration", 0) or 0))
    init_fpc = max(0, int(live_buffs.get("fullPowerCharge", 0) or 0))
    eff_init_genki = max(0, int(initial_genki))

    if memory_passives and not live_buffs:
        for mp in memory_passives:
            if not isinstance(mp, dict):
                continue
            key = str(mp.get("key") or "")
            try:
                val = max(0, int(mp.get("value") or 0))
            except (TypeError, ValueError):
                val = 0
            if val <= 0:
                continue
            if key == "start_good_impression":
                init_gi += val
            elif key == "start_motivation":
                init_mot += val
            elif key == "start_genki":
                eff_init_genki += val
            elif key == "start_good_condition":
                init_gc += val
            elif key == "start_concentration":
                init_conc += val
            elif key == "start_full_power_charge":
                init_fpc += val

    if scenario_id in ("hif_honsen", "hif") and any(
        int(it.get("item_id") or 0) == 436
        or "ポーチ" in str(it.get("canonical_name") or it.get("name") or "")
        or "花ロボ" in str(it.get("canonical_name") or it.get("name") or "")
        for it in (equipped_p_items or ())
    ):
        if plan_l == "sense":
            if not live_buffs:
                init_gc = max(init_gc, 6)
                init_conc = max(init_conc, 6)
            else:
                raw_gc, raw_conc = init_gc, init_conc
                init_gc = max(init_gc, 3)
                init_conc = max(init_conc, 3)
                if raw_conc >= 6 and raw_gc < 3:
                    init_gc = 6
                if raw_gc >= 6 and raw_conc < 3:
                    init_conc = 6
        elif plan_l == "logic":
            if not live_buffs:
                init_gi = max(init_gi, 6)
                init_mot = max(init_mot, 6)
            else:
                raw_gi, raw_mot = init_gi, init_mot
                init_gi = max(init_gi, 3)
                init_mot = max(init_mot, 3)
                if raw_mot >= 6 and raw_gi < 3:
                    init_gi = 6
                if raw_gi >= 6 and raw_mot < 3:
                    init_mot = 6

    eff_idol_name = idol_name
    if plan_l == "sense" and eff_idol_name == "[ガラクタロード] 秦谷 美鈴":
        eff_idol_name = "[ガラクタロード] 姫崎 莉波"

    return BattleConfig(
        scenario_id=scenario_id,
        stage_name=stage_name,
        p_idol_id=int(p_idol_id) if p_idol_id is not None else (139 if plan_l == "sense" else 163),
        idol_name=eff_idol_name or ("姫崎 莉波" if plan_l == "sense" else "秦谷 美鈴"),
        plan=plan_l,
        idol_stats=stats,
        initial_stamina=max(1, int(stamina)),
        max_stamina=max(1, int(max_stamina)),
        initial_genki=eff_init_genki,
        initial_good_impression=init_gi,
        initial_motivation=init_mot,
        initial_good_condition=init_gc,
        initial_perfect_condition=init_pc,
        initial_concentration=init_conc,
        initial_full_power_charge=init_fpc,
        total_turns=t_total,
        turn_schedule=schedule,
        randomize_turn_schedule=(t_total >= 4),
        turn_schedule_attr_multipliers=sched_attr_mults,
        turn_schedule_counts=sched_counts,
        deck=deck_cfgs,
        drinks=drink_cfgs,
        p_items=build_p_items_from_loadout(
            equipped_p_items,
            scenario_id=scenario_id,
            total_turns=t_total,
            week=week,
            plan=plan_l,
        ),
    )


@dataclass
class LiveBattlePlannerConfig:
    """Hyperparameters for online per-stage PPO training + SO-ISMCTS action selection."""

    ppo_hidden_dim: int = 128
    ppo_episodes: int = 2000
    ppo_episodes_per_update: int = 32
    ismcts_simulations: int = 64
    seed: int = 2026
    finisher_hold_min_turns_remaining: int = 3
    prefer_double_effect_before_finisher: bool = True


class LiveBattlePlanner:
    """Maintains battle-level Maskable PPO + SO-ISMCTS policies synced to live ScreenObservation."""

    def __init__(
        self,
        db: Optional[GakumasDB] = None,
        cfg: Optional[LiveBattlePlannerConfig] = None,
    ):
        self._owns_db = db is None
        self.db = db or GakumasDB("gakumas_library.db")
        self.cfg = cfg or LiveBattlePlannerConfig()
        self._rng = random.Random(self.cfg.seed)

        self._active_signature: Optional[Tuple[Any, ...]] = None
        self._battle_initial_turns: Optional[int] = None
        self._last_turns_remaining: Optional[int] = None
        self._last_week: Optional[int] = None
        self._last_trusted_obs_score: int = 0

        self._battle_cfg: Optional[BattleConfig] = None
        self._codec: Optional[GakumasActionCodec] = None
        self._encoder: Optional[GakumasFeatureEncoder] = None
        self._net: Optional[PolicyValueNet] = None
        self._ismcts: Optional[ISMCTSAgent] = None
        self._sim: Optional[HeadlessBattleSimulator] = None
        self._card_memory_hooks: Dict[Any, Dict[str, Any]] = {}
        self._card_customizations: Dict[Any, Dict[str, Any]] = {}
        self._trained_policy_cache: Dict[
            Tuple[Any, ...],
            Tuple[GakumasActionCodec, GakumasFeatureEncoder, PolicyValueNet, Dict[str, Any]],
        ] = {}
        self.banned_drink_ids: set[int] = set(_BANNED_DRINK_IDS)

    def set_banned_drink_ids(self, drink_ids: Optional[Sequence[int]]) -> None:
        self.banned_drink_ids = {int(x) for x in (drink_ids or ())}

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def reset_battle(self) -> None:
        """Clear cached per-stage battle simulator/model when exiting battle."""
        self._active_signature = None
        self._battle_initial_turns = None
        self._last_turns_remaining = None
        self._last_week = None
        self._last_trusted_obs_score = 0
        self._battle_cfg = None
        self._codec = None
        self._encoder = None
        self._net = None
        self._ismcts = None
        self._sim = None
        self._card_memory_hooks = {}
        self._card_customizations = {}

    def _is_new_battle(self, obs: ScreenObservation, tracker: Any) -> bool:
        if self._sim is None or self._battle_initial_turns is None:
            return True
        if self._last_week is not None and obs.week != self._last_week:
            return True
        # Distinguish genuine new round/battle (large reset jump) from +1 extra-turn card (#171)
        if self._last_turns_remaining is not None and obs.turns_remaining > self._last_turns_remaining + 1:
            return True
        return False

    def _train_for_battle(self, obs: ScreenObservation, tracker: Any) -> None:
        deck_ids = list(getattr(tracker, "produce_deck", []) or [])
        if not deck_ids and obs.cards:
            deck_ids = [int(c.card_id) for c in obs.cards]
        else:
            seen_ids = set(deck_ids)
            for c in obs.cards or ():
                cid_i = int(c.card_id)
                if cid_i > 0 and cid_i not in seen_ids:
                    deck_ids.append(cid_i)
                    seen_ids.add(cid_i)
        total_turns = max(1, int(obs.turns_remaining or obs.total_weeks or 10))
        self._battle_initial_turns = total_turns
        self._last_week = int(obs.week)
        self._last_turns_remaining = total_turns
        self._last_trusted_obs_score = 0

        p_idol_id = getattr(tracker, "p_idol_id", None)
        p_row = self.db.get_p_idol_by_id(p_idol_id) if p_idol_id else None
        idol_name = f"[{p_row['title']}] {p_row['idol_name']}" if p_row else "秦谷 美鈴"

        obs_stats = obs.idol_stats or {}
        tr_stats = getattr(tracker, "idol_stats", None) or {}
        if max((int(v or 0) for v in obs_stats.values()), default=0) > 300:
            raw_s = obs_stats
        elif max((int(v or 0) for v in tr_stats.values()), default=0) > 300:
            raw_s = tr_stats
        elif obs.scenario_id in ("hif_honsen", "hif"):
            raw_s = {"vocal": 2600, "dance": 1400, "visual": 1900}
        else:
            raw_s = obs_stats
        clamped_stats = {
            k: min(3200, max(0, int(v or 0)))
            for k, v in raw_s.items()
        } or None

        top_attr = max(
            (
                ("Vo", (clamped_stats or {}).get("vocal", 2400)),
                ("Da", (clamped_stats or {}).get("dance", 1400)),
                ("Vi", (clamped_stats or {}).get("visual", 1800)),
            ),
            key=lambda kv: kv[1],
        )[0]
        live_mult_pair = _parse_live_turn_multiplier(obs)
        live_top_mult = float(live_mult_pair[1]) if (live_mult_pair and live_mult_pair[0] == top_attr) else None

        train_stamina = int(obs.stamina or 0)
        if train_stamina <= 0 and any(getattr(c, "visual_playable", None) is True for c in (obs.cards or ())):
            train_stamina = max(1, int(getattr(tracker, "stamina", 0) or obs.max_stamina or 44))

        safe_obs_drinks = [
            d for d in (obs.drinks or ())
            if not _is_banned_drink(
                getattr(d, "drink_id", None),
                getattr(d, "name", ""),
                getattr(d, "dsl_summary", ""),
                banned_ids=self.banned_drink_ids,
            )
        ]
        card_hooks = getattr(tracker, "card_memory_hooks", None)
        card_custs = getattr(tracker, "card_customizations", None)
        self._card_memory_hooks = {k: dict(v) for k, v in (card_hooks or {}).items() if isinstance(v, dict)}
        self._card_customizations = {k: dict(v) for k, v in (card_custs or {}).items() if isinstance(v, dict)}
        mem_passives = getattr(tracker, "memory_passives", None)
        self._battle_cfg = build_live_battle_config(
            db=self.db,
            deck_card_ids=deck_ids,
            equipped_p_items=getattr(tracker, "equipped_p_items", None),
            drinks=safe_obs_drinks,
            idol_stats=clamped_stats,
            stamina=train_stamina,
            max_stamina=obs.max_stamina,
            total_turns=total_turns,
            week=int(obs.week or 0),
            buffs=obs.buffs,
            p_idol_id=p_idol_id,
            idol_name=idol_name,
            plan=obs.plan,
            card_memory_hooks=card_hooks,
            card_customizations=card_custs,
            memory_passives=mem_passives,
            stage_name=obs.week_activity or f"Week {obs.week}",
            scenario_id=obs.scenario_id,
            initial_genki=obs.genki,
            live_top_multiplier=live_top_mult,
        )

        finisher_window = (
            max(1, int(self.cfg.finisher_hold_min_turns_remaining) - 1)
            if self.cfg.prefer_double_effect_before_finisher
            else 0
        )
        hooks_sig = tuple(
            sorted((str(k), tuple(sorted(v.items()))) for k, v in (card_hooks or {}).items())
        )
        custs_sig = tuple(
            sorted((str(k), json.dumps(v, sort_keys=True)) for k, v in (card_custs or {}).items())
        )
        cache_key = (
            obs.scenario_id,
            str(obs.plan or "").lower(),
            p_idol_id,
            int(obs.week),
            total_turns,
            tuple(deck_ids),
            hooks_sig,
            custs_sig,
            tuple(it.item_id for it in self._battle_cfg.p_items),
            (
                self._battle_cfg.initial_good_impression,
                self._battle_cfg.initial_motivation,
                self._battle_cfg.initial_good_condition,
                self._battle_cfg.initial_concentration,
            ),
            tuple(d.drink_id for d in self._battle_cfg.drinks),
            tuple(sorted(self._battle_cfg.idol_stats.items())),
            int(self.cfg.ppo_episodes),
            int(self.cfg.ppo_hidden_dim),
            int(self.cfg.finisher_hold_min_turns_remaining),
            bool(self.cfg.prefer_double_effect_before_finisher),
        )
        cached_policy = self._trained_policy_cache.get(cache_key)
        cache_hit = cached_policy is not None
        if cached_policy is not None:
            codec, encoder, net, summary = cached_policy
        else:
            env = GakumasBattleEnv(
                self._battle_cfg,
                db=self.db,
                score_scale=1_000_000.0,
                finisher_play_max_turns_remaining=finisher_window,
            )
            ppo_cfg = MaskablePPOConfig(
                hidden_dim=self.cfg.ppo_hidden_dim,
                total_episodes=self.cfg.ppo_episodes,
                episodes_per_update=min(self.cfg.ppo_episodes_per_update, max(1, self.cfg.ppo_episodes)),
                minibatch_size=256,
                val_interval_updates=max(
                    2, min(10, max(1, self.cfg.ppo_episodes // max(1, self.cfg.ppo_episodes_per_update)))
                ),
                val_episodes=min(16, max(4, self.cfg.ppo_episodes // 8)),
                seed=self.cfg.seed,
            )
            trainer = MaskablePPOTrainer(env, ppo_cfg)
            net, summary = trainer.train()
            codec = env.codec
            encoder = env.encoder
            self._trained_policy_cache[cache_key] = (codec, encoder, net, summary)

        logger.info(
            "Trained battle PPO+ISMCTS policy (W%s, turns=%d, deckN=%d, p_items=%d, stats=%s, eps=%d, bestVal=%.0f, finalVal=%.0f, cacheHit=%s).",
            obs.week,
            total_turns,
            len(self._battle_cfg.deck),
            len(self._battle_cfg.p_items),
            self._battle_cfg.idol_stats,
            int(summary.get("episodes_trained", self.cfg.ppo_episodes)),
            float(summary.get("best_val_mean", 0.0)),
            float(summary.get("final_val_mean", 0.0)),
            cache_hit,
        )

        self._codec = codec
        self._encoder = encoder
        self._net = net
        self._ismcts = ISMCTSAgent(
            codec=self._codec,
            encoder=self._encoder,
            net=self._net,
            cfg=ISMCTSConfig(num_simulations=self.cfg.ismcts_simulations, value_weight=0.35),
        )
        self._sim = HeadlessBattleSimulator(self._battle_cfg, db=self.db, seed=self.cfg.seed, record_logs=False)
        self._active_signature = (obs.scenario_id, obs.week, total_turns, tuple(deck_ids))

    def _find_or_create_card_instance(self, card: RecognizedCard, taken_uids: set[int]) -> CardInstance:
        assert self._sim is not None
        cid = int(card.card_id)
        row = self.db.get_card_by_id(cid)
        base_id = int(row.get("base_id") or cid) if row else cid

        def _has_unused_mem(inst: CardInstance) -> bool:
            assert self._sim is not None
            return self._sim.has_unused_memory_extra_use(inst) or self._sim.has_unused_memory_exile(inst)

        for prefer_mem in (True, False):
            for pile in (self._sim.hand, self._sim.draw_pile, self._sim.discard_pile, self._sim.exhaust_pile):
                for idx, inst in enumerate(pile):
                    if inst.uid not in taken_uids and inst.cfg.card_id == cid:
                        if prefer_mem and not _has_unused_mem(inst):
                            continue
                        pile.pop(idx)
                        return inst

        for prefer_mem in (True, False):
            for pile in (self._sim.hand, self._sim.draw_pile, self._sim.discard_pile, self._sim.exhaust_pile):
                for idx, inst in enumerate(pile):
                    if inst.uid not in taken_uids and inst.cfg.base_id == base_id:
                        if prefer_mem and not _has_unused_mem(inst):
                            continue
                        pile.pop(idx)
                        if row and row.get("upgraded") and not inst.cfg.upgraded:
                            inst.cfg = self._sim._get_card_config_cached(
                                cid,
                                force_initial_hand_override=inst.cfg.force_initial_hand,
                                exiles_trouble_on_play=inst.cfg.exiles_trouble_on_play,
                                free_extra_use_first_play=inst.cfg.free_extra_use_first_play,
                                memory_extra_use_limit=inst.cfg.memory_extra_use_limit,
                                memory_draw_on_play=inst.cfg.memory_draw_on_play,
                                customizations=inst.cfg.customizations,
                                custom_modifiers=inst.cfg.custom_modifiers,
                            )
                        return inst

        # Mid-battle generated card (e.g. #23 眠気) or unseen reward card
        new_uid = self._sim.next_uid
        self._sim.next_uid += 1
        use_legacy = (
            self._battle_cfg is not None
            and self._battle_cfg.p_idol_id in (None, 163)
            and str(self._battle_cfg.plan or "").lower() == "logic"
        )
        cfg_new = _build_card_config_with_hooks(
            self.db,
            cid,
            hooks=self._card_memory_hooks,
            custs=self._card_customizations,
            use_misuzu_legacy=use_legacy,
        ) or _make_card_config(self.db, cid)
        inst = CardInstance(uid=new_uid, cfg=cfg_new)
        self._sim._all_instances[new_uid] = inst
        return inst

    def _sync_sim_to_observation(self, obs: ScreenObservation, legal_actions: Sequence[LegalAction]) -> None:
        assert self._sim is not None and self._battle_initial_turns is not None
        sim = self._sim

        live_rem = max(1, int(obs.turns_remaining or 1))
        if live_rem > sim.turns_remaining:
            extra_t = live_rem - sim.turns_remaining
            top_spec = sim.turn_schedule[-1] if sim.turn_schedule else TurnMultiplierSpec("Vo", 40.0)
            sim.turn_schedule.extend(TurnMultiplierSpec(top_spec.attribute, top_spec.multiplier) for _ in range(extra_t))
        if live_rem > self._battle_initial_turns:
            self._battle_initial_turns = live_rem

        actual_turn = max(1, self._battle_initial_turns - live_rem + 1)
        is_new_turn = actual_turn != sim.turn
        if is_new_turn:
            if actual_turn > sim.turn:
                for _ in range(actual_turn - sim.turn):
                    sim.timed_gi_effect_buffs = sim._step_timed_buffs(sim.timed_gi_effect_buffs)
                    sim.timed_gi_gain_buffs = sim._step_timed_buffs(sim.timed_gi_gain_buffs)
                    sim.timed_score_buffs = sim._step_timed_buffs(sim.timed_score_buffs)
                    sim.timed_extra_gi_eot = sim._step_timed_buffs(sim.timed_extra_gi_eot)
                    sim.timed_conc_effect_buffs = sim._step_timed_buffs(sim.timed_conc_effect_buffs)
                    sim.timed_gc_gain_buffs = sim._step_timed_buffs(sim.timed_gc_gain_buffs)
                    sim.timed_gc_addition_buffs = sim._step_timed_buffs(sim.timed_gc_addition_buffs)
                    sim.timed_conc_gain_buffs = sim._step_timed_buffs(sim.timed_conc_gain_buffs)
            sim.turn_cards_used = 0
            sim.fresh_turn_stat_buffs.clear()
        sim.in_unfresh_phase = False
        sim.turn = actual_turn
        sim.turns_elapsed = max(0, actual_turn - 1)
        sim.turns_remaining = live_rem
        sim.stamina = max(0, int(obs.stamina))
        sim.max_stamina = max(1, int(obs.max_stamina))
        sim.genki = max(0, int(obs.genki))
        obs_sc = max(0, int(obs.score or 0))
        if obs_sc > 0:
            max_plausible = max(
                sim.score * 3 + 120_000,
                self._last_trusted_obs_score * 3 + 120_000,
                250_000 if sim.turn <= 4 else 450_000,
            )
            if obs_sc <= max_plausible:
                self._last_trusted_obs_score = max(self._last_trusted_obs_score, obs_sc)
                sim.score = max(sim.score, obs_sc)
            else:
                logger.warning(
                    "Ignoring implausible battle score OCR spike %d (sim=%d, last_trusted=%d, T%d).",
                    obs_sc,
                    sim.score,
                    self._last_trusted_obs_score,
                    sim.turn,
                )
        if is_new_turn:
            sim.card_uses_remaining = max(1, int(obs.card_uses_remaining or 1))
        else:
            sim.card_uses_remaining = max(1, sim.card_uses_remaining, int(obs.card_uses_remaining or 1))
        live_mult_pair = _parse_live_turn_multiplier(obs)
        if live_mult_pair is not None and 0 <= (sim.turn - 1) < len(sim.turn_schedule):
            sim.turn_schedule[sim.turn - 1] = TurnMultiplierSpec(live_mult_pair[0], live_mult_pair[1])
        self._last_turns_remaining = live_rem

        has_grey_mot_card = False
        has_grey_conc_card = False
        has_grey_gc_card = False
        for c in (obs.cards or ()):
            if getattr(c, "visual_playable", None) is False and int(getattr(c, "card_id", 0) or 0) > 0:
                crow = self.db.get_card_by_id(int(c.card_id))
                if crow:
                    c_cost_s = str(crow.get("dsl_cost") or "")
                    c_cond_s = str(crow.get("dsl_conditions") or "")
                    if "motivation-=" in c_cost_s:
                        has_grey_mot_card = True
                    if "concentration-=" in c_cost_s:
                        has_grey_conc_card = True
                    if "goodConditionTurns-=" in c_cost_s or "goodConditionTurns>=" in c_cond_s:
                        has_grey_gc_card = True

        for k in (
            "goodImpressionTurns",
            "motivation",
            "goodConditionTurns",
            "perfectConditionTurns",
            "concentration",
            "halfCostTurns",
            "doubleCostTurns",
            "nullifyDebuff",
        ):
            if k in obs.buffs and obs.buffs[k] is not None:
                obs_val = max(0, int(obs.buffs[k]))
                prev_val = int(sim.buffs.get(k, 0) or 0)
                if k == "goodImpressionTurns" and prev_val >= 3 and obs_val < prev_val - 2:
                    # GI decays by at most 1 per turn; guard against 12 -> 1 or 109 -> 1 OCR truncation
                    continue
                if k == "motivation" and (not has_grey_mot_card) and prev_val >= 4 and obs_val < prev_val - 4:
                    # Single-digit OCR truncation (e.g. 14 -> 1) should not overwrite higher tracked buff value
                    continue
                if k == "concentration" and (not has_grey_conc_card) and prev_val >= 4 and obs_val < prev_val - 4:
                    continue
                if k == "goodConditionTurns" and (not has_grey_gc_card) and prev_val >= 3 and obs_val < prev_val - 2:
                    continue
                sim.buffs[k] = obs_val

        # Strictly conserve existing CardInstances while matching live hand to obs.cards
        if obs.cards:
            taken_uids: set[int] = set()
            new_hand: List[CardInstance] = []
            for card in obs.cards:
                inst = self._find_or_create_card_instance(card, taken_uids)
                taken_uids.add(inst.uid)
                new_hand.append(inst)
            leftovers = [c for c in sim.hand if c.uid not in taken_uids]
            if leftovers:
                if actual_turn == 1 and sim.turns_elapsed == 0:
                    sim.draw_pile.extend(leftovers)
                else:
                    sim.discard_pile.extend(leftovers)
            sim.hand = new_hand

            # Live UI playability reconciliation: if HUD marks a card playable, guarantee sim buff prerequisites match
            for card, inst in zip(obs.cards, sim.hand):
                if getattr(card, "playable", True) and not sim.can_play_card(inst):
                    cost_var, eff_amt = sim._effective_cost(inst.cfg)
                    vis_play = getattr(card, "visual_playable", None)
                    if cost_var not in ("cost", "stamina") and int(sim.buffs.get(cost_var, 0) or 0) < eff_amt:
                        if cost_var in ("motivation", "concentration", "goodConditionTurns") and vis_play is not True:
                            continue
                        sim.buffs[cost_var] = int(eff_amt)
                    elif (
                        vis_play is True
                        and cost_var in ("cost", "stamina")
                        and int(obs.stamina or 0) <= 0
                        and (sim.stamina + (sim.genki if cost_var == "cost" else 0)) < eff_amt
                    ):
                        sim.stamina = max(sim.stamina, int(eff_amt))
                    row = self.db.get_card_by_id(inst.cfg.card_id)
                    if row:
                        cond_txt = (
                            f"{row.get('dsl_conditions', '')} "
                            f"{row.get('dsl_actions', '')} "
                            f"{row.get('dsl_effects', '')}"
                        )
                        for bk, req_s in re.findall(
                            r"(goodImpressionTurns|motivation|goodConditionTurns|concentration)\s*>=\s*(\d+)",
                            cond_txt,
                        ):
                            if bk in ("goodConditionTurns", "concentration") and vis_play is not True:
                                continue
                            if int(sim.buffs.get(bk, 0) or 0) < int(req_s):
                                sim.buffs[bk] = int(req_s)
        elif not any(a.action_type == "play_card" for a in legal_actions):
            if sim.hand:
                sim.discard_pile.extend(sim.hand)
                sim.hand = []

        # Sync available drinks from live HUD
        live_drinks: List[DrinkConfig] = []
        for d in obs.drinks or ():
            if not getattr(d, "playable", True):
                continue
            if _is_banned_drink(
                getattr(d, "drink_id", None),
                getattr(d, "name", ""),
                getattr(d, "dsl_summary", ""),
                banned_ids=self.banned_drink_ids,
            ):
                continue
            did = int(d.drink_id)
            if did > 0 and self.db.get_drink_by_id(did) is not None:
                live_drinks.append(_make_drink_config(self.db, did))
        sim.available_drinks = live_drinks

    def _map_legal_actions_to_canonical(
        self, legal_actions: Sequence[LegalAction]
    ) -> Dict[int, LegalAction]:
        assert self._sim is not None and self._codec is not None
        aid_to_action: Dict[int, LegalAction] = {}
        for act in legal_actions:
            if act.action_type == "play_card":
                cid = int((act.payload or {}).get("card_id") or 0)
                matched_inst = next((inst for inst in self._sim.hand if inst.cfg.card_id == cid), None)
                if matched_inst is None:
                    matched_inst = next((inst for inst in self._sim.hand if inst.cfg.base_id == cid), None)
                if matched_inst is not None:
                    aid = self._codec.card_slot_index(matched_inst.cfg)
                    if aid >= 0:
                        aid_to_action.setdefault(aid, act)
                elif cid in self._codec.card_id_to_fallback_action:
                    aid = self._codec.card_id_to_fallback_action[cid]
                    aid_to_action.setdefault(aid, act)
            elif act.action_type == "use_drink":
                did = int((act.payload or {}).get("drink_id") or 0)
                if _is_banned_drink(
                    did,
                    str((act.payload or {}).get("name") or ""),
                    str(act.description or ""),
                    banned_ids=self.banned_drink_ids,
                ):
                    continue
                aid = self._codec.drink_id_to_action.get(did)
                if aid is not None:
                    aid_to_action.setdefault(aid, act)
            elif act.action_type == "end_turn":
                aid_to_action.setdefault(self._codec.end_turn_action_id, act)
        return aid_to_action

    def _is_non_l_active_card_id(self, card_id: int) -> bool:
        """Return True if `card_id` resolves to a non-L-rarity Active skill card."""
        if int(card_id or 0) <= 0:
            return False
        crow = self.db.get_card_by_id(int(card_id))
        return bool(
            crow
            and str(crow.get("type") or "").lower() == "active"
            and str(crow.get("rarity") or "").upper() != "L"
        )

    def _is_late_scaling_finisher_card(self, card_id: int) -> bool:
        """Return True for limit=1 finishers whose score scales with total cardsUsed (e.g. #813/#814)."""
        if int(card_id or 0) <= 0:
            return False
        crow = self.db.get_card_by_id(int(card_id))
        if not crow:
            return False
        base_id = int(crow.get("base_id") or card_id)
        return crow.get("limit_count") == 1 and (
            base_id == 813 or "cardsUsed" in str(crow.get("dsl_actions") or "")
        )

    def _is_double_effect_enabler_card(self, card_id: int) -> bool:
        """Return True for cards that grant doubleCardEffectCards+= (e.g. #162/#163)."""
        if int(card_id or 0) <= 0:
            return False
        crow = self.db.get_card_by_id(int(card_id))
        if not crow:
            return False
        base_id = int(crow.get("base_id") or card_id)
        return base_id == 162 or "doubleCardEffectCards+=" in str(crow.get("dsl_actions") or "")

    def _is_active_retrigger_drink(self, drink_id: int) -> bool:
        """Return True if drink grants a same-turn Active-only card effect retrigger (e.g. #25 特製ハツボシエキス)."""
        if int(drink_id or 0) == 25:
            return True
        if int(drink_id or 0) > 0:
            drow = self.db.get_drink_by_id(int(drink_id))
            if drow and "doublecardeffectcards" in str(drow.get("dsl_actions") or "").lower():
                return True
        return False

    def _is_timed_score_buff_card_id(self, card_id: int) -> bool:
        """Return True for cards granting timed `setScoreBuff` burst (e.g. #775/#776 話題沸騰)."""
        if int(card_id or 0) <= 0:
            return False
        crow = self.db.get_card_by_id(int(card_id))
        if not crow:
            return False
        base_id = int(crow.get("base_id") or card_id)
        if base_id == 775:
            return True
        return crow.get("limit_count") == 1 and bool(
            re.search(r"setscorebuff\([^,]+,\s*(\d+)\)", str(crow.get("dsl_actions") or "").lower())
        )

    def _score_buff_drink_info(self, drink_id: int) -> Tuple[bool, int, float]:
        """Return `(is_score_buff, duration_turns, amount)` for drinks such as #12 and #14."""
        did = int(drink_id or 0)
        if did <= 0:
            return False, 0, 0.0
        drow = self.db.get_drink_by_id(did)
        dsl_l = str(drow.get("dsl_actions") or "").lower() if drow else ""
        m = re.search(r"setscorebuff\(\s*([0-9.]+)\s*(?:,\s*(\d+))?\s*\)", dsl_l)
        if m:
            amt = float(m.group(1))
            dur = int(m.group(2)) if m.group(2) else 999
            return True, dur, amt
        if did == 12:
            return True, 3, 0.3
        if did == 14:
            return True, 5, 0.1
        return False, 0, 0.0

    def _is_tutor_drink(self, drink_id: int) -> bool:
        """Return True for drinks that move a chosen card from draw/discard to hand (e.g. #22 初星黒酢)."""
        did = int(drink_id or 0)
        if did == 22:
            return True
        if did > 0:
            drow = self.db.get_drink_by_id(did)
            if drow and "moveselectedtohand" in str(drow.get("dsl_actions") or "").lower():
                return True
        return False

    def decide_battle_action(
        self,
        obs: ScreenObservation,
        legal_actions: Sequence[LegalAction],
        tracker: Any = None,
    ) -> Tuple[int, str, Dict[str, Any]]:
        """Run PPO+SO-ISMCTS for the current battle frame and return (choice_id, reason, debug)."""
        if not legal_actions:
            return 0, "No legal actions available", {}

        if self._is_new_battle(obs, tracker):
            self._train_for_battle(obs, tracker)
        elif tracker is not None:
            tr_hooks = getattr(tracker, "card_memory_hooks", None)
            tr_custs = getattr(tracker, "card_customizations", None)
            if tr_hooks:
                self._card_memory_hooks = {k: dict(v) for k, v in tr_hooks.items() if isinstance(v, dict)}
            if tr_custs:
                self._card_customizations = {k: dict(v) for k, v in tr_custs.items() if isinstance(v, dict)}

        assert self._sim is not None and self._codec is not None
        assert self._encoder is not None and self._net is not None and self._ismcts is not None

        self._sync_sim_to_observation(obs, legal_actions)
        aid_to_ui = self._map_legal_actions_to_canonical(legal_actions)
        mapped_choice_ids = {act.choice_id for act in aid_to_ui.values()}
        unmapped_play_acts = [
            act for act in legal_actions
            if act.action_type == "play_card" and act.choice_id not in mapped_choice_ids
        ]

        def _match_unmapped_hand(u_act: LegalAction) -> Optional[Tuple[int, CardInstance]]:
            cid = int((u_act.payload or {}).get("card_id") or 0)
            if cid <= 0:
                return None
            h_idx = next((i for i, inst in enumerate(self._sim.hand) if inst.cfg.card_id == cid), None)
            if h_idx is None:
                h_idx = next((i for i, inst in enumerate(self._sim.hand) if inst.cfg.base_id == cid), None)
            if h_idx is None:
                return None
            inst = self._sim.hand[h_idx]
            if not self._sim.can_play_card(inst):
                return None
            return h_idx, inst

        for u_act in unmapped_play_acts:
            matched_u = _match_unmapped_hand(u_act)
            if matched_u is None:
                continue
            _u_idx, u_inst = matched_u
            u_cid = int((u_act.payload or {}).get("card_id") or 0)
            dsl_u = str(u_inst.cfg.dsl_actions or "").lower()
            if u_cid in (170, 171) or ("turnsremaining+=" in dsl_u and self._sim.is_free_action_card(u_inst)):
                reason_u = (
                    f"PPO+ISMCTS(T{self._sim.turn}/rem={self._sim.turns_remaining}, "
                    f"unmapped extra-turn priority -> {u_act.description})"
                )
                return u_act.choice_id, reason_u, {"best_q_score": float(self._sim.score), "unmapped_override": True}

        live_rem_turns = int(obs.turns_remaining or 1)
        has_unexhausted_813 = self._sim.has_unexhausted_card_family(813) or any(
            self._is_late_scaling_finisher_card(int((a.payload or {}).get("card_id") or 0))
            for a in legal_actions
            if a.action_type == "play_card"
        )
        fin_window = (
            max(1, int(self.cfg.finisher_hold_min_turns_remaining) - 1)
            if self.cfg.prefer_double_effect_before_finisher
            else 0
        )
        hold_for_final_turns = (
            self.cfg.prefer_double_effect_before_finisher
            and live_rem_turns >= self.cfg.finisher_hold_min_turns_remaining
            and self._sim.double_card_effect_cards == 0
        )

        has_playable_non_l_active = any(
            act.action_type == "play_card"
            and self._is_non_l_active_card_id(int((act.payload or {}).get("card_id") or 0))
            for act in legal_actions
        )

        live_hand_size = len(obs.cards)
        has_playable_non_fin = any(
            a.action_type == "play_card"
            and not self._is_late_scaling_finisher_card(int((a.payload or {}).get("card_id") or 0))
            and int((a.payload or {}).get("card_id") or 0) != 23
            for a in legal_actions
        )
        allowed_root: Optional[set[int]] = set(aid_to_ui.keys()) if aid_to_ui else None
        if aid_to_ui:
            def _allow_root_aid(act: LegalAction) -> bool:
                if act.action_type == "use_drink":
                    did = int((act.payload or {}).get("drink_id") or 0)
                    dr_cfg = next((d for d in self._sim.available_drinks if d.drink_id == did), None)
                    if self._is_tutor_drink(did) and live_hand_size >= self._sim.MAX_HAND_SIZE:
                        return False
                    if dr_cfg is not None and self._sim.should_defer_drink_for_hand_cap(
                        dr_cfg,
                        hand_size=live_hand_size,
                        has_playable_non_finisher=has_playable_non_fin,
                    ):
                        return False
                    if self._is_active_retrigger_drink(did):
                        if not has_playable_non_l_active:
                            return False
                        if dr_cfg is not None and not self._sim.can_use_active_retrigger_drink_now(dr_cfg):
                            return False
                    if fin_window > 0 and self._sim.double_card_effect_cards == 0 and has_unexhausted_813:
                        if self._is_tutor_drink(did) and live_rem_turns >= self.cfg.finisher_hold_min_turns_remaining:
                            return False
                        is_sb, sb_dur, _amt = self._score_buff_drink_info(did)
                        if is_sb and 0 < sb_dur < 999 and live_rem_turns > sb_dur:
                            return False
                elif act.action_type == "play_card" and fin_window > 0 and self._sim.double_card_effect_cards == 0:
                    cid = int((act.payload or {}).get("card_id") or 0)
                    if hold_for_final_turns and (
                        self._is_late_scaling_finisher_card(cid)
                        or (self._is_double_effect_enabler_card(cid) and has_unexhausted_813)
                    ):
                        return False
                    if (
                        has_unexhausted_813
                        and self._is_timed_score_buff_card_id(cid)
                        and live_rem_turns > max(3, self.cfg.finisher_hold_min_turns_remaining)
                    ):
                        return False
                return True

            filtered_aids = {aid for aid, act in aid_to_ui.items() if _allow_root_aid(act)}
            if filtered_aids:
                allowed_root = filtered_aids
            if allowed_root and len(allowed_root) > 1 and self._codec.end_turn_action_id in allowed_root:
                has_root_non_trouble = any(
                    aid != self._codec.end_turn_action_id
                    and aid_to_ui[aid].action_type == "play_card"
                    and int((aid_to_ui[aid].payload or {}).get("card_id") or 0) != 23
                    for aid in allowed_root
                )
                has_root_trouble = any(
                    aid != self._codec.end_turn_action_id
                    and aid_to_ui[aid].action_type == "play_card"
                    and int((aid_to_ui[aid].payload or {}).get("card_id") or 0) == 23
                    for aid in allowed_root
                )
                if self._sim.should_prune_blank_end_turn(
                    has_non_trouble_card=has_root_non_trouble,
                    has_trouble_card=has_root_trouble,
                    stamina=int(obs.stamina or 0),
                    turns_remaining=live_rem_turns,
                ):
                    allowed_root.discard(self._codec.end_turn_action_id)

            locked_fin_2x = False
            if self._sim.double_card_effect_cards > 0 and allowed_root:
                fin_root_aids = {
                    aid
                    for aid in allowed_root
                    if aid_to_ui[aid].action_type == "play_card"
                    and self._is_late_scaling_finisher_card(int((aid_to_ui[aid].payload or {}).get("card_id") or 0))
                }
                if fin_root_aids:
                    fin_pool = {
                        aid
                        for aid in allowed_root
                        if aid in fin_root_aids
                        or (
                            aid_to_ui[aid].action_type == "use_drink"
                            and not self._is_active_retrigger_drink(int((aid_to_ui[aid].payload or {}).get("drink_id") or 0))
                        )
                    }
                    if fin_pool:
                        allowed_root = fin_pool
                        locked_fin_2x = True

            if (not locked_fin_2x) and self._sim.has_armed_active_retrigger(same_turn_only=True) and allowed_root:
                active_root_aids = {
                    aid
                    for aid in allowed_root
                    if aid_to_ui[aid].action_type == "play_card"
                    and self._is_non_l_active_card_id(int((aid_to_ui[aid].payload or {}).get("card_id") or 0))
                }
                if active_root_aids:
                    armed_pool = {
                        aid
                        for aid in allowed_root
                        if aid in active_root_aids
                        or (
                            aid_to_ui[aid].action_type == "use_drink"
                            and not self._is_active_retrigger_drink(int((aid_to_ui[aid].payload or {}).get("drink_id") or 0))
                        )
                    }
                    if armed_pool:
                        allowed_root = armed_pool

        best_aid, mcts_stats = self._ismcts.select_action(self._sim, self._rng, allowed_root_actions=allowed_root)
        chosen_ui = aid_to_ui.get(best_aid)

        if chosen_ui is None and aid_to_ui:
            # Restrict policy-value network distribution directly to UI-legal canonical actions
            enc_obs = self._encoder.encode(self._sim)
            ui_mask = np.zeros(self._codec.num_actions, dtype=bool)
            cand_pool = allowed_root if allowed_root is not None else set(aid_to_ui.keys())
            if self._sim.card_uses_remaining == 1:
                dom_cand_aids = {
                    aid
                    for aid in cand_pool
                    if (
                        aid_to_ui[aid].action_type == "play_card"
                        and (
                            inst := next(
                                (
                                    c for c in self._sim.hand
                                    if self._codec.card_slot_index(c.cfg) == aid
                                ),
                                None,
                            )
                        ) is not None
                        and self._sim.is_safe_dominating_free_card(inst)
                    )
                }
                if dom_cand_aids:
                    pruned_pool = {
                        aid
                        for aid in cand_pool
                        if aid in dom_cand_aids or aid_to_ui[aid].action_type == "use_drink"
                    }
                    if pruned_pool:
                        cand_pool = pruned_pool
            for aid in cand_pool:
                ui_mask[aid] = True
            probs, _ = self._net.predict_single(enc_obs, ui_mask)
            ranked_aids = sorted(cand_pool, key=lambda a: float(probs[a]), reverse=True)
            for cand_aid in ranked_aids:
                cand_act = aid_to_ui[cand_aid]
                if cand_act.action_type != "end_turn" or len(aid_to_ui) == 1:
                    chosen_ui = cand_act
                    best_aid = cand_aid
                    break
            if chosen_ui is None:
                chosen_ui = aid_to_ui[ranked_aids[0]]
                best_aid = ranked_aids[0]

        if chosen_ui is None:
            play_or_drink = [
                a for a in legal_actions
                if a.action_type == "play_card"
                or (
                    a.action_type == "use_drink"
                    and not _is_banned_drink(
                        (a.payload or {}).get("drink_id"),
                        str((a.payload or {}).get("name") or ""),
                        str(a.description or ""),
                        banned_ids=self.banned_drink_ids,
                    )
                )
            ]
            chosen_ui = play_or_drink[0] if play_or_drink else legal_actions[0]

        override_note = ""
        if unmapped_play_acts:
            best_q = float(mcts_stats.get("best_q_score", 0.0))
            best_u_eval: Optional[Tuple[float, LegalAction, CardInstance]] = None
            require_armed_active = (
                self._sim.has_armed_active_retrigger(same_turn_only=True)
                and chosen_ui.action_type == "play_card"
                and self._is_non_l_active_card_id(int((chosen_ui.payload or {}).get("card_id") or 0))
            )
            for u_act in unmapped_play_acts:
                u_cid = int((u_act.payload or {}).get("card_id") or 0)
                if u_cid == 23:
                    continue
                if hold_for_final_turns and (
                    self._is_late_scaling_finisher_card(u_cid)
                    or (self._is_double_effect_enabler_card(u_cid) and has_unexhausted_813)
                ):
                    continue
                if require_armed_active and not self._is_non_l_active_card_id(u_cid):
                    continue
                matched_u = _match_unmapped_hand(u_act)
                if matched_u is None:
                    continue
                u_idx, u_inst = matched_u
                eval_vals: List[float] = []
                for _det in range(6):
                    d_clone = self._sim.clone()
                    d_clone.rng = random.Random(self._rng.randrange(1 << 63))
                    d_clone.rng.shuffle(d_clone.draw_pile)
                    d_clone.step("card", u_idx)
                    leaf_val = self._ismcts._evaluate_leaf(d_clone, d_clone.rng) * self._ismcts.cfg.score_scale
                    eval_vals.append(float(leaf_val))
                if eval_vals:
                    avg_u_q = sum(eval_vals) / len(eval_vals)
                    if best_u_eval is None or avg_u_q > best_u_eval[0]:
                        best_u_eval = (avg_u_q, u_act, u_inst)
            if best_u_eval is not None:
                best_u_q, best_u_act, best_u_inst = best_u_eval
                u_cid_best = int((best_u_act.payload or {}).get("card_id") or 0)
                hold_unmapped_for_late = (
                    chosen_ui.action_type == "end_turn"
                    and int(obs.turns_remaining or 1) >= self.cfg.finisher_hold_min_turns_remaining
                    and (
                        self._is_late_scaling_finisher_card(u_cid_best)
                        or self._is_double_effect_enabler_card(u_cid_best)
                    )
                    and best_u_q <= best_q
                )
                if (
                    not hold_unmapped_for_late
                    and (
                        best_u_q > best_q
                        or (self._sim.is_safe_dominating_free_card(best_u_inst) and best_u_q >= best_q * 0.95)
                        or chosen_ui.action_type == "end_turn"
                    )
                ):
                    chosen_ui = best_u_act
                    mcts_stats["best_q_score"] = max(best_q, best_u_q)
                    override_note += f" [Unmapped fallback simQ={best_u_q:,.0f}]"
        if chosen_ui.action_type == "end_turn" and obs.stamina >= 4:
            non_trouble_cards = [
                a for a in legal_actions
                if a.action_type == "play_card" and int((a.payload or {}).get("card_id") or 0) != 23
            ]
            if self._sim.double_card_effect_cards > 0 and live_rem_turns > 1 and has_unexhausted_813:
                non_trouble_cards = [
                    a for a in non_trouble_cards
                    if self._is_late_scaling_finisher_card(int((a.payload or {}).get("card_id") or 0))
                ]
            elif live_rem_turns >= self.cfg.finisher_hold_min_turns_remaining:
                filtered_non_hold = []
                for a in non_trouble_cards:
                    a_cid = int((a.payload or {}).get("card_id") or 0)
                    if self._is_double_effect_enabler_card(a_cid):
                        continue
                    if self._is_late_scaling_finisher_card(a_cid) and self._sim.double_card_effect_cards <= 0:
                        continue
                    if (
                        has_unexhausted_813
                        and self._is_timed_score_buff_card_id(a_cid)
                        and live_rem_turns > max(3, self.cfg.finisher_hold_min_turns_remaining)
                    ):
                        continue
                    filtered_non_hold.append(a)
                non_trouble_cards = filtered_non_hold
            if non_trouble_cards:
                aid_scores: Dict[int, float] = {}
                try:
                    enc_obs = self._encoder.encode(self._sim)
                    ui_mask = np.zeros(self._codec.num_actions, dtype=bool)
                    for aid in aid_to_ui:
                        ui_mask[aid] = True
                    probs, _ = self._net.predict_single(enc_obs, ui_mask)
                    for aid in aid_to_ui:
                        aid_scores[aid] = float(probs[aid])
                except Exception:
                    probs = None
                top_q_by_name = {str(nm): float(qv) for nm, _cnt, qv in (mcts_stats.get("top_actions") or [])}
                ui_to_aid = {act.choice_id: aid for aid, act in aid_to_ui.items()}

                def _non_skip_rank(act: LegalAction) -> Tuple[int, float, float]:
                    aid = ui_to_aid.get(act.choice_id)
                    aname = self._codec.action_name(aid) if aid is not None else ""
                    q_val = top_q_by_name.get(aname, -1e9)
                    p_val = aid_scores.get(aid, 0.0) if aid is not None else 0.0
                    is_free = 1 if ("不耗行动点" in act.description or "🟢0消耗" in act.description) else 0
                    return (1 + is_free, q_val, p_val)

                has_unmapped = any(act.choice_id not in ui_to_aid for act in non_trouble_cards)
                has_free_or_final = (
                    obs.stamina >= 6
                    or int(obs.turns_remaining or 1) <= 1
                    or any("不耗行动点" in a.description or "🟢0消耗" in a.description for a in non_trouble_cards)
                )
                if has_unmapped or has_free_or_final:
                    chosen_ui = max(non_trouble_cards, key=_non_skip_rank)
                    override_note = " [Anti-SKIP override]"
            if (
                chosen_ui.action_type == "end_turn"
                and self._sim.double_card_effect_cards == 0
                and not self._sim.has_armed_active_retrigger(same_turn_only=True)
                and live_rem_turns >= 2
            ):
                trouble_cards = [
                    a for a in legal_actions
                    if a.action_type == "play_card" and int((a.payload or {}).get("card_id") or 0) == 23
                ]
                if trouble_cards:
                    chosen_ui = trouble_cards[0]
                    override_note = " [Exile #23 眠気 instead of blank SKIP]"

        if self.cfg.prefer_double_effect_before_finisher and chosen_ui is not None:
            chosen_cid = int((chosen_ui.payload or {}).get("card_id") or 0)
            playable_cards = [
                a for a in legal_actions
                if a.action_type == "play_card" and int((a.payload or {}).get("card_id") or 0) != 23
            ]
            top_q_map = {str(nm): float(qv) for nm, _cnt, qv in (mcts_stats.get("top_actions") or [])}
            ui_to_aid_hold = {act.choice_id: aid for aid, act in aid_to_ui.items()}
            end_act = aid_to_ui.get(self._codec.end_turn_action_id) or next(
                (a for a in legal_actions if a.action_type == "end_turn"), None
            )

            if (
                hold_for_final_turns
                and chosen_ui.action_type == "play_card"
                and (
                    self._is_late_scaling_finisher_card(chosen_cid)
                    or (self._is_double_effect_enabler_card(chosen_cid) and has_unexhausted_813)
                )
            ):
                alt_cards = [
                    a for a in playable_cards
                    if not self._is_late_scaling_finisher_card(int((a.payload or {}).get("card_id") or 0))
                    and not (
                        self._is_double_effect_enabler_card(int((a.payload or {}).get("card_id") or 0))
                        and has_unexhausted_813
                    )
                ]
                if alt_cards:
                    best_alt = max(
                        alt_cards,
                        key=lambda a: (
                            1 if ("不耗行动点" in a.description or "🟢0消耗" in a.description) else 0,
                            top_q_map.get(
                                self._codec.action_name(ui_to_aid_hold[a.choice_id]), -1e9
                            )
                            if a.choice_id in ui_to_aid_hold
                            else -1e9,
                        ),
                    )
                    chosen_ui = best_alt
                    override_note += " [Hold #814/#163 for last 2 turns]"
                else:
                    trouble_alt = next(
                        (
                            a for a in legal_actions
                            if a.action_type == "play_card" and int((a.payload or {}).get("card_id") or 0) == 23
                        ),
                        None,
                    )
                    if (
                        trouble_alt is not None
                        and live_rem_turns >= 2
                        and obs.stamina >= 4
                        and self._sim.double_card_effect_cards == 0
                        and not self._sim.has_armed_active_retrigger(same_turn_only=True)
                    ):
                        chosen_ui = trouble_alt
                        override_note += " [Hold #814/#163 -> Exile #23 眠気]"
                    elif end_act is not None:
                        chosen_ui = end_act
                        override_note += " [Hold #814/#163 for last 2 turns]"
            elif (
                chosen_ui.action_type == "play_card"
                and self._is_late_scaling_finisher_card(chosen_cid)
                and self._sim.double_card_effect_cards == 0
            ):
                combo_163 = next(
                    (
                        a for a in playable_cards
                        if self._is_double_effect_enabler_card(int((a.payload or {}).get("card_id") or 0))
                    ),
                    None,
                )
                if combo_163 is not None and int(self._sim.buffs.get("goodConditionTurns", 0) or 0) >= 13:
                    chosen_ui = combo_163
                    override_note += " [Combo #162/#163->#814 2x]"
            elif self._sim.double_card_effect_cards > 0:
                already_finisher = (
                    chosen_ui.action_type == "play_card"
                    and self._is_late_scaling_finisher_card(chosen_cid)
                )
                finisher_act = next(
                    (
                        a for a in playable_cards
                        if self._is_late_scaling_finisher_card(int((a.payload or {}).get("card_id") or 0))
                    ),
                    None,
                )
                chosen_did = int((chosen_ui.payload or {}).get("drink_id") or 0) if chosen_ui.action_type == "use_drink" else 0
                chosen_is_safe_drink = (
                    chosen_ui.action_type == "use_drink"
                    and not self._is_active_retrigger_drink(chosen_did)
                    and not _is_banned_drink(chosen_did, str(chosen_ui.description or ""), banned_ids=self.banned_drink_ids)
                )
                if not already_finisher and not chosen_is_safe_drink:
                    if finisher_act is not None:
                        chosen_ui = finisher_act
                        override_note += " [Armed 2x->#814]"
                    else:
                        tutor_act = next(
                            (
                                a for a in legal_actions
                                if a.action_type == "use_drink"
                                and self._is_tutor_drink(int((a.payload or {}).get("drink_id") or 0))
                                and not _is_banned_drink(
                                    (a.payload or {}).get("drink_id"),
                                    str((a.payload or {}).get("name") or ""),
                                    str(a.description or ""),
                                    banned_ids=self.banned_drink_ids,
                                )
                            ),
                            None,
                        )
                        if (
                            tutor_act is not None
                            and len(obs.cards) < self._sim.MAX_HAND_SIZE
                            and len(self._sim.hand) < self._sim.MAX_HAND_SIZE
                            and any(c.cfg.base_id == 813 for z in (self._sim.draw_pile, self._sim.discard_pile) for c in z)
                        ):
                            chosen_ui = tutor_act
                            override_note += " [Armed 2x->Tutor #814 via #22]"
                        elif (
                            live_rem_turns > 1
                            and len(self._sim.draw_pile) <= 3
                            and any(c.cfg.base_id == 813 for c in self._sim.draw_pile)
                            and chosen_ui.action_type == "play_card"
                            and end_act is not None
                        ):
                            crow_chosen = self.db.get_card_by_id(chosen_cid) if chosen_cid > 0 else None
                            chosen_rarity = str(crow_chosen.get("rarity") or "") if crow_chosen else ""
                            if chosen_rarity != "L":
                                chosen_ui = end_act
                                override_note += " [Carry 2x->rem1 #814]"

            final_chosen_cid = int((chosen_ui.payload or {}).get("card_id") or 0) if chosen_ui.action_type == "play_card" else 0
            if (
                chosen_ui.action_type == "play_card"
                and self._is_late_scaling_finisher_card(final_chosen_cid)
            ):
                sb_drinks = []
                for a in legal_actions:
                    if a.action_type != "use_drink":
                        continue
                    did = int((a.payload or {}).get("drink_id") or 0)
                    if _is_banned_drink(
                        did,
                        str((a.payload or {}).get("name") or ""),
                        str(a.description or ""),
                        banned_ids=self.banned_drink_ids,
                    ):
                        continue
                    is_sb, _dur, amt = self._score_buff_drink_info(did)
                    if is_sb:
                        dr_cfg_sb = next((d for d in self._sim.available_drinks if d.drink_id == did), None)
                        if (
                            dr_cfg_sb is not None
                            and live_rem_turns > 1
                            and self._sim.should_defer_drink_for_hand_cap(
                                dr_cfg_sb,
                                hand_size=live_hand_size,
                                has_playable_non_finisher=has_playable_non_fin,
                            )
                        ):
                            continue
                        sb_drinks.append((amt, did, a))
                if sb_drinks:
                    sb_drinks.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                    _, sb_did, sb_act = sb_drinks[0]
                    chosen_ui = sb_act
                    override_note += f" [Pre-#814 ScoreBuff Drink #{sb_did}]"

        q_best = float(mcts_stats.get("best_q_score", 0.0))
        reason = (
            f"PPO+ISMCTS(T{self._sim.turn}/rem={self._sim.turns_remaining}, "
            f"simQ={q_best:,.0f}, act={chosen_ui.description}){override_note}"
        )
        return chosen_ui.choice_id, reason, mcts_stats

    def notify_action_executed(self, action: Optional[LegalAction]) -> None:
        """Advance internal simulator state to preserve trigger counters and memory limits across live steps."""
        if action is None or self._sim is None or self._sim.turns_remaining <= 0:
            return
        try:
            if action.action_type == "play_card":
                cid = int((action.payload or {}).get("card_id") or 0)
                idx = next((i for i, inst in enumerate(self._sim.hand) if inst.cfg.card_id == cid), None)
                if idx is None:
                    idx = next((i for i, inst in enumerate(self._sim.hand) if inst.cfg.base_id == cid), None)
                if idx is not None and self._sim.can_play_card(self._sim.hand[idx]):
                    self._sim.step("card", idx)
            elif action.action_type == "use_drink":
                did = int((action.payload or {}).get("drink_id") or 0)
                idx = next((i for i, dr in enumerate(self._sim.available_drinks) if dr.drink_id == did), None)
                if idx is not None:
                    self._sim.step("drink", idx)
            elif action.action_type == "end_turn":
                self._sim.step("end_turn", -1)
        except Exception:
            # Keep live runner resilient even if UI execution differed slightly
            pass
