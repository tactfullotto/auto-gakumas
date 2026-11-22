"""Live battle planner wrapping Maskable PPO + SO-ISMCTS for H.I.F Round 1 & Round 2."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from demo.core.db import GakumasDB
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
    _make_card_config,
    _make_drink_config,
    generate_stage_turn_schedule,
)
from demo.sim.ismcts import ISMCTSAgent, ISMCTSConfig
from demo.sim.maskable_ppo import MaskablePPOConfig, MaskablePPOTrainer, PolicyValueNet
from demo.sim.rl_env import GakumasActionCodec, GakumasBattleEnv, GakumasFeatureEncoder


_KNOWN_STAGE_P_ITEMS: Dict[int, PItemConfig] = {
    it.item_id: it for it in _build_hif_p_items(include_cheering_stick=True)
}
_BANNED_DRINK_IDS = frozenset({19})  # #19 初星スペシャル青汁 (banned)


def _is_banned_drink(drink_id: Optional[int], name: str = "", desc: str = "") -> bool:
    if drink_id is not None and int(drink_id) in _BANNED_DRINK_IDS:
        return True
    combined = f"{name or ''} {desc or ''}"
    return (
        "青汁" in combined
        or "ランダムな強化済みスキルカード" in combined
        or "addrandomupgradedssrcardtohand" in combined.lower()
    )

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
) -> List[PItemConfig]:
    """Convert scanned produce library P-items into battle simulator PItemConfig objects."""
    is_r2 = total_turns >= 11 or int(week or 0) >= 8
    if not equipped_p_items:
        if scenario_id in ("hif_honsen", "hif"):
            return _build_hif_p_items(include_cheering_stick=is_r2)
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
    card_memory_hooks: Optional[Dict[int, Dict[str, Any]]] = None,
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

    deck_cfgs: List[CardConfig] = []
    for cid in deck_card_ids:
        row = db.get_card_by_id(int(cid))
        if row is None:
            continue
        cid_int = int(cid)
        base_id = int(row.get("base_id") or cid_int)
        h_meta = hooks.get(cid_int) or hooks.get(str(cid_int)) or hooks.get(base_id) or hooks.get(str(base_id)) or {}
        force_init: Optional[bool] = True if h_meta.get("force_initial_hand") else None
        if force_init is None and cid_int in (127, 129, 159):
            force_init = True

        # Only genuine trouble-exile memories trigger exiles_trouble_on_play (ignore #426/#427 キセキの魔法 condition text).
        # Stage-wide once per base_id is enforced dynamically in HeadlessBattleSimulator across duplicate copies.
        exile_once = (
            base_id != 426
            and (bool(h_meta.get("exiles_trouble_on_play")) or base_id in (128, 296, 298))
        )
        extra_once = bool(h_meta.get("free_extra_use_first_play")) or (base_id == 136)

        deck_cfgs.append(
            _make_card_config(
                db,
                cid_int,
                force_initial_hand_override=force_init,
                exiles_trouble_on_play=exile_once,
                free_extra_use_first_play=extra_once,
            )
        )

    if not deck_cfgs:
        # Fallback starter logic deck if scan wasn't available
        for fallback_cid in (920, 922, 159, 129, 127, 137, 23):
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
    if scenario_id in ("hif_honsen", "hif") and any(
        int(it.get("item_id") or 0) == 436
        or "ポーチ" in str(it.get("canonical_name") or it.get("name") or "")
        or "花ロボ" in str(it.get("canonical_name") or it.get("name") or "")
        for it in (equipped_p_items or ())
    ):
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

    return BattleConfig(
        scenario_id=scenario_id,
        stage_name=stage_name,
        p_idol_id=int(p_idol_id or 163),
        idol_name=idol_name or "秦谷 美鈴",
        plan=(plan or "logic").lower(),
        idol_stats=stats,
        initial_stamina=max(1, int(stamina)),
        max_stamina=max(1, int(max_stamina)),
        initial_genki=max(0, int(initial_genki)),
        initial_good_impression=init_gi,
        initial_motivation=init_mot,
        total_turns=t_total,
        turn_schedule=schedule,
        randomize_turn_schedule=(t_total >= 4),
        turn_schedule_attr_multipliers=sched_attr_mults,
        turn_schedule_counts=sched_counts,
        deck=deck_cfgs,
        drinks=drink_cfgs,
        p_items=build_p_items_from_loadout(equipped_p_items, scenario_id=scenario_id, total_turns=t_total, week=week),
    )


@dataclass
class LiveBattlePlannerConfig:
    """Hyperparameters for online per-stage PPO training + SO-ISMCTS action selection."""

    ppo_hidden_dim: int = 128
    ppo_episodes: int = 2000
    ppo_episodes_per_update: int = 32
    ismcts_simulations: int = 64
    seed: int = 2026


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
        self._trained_policy_cache: Dict[
            Tuple[Any, ...],
            Tuple[GakumasActionCodec, GakumasFeatureEncoder, PolicyValueNet, Dict[str, Any]],
        ] = {}

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
            )
        ]
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
            card_memory_hooks=getattr(tracker, "card_memory_hooks", None),
            stage_name=obs.week_activity or f"Week {obs.week}",
            scenario_id=obs.scenario_id,
            initial_genki=obs.genki,
            live_top_multiplier=live_top_mult,
        )

        cache_key = (
            obs.scenario_id,
            int(obs.week),
            total_turns,
            tuple(deck_ids),
            tuple(d.drink_id for d in self._battle_cfg.drinks),
            tuple(sorted(self._battle_cfg.idol_stats.items())),
            int(self.cfg.ppo_episodes),
            int(self.cfg.ppo_hidden_dim),
        )
        cached_policy = self._trained_policy_cache.get(cache_key)
        cache_hit = cached_policy is not None
        if cached_policy is not None:
            codec, encoder, net, summary = cached_policy
        else:
            env = GakumasBattleEnv(self._battle_cfg, db=self.db, score_scale=1_000_000.0)
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
                            )
                        return inst

        # Mid-battle generated card (e.g. #23 眠気) or unseen reward card
        new_uid = self._sim.next_uid
        self._sim.next_uid += 1
        inst = CardInstance(uid=new_uid, cfg=_make_card_config(self.db, cid))
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
            sim.turn_cards_used = 0
            sim.fresh_turn_stat_buffs.clear()
            sim.in_unfresh_phase = True
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
        for c in (obs.cards or ()):
            if getattr(c, "visual_playable", None) is False and int(getattr(c, "card_id", 0) or 0) > 0:
                crow = self.db.get_card_by_id(int(c.card_id))
                if crow and "motivation-=" in str(crow.get("dsl_cost") or ""):
                    has_grey_mot_card = True
                    break

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
                        if cost_var == "motivation" and vis_play is not True:
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
                        cond_txt = f"{row.get('dsl_actions', '')} {row.get('dsl_effects', '')}"
                        for bk, req_s in re.findall(
                            r"(goodImpressionTurns|motivation|goodConditionTurns|concentration)\s*>=\s*(\d+)",
                            cond_txt,
                        ):
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

    def _is_active_retrigger_drink(self, drink_id: int) -> bool:
        """Return True if drink grants a same-turn Active-only card effect retrigger (e.g. #25 特製ハツボシエキス)."""
        if int(drink_id or 0) == 25:
            return True
        if int(drink_id or 0) > 0:
            drow = self.db.get_drink_by_id(int(drink_id))
            if drow and "doublecardeffectcards" in str(drow.get("dsl_actions") or "").lower():
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

        has_playable_non_l_active = any(
            act.action_type == "play_card"
            and self._is_non_l_active_card_id(int((act.payload or {}).get("card_id") or 0))
            for act in legal_actions
        )

        allowed_root: Optional[set[int]] = set(aid_to_ui.keys()) if aid_to_ui else None
        if aid_to_ui:
            def _allow_root_aid(act: LegalAction) -> bool:
                if act.action_type == "use_drink":
                    did = int((act.payload or {}).get("drink_id") or 0)
                    if self._is_active_retrigger_drink(did):
                        if not has_playable_non_l_active:
                            return False
                        dr_cfg = next((d for d in self._sim.available_drinks if d.drink_id == did), None)
                        if dr_cfg is not None and not self._sim.can_use_active_retrigger_drink_now(dr_cfg):
                            return False
                return True

            filtered_aids = {aid for aid, act in aid_to_ui.items() if _allow_root_aid(act)}
            if filtered_aids:
                allowed_root = filtered_aids

            if self._sim.has_armed_active_retrigger(same_turn_only=True) and allowed_root:
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
                if (
                    best_u_q > best_q
                    or (self._sim.is_safe_dominating_free_card(best_u_inst) and best_u_q >= best_q * 0.95)
                    or chosen_ui.action_type == "end_turn"
                ):
                    chosen_ui = best_u_act
                    mcts_stats["best_q_score"] = max(best_q, best_u_q)
                    override_note += f" [Unmapped fallback simQ={best_u_q:,.0f}]"
        if chosen_ui.action_type == "end_turn" and obs.stamina >= 6:
            non_trouble_cards = [
                a for a in legal_actions
                if a.action_type == "play_card" and int((a.payload or {}).get("card_id") or 0) != 23
            ]
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
                    obs.stamina >= 10
                    or int(obs.turns_remaining or 1) <= 1
                    or any("不耗行动点" in a.description or "🟢0消耗" in a.description for a in non_trouble_cards)
                )
                if has_unmapped or has_free_or_final:
                    chosen_ui = max(non_trouble_cards, key=_non_skip_rank)
                    override_note = " [Anti-SKIP override]"

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
