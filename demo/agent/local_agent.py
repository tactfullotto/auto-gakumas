"""Local zero-LLM Hybrid Decision Agent (Rule-based produce + PPO+ISMCTS battle)."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from demo.core.db import GakumasDB
from demo.core.hif_catalog import ProducePreferences
from demo.core.idol_guides import PIdolGuide, get_p_idol_guide, resolve_p_idol_id
from demo.core.tracker import GameStateTracker
from demo.schemas import DecisionResult, LegalAction, ScreenObservation
from demo.sim.live_battle_planner import LiveBattlePlanner, LiveBattlePlannerConfig

logger = logging.getLogger(__name__)


_EFFECT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "goodImpressionTurns": (
        "goodimpressionturns",
        "goodimpression",
        "好印象",
        "効果増加",
        "毎ターン",
    ),
    "motivation": (
        "motivation",
        "やる気",
        "干劲",
        "元気",
        "元气",
    ),
    "goodConditionTurns": (
        "goodconditionturns",
        "perfectcondition",
        "好調",
        "絶好調",
        "绝好调",
        "好调",
    ),
    "concentration": (
        "concentration",
        "集中",
    ),
    "fullPowerCharge": (
        "fullpowercharge",
        "fullpower",
        "全力",
        "強気",
        "温存",
    ),
}

_SECONDARY_EFFECT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "goodImpressionTurns": ("やる気", "motivation", "元気", "使用数", "cardusesremaining", "draw"),
    "motivation": ("元気", "ブロック", "使用数", "draw"),
    "goodConditionTurns": ("集中", "使用数", "draw"),
    "concentration": ("好調", "絶好調", "使用数", "draw"),
    "fullPowerCharge": ("強気", "温存", "使用数", "draw"),
}


STAT_CAP = 3200
STAT_NEAR_CAP = 3050

_CORE_KEEP_CARD_IDS = frozenset({
    919, 920, 921, 922, 158, 159, 126, 127, 128, 129, 296, 297, 298, 299,
    170, 171, 172, 173, 174, 175, 178, 179, 304, 305, 520, 521, 616, 617,
    728, 777, 778, 136, 137, 710, 711, 98, 99, 130, 131, 138, 139,
    180, 181, 724, 725, 781,
})
_CORE_KEEP_BASE_IDS = frozenset({
    919, 921, 158, 126, 128, 296, 298, 170, 172, 174, 178, 304, 520, 616,
    728, 777, 136, 710, 98, 130, 138, 180, 724, 725, 781,
})
_ELITE_LOGIC_ACTIVE_BASE_IDS = frozenset({
    921, 777, 304, 781,
})
_FORBIDDEN_BASE_IDS_BY_P_IDOL: Dict[int, frozenset] = {
    # [ガラクタロード] 秦谷 美鈴 (logic / GI): #152/#153 届いて！ (clears shield) and #426/#427 キセキの魔法 (HP cost & 眠気 synergy)
    163: frozenset({152, 426}),
}
_FORBIDDEN_NAME_TOKENS_BY_P_IDOL: Dict[int, Tuple[str, ...]] = {
    163: ("届いて", "キセキの魔法"),
}
_BANNED_DRINK_IDS = frozenset({19})  # #19 初星スペシャル青汁 (random SSR generation banned)


class LocalHybridDecisionAgent:
    """
    100% local, zero-LLM decision agent implementing the AgentDecisionMaker interface.
    - Home -> Produce -> Home termination guard
    - Setup (`H.I.F 本戦`, `1.アイドル選択`, `2.開始確認`) without spending Jewels or retiring
    - Produce Schedule: unconditionally prioritizes SP courses (`is_sp` / `SP`), balanced by idol stats
    - Card Rewards & Events: aligns with P-Idol `recommended_effect` (e.g. `goodImpressionTurns`) + PIdolGuide
    - Shop (`相談` / `インターバル`): strictly forbids buying/adding new skill cards
    - Battle Turn: Maskable PPO + Online SO-ISMCTS via `LiveBattlePlanner`
    """

    MIN_STAT_FLOOR: int = 1150

    def __init__(
        self,
        db: Optional[GakumasDB] = None,
        planner_cfg: Optional[LiveBattlePlannerConfig] = None,
        library_out_path: Optional[Union[str, Path]] = None,
        preferences: Optional[Union[ProducePreferences, Dict[str, Any]]] = None,
    ):
        self._owns_db = db is None
        self.db = db or GakumasDB("gakumas_library.db")
        self.is_client_available = True
        self.tracker: Optional[GameStateTracker] = None
        self.library_out_path = Path(library_out_path) if library_out_path else None
        self.battle_planner = LiveBattlePlanner(db=self.db, cfg=planner_cfg)

        self.preferences: ProducePreferences = ProducePreferences()
        self._priority_card_ids: Set[int] = set()
        self._priority_card_base_ids: Set[int] = set()
        self._avoid_card_ids: Set[int] = set()
        self._avoid_card_base_ids: Set[int] = set()
        self._keep_card_ids: Set[int] = set()
        self._keep_card_base_ids: Set[int] = set()
        self._card_score_overrides_exact: Dict[int, int] = {}
        self._card_score_overrides_by_base: Dict[int, int] = {}
        self._preferred_drink_ids: Set[int] = set()
        self._banned_drink_ids: Set[int] = {19}
        self._banned_drink_names: Tuple[str, ...] = ("初星スペシャル青汁", "青汁")

        self._has_entered_produce: bool = False
        self._has_seen_produce_setup: bool = False
        self._produce_event_seen_count: int = 0
        self._reward_preview_count: int = 0
        self._last_reward_sig: Optional[Tuple[Any, ...]] = None
        self._reward_seen_slots: Dict[int, Tuple[int, Optional[int], str]] = {}
        self._reward_slot_attempts: Dict[int, int] = {}
        self._reward_visited_grid_slots: Set[int] = set()
        self._reward_cancel_attempts: int = 0
        self._reward_grid_scores: Dict[int, int] = {}
        self._last_clicked_grid_slot: Optional[int] = None
        self._reward_grid_request_scroll: Optional[str] = None
        self._reward_grid_can_flip_scroll: bool = False
        self._reward_reroll_attempts: int = 0
        self._came_from_change_pickup: bool = False
        self._pending_change_gain_card_id: Optional[int] = None
        self._pending_change_gain_base_id: Optional[int] = None
        self._customize_clicked_tiles: Set[str] = set()
        self._customize_cards_tried: int = 0
        self._customize_done_count: int = 0
        self._shop_card_popup_guard: bool = False
        self._shop_expecting_drink_confirm: bool = False
        self._shop_expecting_priority_card_confirm: Optional[int] = None
        self._shop_expecting_priority_card_is_change: bool = False
        self._shop_attempted_drink_ids: Set[int] = set()
        self._shop_attempted_card_ids: Set[int] = set()
        self._shop_clicked_slots: Set[str] = set()
        self._shop_noop_features: Set[str] = set()
        self._shop_last_clicked_feature: Optional[str] = None
        self._shop_last_points: Optional[int] = None
        self._shop_refresh_count: int = 0
        self._shop_refresh_noop_streak: int = 0
        self._shop_last_week: Optional[int] = None
        self._shop_visit_count: int = 0
        self._in_shop_visit: bool = False
        self._first_shop_delete_done: bool = False
        self._expecting_drink_discard: bool = False
        self._banned_drink_discard_attempts: int = 0
        self._banned_drink_discard_ctx: Optional[Tuple[str, int]] = None
        self._pending_rest_hp_recheck: bool = False
        self._rest_hp_recheck_confirmed: bool = False
        self._rest_hp_recheck_week: Optional[int] = None
        self._rest_hp_recheck_cancels: int = 0
        self.set_preferences(preferences)

    def _resolve_card_base_id(self, card_id: Optional[int], base_id: int = 0) -> int:
        if base_id and int(base_id) > 0:
            return int(base_id)
        cid_i = int(card_id or 0)
        if cid_i <= 0:
            return 0
        row = self.db.get_card_by_id(cid_i)
        return int(row.get("base_id") or cid_i) if row else cid_i

    def _expand_card_ids_and_bases(self, card_ids: Sequence[int]) -> Tuple[Set[int], Set[int]]:
        cids: Set[int] = set()
        bases: Set[int] = set()
        for raw_cid in card_ids or ():
            try:
                cid = int(raw_cid)
            except (TypeError, ValueError):
                continue
            if cid <= 0:
                continue
            cids.add(cid)
            row = self.db.get_card_by_id(cid)
            if row:
                base_id = int(row.get("base_id") or cid)
                bases.add(base_id)
                cids.add(base_id)
                if row.get("upgraded_id"):
                    cids.add(int(row["upgraded_id"]))
                if base_id != cid:
                    base_row = self.db.get_card_by_id(base_id)
                    if base_row and base_row.get("upgraded_id"):
                        cids.add(int(base_row["upgraded_id"]))
            else:
                bases.add(cid)
        return cids, bases

    def set_preferences(
        self,
        preferences: Optional[Union[ProducePreferences, Dict[str, Any]]],
    ) -> None:
        """Apply user card/drink preferences cleanly to agent & battle planner."""
        prefs = (
            preferences
            if isinstance(preferences, ProducePreferences)
            else ProducePreferences.from_dict(preferences)
        )
        self.preferences = prefs
        self._priority_card_ids, self._priority_card_base_ids = self._expand_card_ids_and_bases(
            prefs.priority_card_ids
        )
        self._avoid_card_ids, self._avoid_card_base_ids = self._expand_card_ids_and_bases(
            prefs.avoid_card_ids
        )
        self._keep_card_ids, self._keep_card_base_ids = self._expand_card_ids_and_bases(
            prefs.keep_card_ids
        )
        self._card_score_overrides_exact = {
            int(k): int(v) for k, v in (prefs.card_score_overrides or {}).items() if int(k) > 0
        }
        self._card_score_overrides_by_base = {}
        for raw_k, raw_v in self._card_score_overrides_exact.items():
            b_id = self._resolve_card_base_id(raw_k)
            if b_id > 0:
                self._card_score_overrides_by_base[b_id] = int(raw_v)
        for raw_k, raw_v in self._card_score_overrides_exact.items():
            b_id = self._resolve_card_base_id(raw_k)
            if b_id > 0 and raw_k == b_id:
                self._card_score_overrides_by_base[b_id] = int(raw_v)
        self._preferred_drink_ids = {int(d) for d in prefs.preferred_drink_ids}
        self._banned_drink_ids = {int(d) for d in prefs.banned_drink_ids}
        bnames: List[str] = []
        if 19 in self._banned_drink_ids:
            bnames.extend(["初星スペシャル青汁", "青汁"])
        for did in sorted(self._banned_drink_ids):
            try:
                drow = self.db.get_drink_by_id(did)
                if drow and drow.get("name"):
                    nm = str(drow["name"]).strip()
                    if nm and nm not in bnames:
                        bnames.append(nm)
            except Exception:
                pass
        self._banned_drink_names = tuple(bnames)
        if hasattr(self.battle_planner, "set_banned_drink_ids"):
            self.battle_planner.set_banned_drink_ids(prefs.banned_drink_ids)

    def _is_priority_card(self, card_id: Optional[int], base_id: int = 0) -> bool:
        cid_i = int(card_id or 0)
        b_i = self._resolve_card_base_id(cid_i, base_id=base_id)
        return (cid_i > 0 and cid_i in self._priority_card_ids) or (
            b_i > 0 and b_i in self._priority_card_base_ids
        )

    def _is_avoid_card(self, card_id: Optional[int], base_id: int = 0) -> bool:
        cid_i = int(card_id or 0)
        b_i = self._resolve_card_base_id(cid_i, base_id=base_id)
        return (cid_i > 0 and cid_i in self._avoid_card_ids) or (
            b_i > 0 and b_i in self._avoid_card_base_ids
        )

    def _is_keep_card(self, card_id: Optional[int], base_id: int = 0) -> bool:
        cid_i = int(card_id or 0)
        b_i = self._resolve_card_base_id(cid_i, base_id=base_id)
        return (cid_i > 0 and cid_i in self._keep_card_ids) or (
            b_i > 0 and b_i in self._keep_card_base_ids
        )

    def _get_card_score_override(self, card_id: Optional[int], base_id: int = 0) -> int:
        cid_i = int(card_id or 0)
        if cid_i > 0 and cid_i in self._card_score_overrides_exact:
            return int(self._card_score_overrides_exact[cid_i])
        b_i = self._resolve_card_base_id(cid_i, base_id=base_id)
        if b_i > 0 and b_i in self._card_score_overrides_by_base:
            return int(self._card_score_overrides_by_base[b_i])
        return 0

    def close(self) -> None:
        self.battle_planner.close()
        if self._owns_db:
            self.db.close()

    def bind_tracker(self, tracker: GameStateTracker) -> None:
        """Bind runner's GameStateTracker so produce deck, P-idol, and library are live-accessible."""
        self.tracker = tracker

    def reset_session(self) -> None:
        self._has_entered_produce = False
        self._has_seen_produce_setup = False
        self._produce_event_seen_count = 0
        self._reward_preview_count = 0
        self._last_reward_sig = None
        self._reward_seen_slots.clear()
        self._reward_slot_attempts.clear()
        self._reward_visited_grid_slots.clear()
        self._reward_cancel_attempts = 0
        self._reward_grid_scores.clear()
        self._last_clicked_grid_slot = None
        self._reward_grid_request_scroll = None
        self._reward_grid_can_flip_scroll = False
        self._reward_reroll_attempts = 0
        self._came_from_change_pickup = False
        self._pending_change_gain_card_id = None
        self._pending_change_gain_base_id = None
        self._customize_clicked_tiles.clear()
        self._customize_cards_tried = 0
        self._customize_done_count = 0
        self._shop_card_popup_guard = False
        self._shop_expecting_drink_confirm = False
        self._shop_expecting_priority_card_confirm = None
        self._shop_expecting_priority_card_is_change = False
        self._shop_attempted_drink_ids.clear()
        self._shop_attempted_card_ids.clear()
        self._shop_clicked_slots.clear()
        self._shop_noop_features.clear()
        self._shop_last_clicked_feature = None
        self._shop_last_points = None
        self._shop_refresh_count = 0
        self._shop_refresh_noop_streak = 0
        self._shop_refresh_since_purchase = 0
        self._shop_visit_count = 0
        self._in_shop_visit = False
        self._first_shop_delete_done = False
        self._expecting_drink_discard = False
        self._banned_drink_discard_attempts = 0
        self._banned_drink_discard_ctx = None
        self._pending_rest_hp_recheck = False
        self._rest_hp_recheck_confirmed = False
        self._rest_hp_recheck_week = None
        self._rest_hp_recheck_cancels = 0
        self.battle_planner.reset_battle()

    def reset_battle(self) -> None:
        self.battle_planner.reset_battle()

    def notify_battle_action(self, action: Optional[LegalAction]) -> None:
        self.battle_planner.notify_action_executed(action)

    def export_session(self) -> Dict[str, Any]:
        return {
            "has_entered_produce": self._has_entered_produce,
            "reward_preview_count": self._reward_preview_count,
        }

    def load_session(self, state: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(state, dict):
            return False
        self._has_entered_produce = bool(state.get("has_entered_produce", False))
        self._reward_preview_count = int(state.get("reward_preview_count", 0) or 0)
        return True

    def _resolve_idol_and_effect(
        self,
        obs: Optional[ScreenObservation],
        p_idol_id: Optional[int] = None,
    ) -> Tuple[Optional[int], Optional[str], Optional[PIdolGuide]]:
        pid = p_idol_id
        if pid is None and self.tracker is not None:
            pid = self.tracker.p_idol_id
        if pid is None:
            pid = resolve_p_idol_id(
                plan=(obs.plan if obs else "logic"),
                p_idol_id=None,
                deck_ids=getattr(self.tracker, "produce_deck", None),
                hand_cards=(obs.cards if obs else None),
                default_logic_to_sekirara=False,
            )

        rec_eff: Optional[str] = getattr(self.tracker, "recommended_effect", None) if self.tracker else None
        if not rec_eff and pid is not None:
            p_row = self.db.get_p_idol_by_id(int(pid))
            if p_row and p_row.get("recommended_effect"):
                rec_eff = str(p_row["recommended_effect"])

        guide = get_p_idol_guide(pid)
        if not rec_eff and guide is not None:
            rec_eff = guide.recommended_effect
        return pid, rec_eff, guide

    def _effective_p_idol_id(
        self,
        p_idol_id: Optional[int] = None,
        guide: Optional[PIdolGuide] = None,
    ) -> Optional[int]:
        if p_idol_id is not None:
            return int(p_idol_id)
        if guide is not None and getattr(guide, "p_idol_id", None) is not None:
            return int(guide.p_idol_id)
        if self.tracker is not None and getattr(self.tracker, "p_idol_id", None) is not None:
            return int(self.tracker.p_idol_id)
        return None

    def _is_forbidden_card(
        self,
        *,
        card_id: Optional[int],
        base_id: int = 0,
        cname: str = "",
        title: str = "",
        p_idol_id: Optional[int] = None,
        guide: Optional[PIdolGuide] = None,
    ) -> bool:
        cid_i = int(card_id) if card_id else 0
        b_i = self._resolve_card_base_id(cid_i, base_id=base_id)
        if self._is_avoid_card(cid_i, base_id=b_i):
            return True
        pid = self._effective_p_idol_id(p_idol_id=p_idol_id, guide=guide)
        if pid is None:
            return False
        forb_bases = _FORBIDDEN_BASE_IDS_BY_P_IDOL.get(pid)
        if forb_bases:
            if b_i in forb_bases or cid_i in forb_bases or (pid == 163 and cid_i in (152, 153, 426, 427)):
                return True
        forb_names = _FORBIDDEN_NAME_TOKENS_BY_P_IDOL.get(pid, ())
        nm_chk = f"{cname} {title}"
        return any(tok in nm_chk for tok in forb_names)

    def _is_trouble_card_id(self, card_id: Optional[int]) -> bool:
        if not card_id:
            return False
        cid_i = int(card_id)
        if cid_i == 23:
            return True
        row = self.db.get_card_by_id(cid_i)
        if not row:
            return False
        return (
            str(row.get("type") or "").lower() == "trouble"
            or str(row.get("rarity") or "").upper() == "T"
        )

    def _deck_has_trouble_card(self) -> bool:
        deck_ids = list(getattr(self.tracker, "produce_deck", []) or []) if self.tracker is not None else []
        return any(self._is_trouble_card_id(cid_d) for cid_d in deck_ids)

    def _deck_has_removable_unwanted_card(
        self,
        *,
        target_effect: Optional[str],
        guide: Optional[PIdolGuide],
        plan: str,
        p_idol_id: Optional[int] = None,
        ignore_upgrade: bool = False,
    ) -> Optional[bool]:
        """Return True if any non-trouble card in produce_deck scores as unwanted for swap/removal."""
        deck_ids = list(getattr(self.tracker, "produce_deck", []) or []) if self.tracker is not None else []
        if not deck_ids:
            return None
        for cid_d in deck_ids:
            cid_i = int(cid_d)
            crow = self.db.get_card_by_id(cid_i)
            ctype = str(crow.get("type") or "") if crow else ""
            if cid_i == 23 or ctype == "trouble":
                continue
            rem_score = self._score_skill_card(
                card_id=cid_i,
                title=str(crow.get("name") or "") if crow else "",
                desc=str(crow.get("description") or "") if crow else "",
                target_effect=target_effect,
                guide=guide,
                plan=plan,
                for_removal=True,
                p_idol_id=p_idol_id,
                check_same_card=False,
                ignore_upgrade=ignore_upgrade,
            )
            if rem_score >= 100:
                return True
        return False

    @staticmethod
    def _stat_values(
        obs: Optional[ScreenObservation],
        *,
        p_idol_id: Optional[int] = None,
        db: Optional[Any] = None,
    ) -> Dict[str, int]:
        from demo.core.idol_guides import get_default_idol_stats

        dflt = get_default_idol_stats(p_idol_id, db=db)
        stats = (obs.idol_stats if obs and obs.idol_stats else None) or dflt
        return {
            "vo": min(STAT_CAP, max(0, int(stats.get("vocal", dflt["vocal"]) or dflt["vocal"]))),
            "da": min(STAT_CAP, max(0, int(stats.get("dance", dflt["dance"]) or dflt["dance"]))),
            "vi": min(STAT_CAP, max(0, int(stats.get("visual", dflt["visual"]) or dflt["visual"]))),
        }

    @classmethod
    def _stat_rank_map(
        cls,
        obs: Optional[ScreenObservation],
        *,
        p_idol_id: Optional[int] = None,
        db: Optional[Any] = None,
    ) -> Dict[str, int]:
        from demo.core.idol_guides import get_idol_stat_order

        vals = cls._stat_values(obs, p_idol_id=p_idol_id, db=db)
        order = get_idol_stat_order(p_idol_id, db=db)
        key_short = {"vocal": "vo", "dance": "da", "visual": "vi"}
        tie_rank = {key_short[k]: (3 - idx) for idx, k in enumerate(order)}
        ordered = sorted(
            vals.items(),
            key=lambda kv: (kv[1], tie_rank.get(kv[0], 0)),
            reverse=True,
        )
        return {k: (3 - idx) for idx, (k, _) in enumerate(ordered)}

    @staticmethod
    def _estimate_schedule_gain(text_l: str, is_sp: bool) -> int:
        if is_sp:
            return 420
        if any(k in text_l for k in ("公開レッスン", "追い込み", "レッスン")):
            return 280
        if "授業" in text_l:
            return 220
        return 80

    @classmethod
    def _stat_attr_bonus(
        cls,
        obs: Optional[ScreenObservation],
        text_l: str,
        *,
        est_gain: int = 240,
        is_sp: bool = False,
        prefer_non_highest: bool = False,
        enforce_floor: bool = False,
        p_idol_id: Optional[int] = None,
        db: Optional[Any] = None,
    ) -> int:
        attr: Optional[str] = None
        if "vo" in text_l or "ボーカル" in text_l:
            attr = "vo"
        elif "vi" in text_l or "ビジュアル" in text_l:
            attr = "vi"
        elif "da" in text_l or "ダンス" in text_l:
            attr = "da"
        if attr is None:
            return 0
        vals = cls._stat_values(obs, p_idol_id=p_idol_id, db=db)
        ranks = cls._stat_rank_map(obs, p_idol_id=p_idol_id, db=db)
        week = int(getattr(obs, "week", 1) or 1)
        reserve = 140 if (ranks.get(attr, 0) == 3 and week <= 6) else 60
        soft_cap = STAT_CAP - reserve
        val = vals[attr]
        headroom = max(0, soft_cap - val)
        useful_gain = min(est_gain, headroom)
        overflow = max(0, est_gain - headroom)
        if val >= STAT_CAP or headroom <= 30:
            return -9_500
        if overflow > 0:
            if useful_gain < 120:
                return -8_800
            if useful_gain < int(est_gain * 0.55):
                return -3_200 - overflow * 6
            return -1_200 - overflow * 4
        if enforce_floor:
            below_floor = [(k, v) for k, v in vals.items() if v < cls.MIN_STAT_FLOOR]
            if below_floor and val < cls.MIN_STAT_FLOOR:
                min_below = min(v for _, v in below_floor)
                deficit = min(cls.MIN_STAT_FLOOR, cls.MIN_STAT_FLOOR - val)
                if val == min_below:
                    return 1_500 + deficit
                return 900 + deficit
        if prefer_non_highest:
            return {3: 0, 2: 120, 1: 80}.get(ranks.get(attr, 0), 0)
        return ranks.get(attr, 0) * 40

    def _score_skill_card(
        self,
        *,
        card_id: Optional[int],
        title: str,
        desc: str,
        target_effect: Optional[str],
        guide: Optional[PIdolGuide],
        plan: str,
        for_removal: bool = False,
        p_idol_id: Optional[int] = None,
        in_change_pickup: bool = False,
        has_removable_unwanted: Optional[bool] = None,
        check_same_card: bool = True,
        ignore_upgrade: bool = False,
        is_customized: bool = False,
    ) -> int:
        """Score a skill card by idol `recommended_effect`, PIdolGuide, rarity, and deck quality."""
        row = self.db.get_card_by_id(int(card_id)) if card_id else None
        cname = str(row["name"]) if row else str(title or "")
        ctype = str(row.get("type") or "") if row else ""
        rarity = str(row.get("rarity") or "").upper() if row else ""
        dsl = f"{row.get('dsl_actions', '')} {row.get('dsl_effects', '')}".lower() if row else ""
        dsl_cond = str(row.get("dsl_conditions") or "").lower() if row else ""
        c_rec = str(row.get("recommended_effect") or "") if row else ""
        text_l = f"{cname} {title} {desc} {dsl}".lower()
        upgraded = (
            False
            if ignore_upgrade
            else bool((row and row.get("upgraded")) or "+" in cname or "+" in title)
        )
        base_id = int(row.get("base_id") or card_id) if (row is not None and card_id) else (int(card_id) if card_id else 0)
        family_ids: Set[int] = set()
        if card_id:
            family_ids.add(int(card_id))
        if base_id:
            family_ids.add(base_id)
        if row and row.get("upgraded_id"):
            family_ids.add(int(row["upgraded_id"]))
        elif base_id and base_id != (int(card_id) if card_id else 0):
            b_row = self.db.get_card_by_id(base_id)
            if b_row and b_row.get("upgraded_id"):
                family_ids.add(int(b_row["upgraded_id"]))

        # Check whether this card has customizations (加点强化 / カスタマイズ済) from vision badge OR tracker metadata
        card_has_custom = bool(is_customized) or ("已加点强化" in desc) or ("カスタマイズ済" in desc)
        if (not card_has_custom) and self.tracker is not None and card_id:
            tr_custs = getattr(self.tracker, "card_customizations", None) or {}
            for fid in family_ids:
                cm = tr_custs.get(fid) or tr_custs.get(str(fid))
                if isinstance(cm, dict) and (cm.get("levels") or cm.get("modifiers")):
                    card_has_custom = True
                    break

        # Trouble cards (`眠気` / `トラブル`): terrible to pick, top priority to remove
        if card_id == 23 or ctype == "trouble" or "眠気" in cname or "トラブル" in cname:
            return 5000 if for_removal else -10_000

        # Customized cards (加点强化済): NEVER swap out (チェンジ does NOT inherit customizations!) or delete!
        if for_removal and card_has_custom:
            return -18_000

        # Extra-turn cards (e.g. #170/#171 私がスター, #728 レジェンドスター): must NEVER be deleted/swapped away
        is_extra_turn_card = (
            (card_id in (170, 171, 728))
            or ("turnsremaining+=" in dsl)
            or ("ターン追加" in text_l)
            or ("私がスター" in cname)
            or ("私がスター" in title)
        )
        is_same_change_target = bool(
            for_removal
            and check_same_card
            and self._came_from_change_pickup
            and self._pending_change_gain_base_id
            and base_id == self._pending_change_gain_base_id
        )
        if is_extra_turn_card:
            if for_removal and not (is_same_change_target and has_removable_unwanted is False):
                return -8_000

        # User-configured avoided cards: hard penalize pickup and prioritize removal/swap
        if self._is_avoid_card(card_id, base_id=base_id):
            return 5_000 if for_removal else -20_000

        # Idol-specific forbidden cards (e.g., #152/#153 届いて！ and #426/#427 キセキの魔法 for p_idol_id=163)
        if self._is_forbidden_card(
            card_id=card_id,
            base_id=base_id,
            cname=cname,
            title=title,
            p_idol_id=p_idol_id,
            guide=guide,
        ):
            return 2_500 if for_removal else -6_000

        # Avoid picking `成長痛` (#714/#715) and unlimited in-battle `眠気`-generating cards
        is_seichoutsu = (card_id in (714, 715)) or ("成長痛" in cname) or ("成長痛" in title)
        generates_nemuki = (
            ("addcardtodeck(23" in dsl)
            or ("addcardtotopofdeck(23" in dsl)
            or ("眠気を山札" in text_l)
        )
        is_unlimited_nemuki = generates_nemuki and (
            row is None or (row.get("limit_count") is None and "レッスン中1回" not in text_l)
        )
        if is_seichoutsu:
            if for_removal:
                return 900 if not upgraded else 650
            return -3_500
        if is_unlimited_nemuki:
            if for_removal:
                return 700 if not upgraded else 450

        score = 0
        if is_extra_turn_card:
            score += 4_500
        elif is_unlimited_nemuki:
            score -= 2_200
        elif generates_nemuki and not for_removal:
            score -= 350

        if row is not None:
            c_plan = str(row.get("plan") or "all").lower()
            if c_plan not in ("all", "free", "", plan.lower()):
                score -= 800

        # Primary `recommended_effect` alignment (guard against anti-synergy cap conditions like `if:goodConditionTurns<=15` on #751 精神統一!)
        is_anti_good_cond_conc_card = bool(
            target_effect == "goodConditionTurns"
            and (
                "goodconditionturns<=" in dsl
                or "goodconditionturns<=" in dsl_cond
                or bool(re.search(r"好調が\d+ターン以下", text_l))
            )
            and "goodconditionturns+=" not in dsl
            and "perfectconditionturns+=" not in dsl
        )
        if is_anti_good_cond_conc_card:
            if for_removal:
                return 900 if not upgraded else 750
            score -= 1_800
        elif target_effect:
            te_l = target_effect.lower()
            has_pos_dsl_gain = (
                f"{te_l}+=" in dsl
                or f"{te_l}*=" in dsl
                or f"set{te_l}" in dsl
                or (target_effect == "goodConditionTurns" and "perfectconditionturns+=" in dsl)
            )
            if c_rec == target_effect:
                score += 650
            elif has_pos_dsl_gain:
                score += 500
            prim_kws = _EFFECT_KEYWORDS.get(target_effect, ())
            # Strip negative cap clauses ("好調が15ターン以下") before checking positive primary keyword alignment
            cleaned_text_for_prim = re.sub(r"好調が\d+ターン以下", "", text_l)
            if any(kw in cleaned_text_for_prim for kw in prim_kws):
                score += 380
            sec_kws = _SECONDARY_EFFECT_KEYWORDS.get(target_effect, ())
            if any(kw in text_l for kw in sec_kws):
                is_pure_conc_on_gc = (
                    target_effect == "goodConditionTurns"
                    and "concentration+=" in dsl
                    and not has_pos_dsl_gain
                    and "cardusesremaining+=" not in dsl
                    and "doublecardeffectcards+=" not in dsl
                )
                score += 30 if is_pure_conc_on_gc else 120

        # PIdolGuide affinity / anti-synergy
        if guide is not None:
            rec_hit = (
                any(fid in guide.recommended_cards for fid in family_ids)
                if ignore_upgrade
                else bool(card_id and int(card_id) in guide.recommended_cards)
            )
            caut_hit = (
                any(fid in guide.caution_cards for fid in family_ids)
                if ignore_upgrade
                else bool(card_id and int(card_id) in guide.caution_cards)
            )
            if rec_hit:
                score += 420
            elif any(k in cname for k in guide.recommended_card_names):
                score += 360
            if caut_hit:
                score -= 350
            elif any(k in cname for k in guide.caution_card_names):
                score -= 280

        if row is not None and row.get("is_signature"):
            score += 350
        if upgraded:
            score += 90
        is_free_action_card = bool("cardusesremaining+=" in dsl or "使用数追加" in text_l)
        if is_free_action_card:
            score += 140

        # Protect multiplicative 好印象 scaling (#521 夢色リップ+) & GI effect-buff engines (#920/#617/#778)
        if "goodimpressionturns*=" in dsl or "setgoodimpressionturnseffectbuff" in dsl:
            score += 2_200

        deck_count = 0
        family_in_deck = False
        if self.tracker is not None and card_id:
            deck_ids_tr = self.tracker.produce_deck or []
            if ignore_upgrade:
                for cid_d in deck_ids_tr:
                    cid_di = int(cid_d)
                    if cid_di in family_ids:
                        deck_count += 1
                        family_in_deck = True
                        continue
                    r_d = self.db.get_card_by_id(cid_di)
                    if r_d and int(r_d.get("base_id") or cid_di) == base_id:
                        deck_count += 1
                        family_in_deck = True
            else:
                deck_count = sum(1 for cid_d in deck_ids_tr if int(cid_d) == int(card_id))
                if base_id:
                    for cid_d in deck_ids_tr:
                        cid_di = int(cid_d)
                        if cid_di == int(card_id) or cid_di == base_id:
                            family_in_deck = True
                            break
                        r_d = self.db.get_card_by_id(cid_di)
                        if r_d and int(r_d.get("base_id") or cid_di) == base_id:
                            family_in_deck = True
                            break

        is_core_keep = bool(
            (card_id and int(card_id) in _CORE_KEEP_CARD_IDS)
            or (base_id and base_id in _CORE_KEEP_BASE_IDS)
        )
        is_unique_card = bool(
            row is not None
            and (
                row.get("is_unique_in_deck")
                or row.get("is_unique")
                or ("重複不可" in str(row.get("description") or ""))
            )
        )
        is_logic_gi_build = (target_effect == "goodImpressionTurns") or (
            plan.lower() == "logic" and target_effect != "motivation"
        )
        is_elite_active = bool(
            ctype == "active"
            and (
                (base_id and base_id in _ELITE_LOGIC_ACTIVE_BASE_IDS)
                or (row is not None and row.get("is_signature"))
                or rarity in ("L", "LEGEND")
                or "setgoodimpressionturnseffectbuff" in dsl
                or "goodimpressionturns*=" in dsl
            )
        )

        # Strongly prefer Mental (M) cards over ordinary Active (A) cards for Logic / GI builds
        if is_logic_gi_build:
            if ctype == "mental":
                score += 850
                if row is not None and row.get("limit_count") == 1:
                    score += 200
            elif ctype == "active" and not is_elite_active:
                score -= 1_100

        # Avoid picking dead duplicates of 重複不可 / unique-in-deck cards already in deck
        if (not for_removal) and is_unique_card and family_in_deck and not in_change_pickup:
            score -= 4_000

        # Same-card swap bonus on Page 1 (`チェンジで獲得`) when no unwanted cards remain:
        # prefer an already-owned UNCUSTOMIZED family so Page 2 can swap X -> X cleanly without losing customizations or core cards.
        if (not for_removal) and in_change_pickup and family_in_deck and (not card_has_custom) and (has_removable_unwanted is False):
            score += 4_000 if upgraded else 3_500
            if not is_unique_card:
                score += 400

        from demo.core.idol_guides import resolve_card_family_meta

        tracker_hooks = (getattr(self.tracker, "card_memory_hooks", None) or {}) if self.tracker is not None else {}
        hook_meta = resolve_card_family_meta(tracker_hooks, card_id, db=self.db)
        is_misuzu_logic = (p_idol_id in (None, 163)) and (plan.lower() == "logic")
        has_real_hook = bool(
            hook_meta.get("force_initial_hand")
            or hook_meta.get("free_extra_use_first_play")
            or int(hook_meta.get("extra_use_limit") or 0) > 0
            or int(hook_meta.get("draw_on_play") or 0) > 0
            or (hook_meta.get("exiles_trouble_on_play") and base_id != 426)
            or (is_misuzu_logic and base_id in (126, 128, 136, 296, 298))
        )
        has_memory_protection = bool(has_real_hook and not (for_removal and deck_count >= 2))
        if has_memory_protection:
            score += 2_500 if for_removal else 1_600

        rarity_pts = {"LEGEND": 240, "SSR": 200, "SR": 120, "R": 60, "N": 10}
        score += rarity_pts.get(rarity, 30)

        # Protect core Logic/H.I.F loop & finisher cards from swaps/removals
        if is_core_keep:
            score += 2_000

        # Apply user-configured card priorities, locks, and manual numeric score overrides
        user_priority = self._is_priority_card(card_id, base_id=base_id)
        user_keep = self._is_keep_card(card_id, base_id=base_id)
        override_delta = self._get_card_score_override(card_id, base_id=base_id)

        if user_priority:
            score += 20_000
        elif user_keep and for_removal:
            score += 15_000
        if override_delta:
            score += override_delta

        has_strong_user_protection = (
            user_priority
            or (for_removal and has_memory_protection)
            or (for_removal and user_keep)
            or (for_removal and override_delta >= 500)
        )

        protect_free_action_active = is_free_action_card and not is_logic_gi_build
        if for_removal and protect_free_action_active:
            score += 760
        is_ordinary_unupgraded_active = (
            ctype == "active"
            and not upgraded
            and not is_elite_active
            and not protect_free_action_active
            and not is_core_keep
            and not has_strong_user_protection
        )
        is_non_elite_active = (
            ctype == "active"
            and not is_elite_active
            and not protect_free_action_active
            and not is_core_keep
            and not has_strong_user_protection
        )

        # Basic starter N cards & non-elite Active cards are top removal/swap candidates (ahead of M cards)
        if rarity == "N" and not upgraded and not has_strong_user_protection:
            if for_removal:
                return 1_500 - override_delta
            score -= 120
        elif is_ordinary_unupgraded_active and rarity in ("R", "SR"):
            if for_removal:
                return (1_100 if rarity == "R" else 900) - override_delta
            score -= 350
        elif is_logic_gi_build and is_non_elite_active:
            if for_removal:
                return (650 if not upgraded else 550) - override_delta
            if not upgraded:
                score -= 350
        elif (not is_core_keep and not has_strong_user_protection) and (
            (rarity == "R" and not upgraded)
            or ((card_id in (68, 69, 94) or (ignore_upgrade and base_id in (68, 94))) and not upgraded)
        ):
            score -= 600
            if for_removal:
                return max(450, -score)
        elif rarity == "SR" and not upgraded and not is_core_keep and not protect_free_action_active and not has_strong_user_protection and ctype != "mental":
            if for_removal:
                return 250 - override_delta

        # Same-card swap preference: on Page 2 of チェンジ, if Page 1 picked a family matching this card,
        # swapping X -> X preserves deck composition when no unwanted cards remain.
        if is_same_change_target and not (user_priority or user_keep):
            if has_removable_unwanted is False:
                return 500 if not upgraded else 300
            return -100

        # Duplicate-copy relief: prefer swapping/deleting a duplicate over a unique core card
        if for_removal and deck_count >= 2:
            score -= 1_200

        return -score if for_removal else score

    def _decide_home(self, legal_actions: List[LegalAction]) -> DecisionResult:
        if self._has_entered_produce:
            first = legal_actions[0]
            return DecisionResult(
                choice_id=first.choice_id,
                reason="LocalAgent: Produce cycle completed and returned to Home; holding without re-entry.",
                source="local_hybrid",
            )
        prod_act = next(
            (
                a for a in legal_actions
                if "プロデュース" in str((a.payload or {}).get("name") or a.description)
            ),
            legal_actions[0],
        )
        return DecisionResult(
            choice_id=prod_act.choice_id,
            reason=f"LocalAgent(home): Entering Produce via {prod_act.description}",
            source="local_hybrid",
        )

    def _decide_produce_setup(self, legal_actions: List[LegalAction]) -> DecisionResult:
        if self._has_entered_produce:
            first = legal_actions[0]
            return DecisionResult(
                choice_id=first.choice_id,
                reason="LocalAgent: Produce cycle completed and returned to setup; holding without re-entry.",
                source="local_hybrid",
            )
        self._has_seen_produce_setup = True
        candidates = [
            a for a in legal_actions
            if not (a.payload or {}).get("is_loadout_target")
            and not any(bad in str((a.payload or {}).get("name") or a.description) for bad in ("ジュエル", "リタイア"))
        ]
        if not candidates:
            candidates = [a for a in legal_actions if not (a.payload or {}).get("is_loadout_target")] or legal_actions

        def _setup_rank(act: LegalAction) -> int:
            nm = str((act.payload or {}).get("name") or act.description)
            if "本戦" in nm:
                return 1000
            if "プロデュース開始" in nm or "開始する" in nm:
                return 900
            if "決定" in nm:
                return 800
            if "次へ" in nm:
                return 700
            if "選抜試験" in nm:
                return 100
            return 300

        best = max(candidates, key=_setup_rank)
        return DecisionResult(
            choice_id=best.choice_id,
            reason=f"LocalAgent(setup): Selected {best.description}",
            source="local_hybrid",
        )

    def _decide_produce_schedule(
        self,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation],
        p_idol_id: Optional[int] = None,
    ) -> DecisionResult:
        self._has_entered_produce = True
        self._reward_preview_count = 0
        self._last_reward_sig = None
        resolved_pid, _, _ = self._resolve_idol_and_effect(obs, p_idol_id=p_idol_id)

        maybe_discard = self._maybe_pick_banned_drink_discard(
            legal_actions,
            "produce_schedule",
            int(getattr(obs, "week", 0) or 0) if obs is not None else 0,
        )
        if maybe_discard is not None:
            return maybe_discard

        stamina = int(getattr(obs, "stamina", 30) or 30)
        max_stam = max(1, int(getattr(obs, "max_stamina", 44) or 44))
        low_hp = stamina < 15
        deck_ids_sched = list(getattr(self.tracker, "produce_deck", []) or [])
        has_trouble_sched = any(self._is_trouble_card_id(cid_s) for cid_s in deck_ids_sched)
        p_pts_sched = int(getattr(obs, "p_points", 0) or 0) if obs is not None else 0
        wk_sched = int(getattr(obs, "week", 0) or 0) if obs is not None else 0

        if self._pending_rest_hp_recheck:
            self._pending_rest_hp_recheck = False
            if stamina < 15:
                rest_act = next(
                    (
                        a for a in legal_actions
                        if any(
                            k in f"{(a.payload or {}).get('title') or (a.payload or {}).get('name') or ''} {a.description or ''}".lower()
                            for k in ("休む", "休息", "rest")
                        )
                    ),
                    None,
                )
                has_w6_soudan_override = bool(
                    stamina >= 6
                    and (has_trouble_sched or (wk_sched >= 5 and p_pts_sched >= 125))
                    and any(
                        any(
                            k in f"{(a.payload or {}).get('title') or (a.payload or {}).get('name') or ''} {a.description or ''}".lower()
                            for k in ("特別指導", "相談")
                        )
                        for a in legal_actions
                    )
                )
                if rest_act is not None and not has_w6_soudan_override:
                    self._rest_hp_recheck_confirmed = True
                    self._rest_hp_recheck_week = wk_sched
                    return DecisionResult(
                        choice_id=rest_act.choice_id,
                        reason=f"LocalAgent(schedule): Re-verified HP={stamina}/{max_stam} < 15; confirming rest via {rest_act.description}",
                        source="local_hybrid",
                    )
            else:
                self._rest_hp_recheck_confirmed = False
                self._rest_hp_recheck_week = None

        def _is_sp_action(act: LegalAction) -> bool:
            p = act.payload or {}
            nm = str(p.get("title") or p.get("name") or "")
            combined = f"{nm} {act.description}".lower()
            return bool(
                p.get("is_sp")
                or "spレッスン" in combined
                or "sp授業" in combined
                or "sp発生" in combined
                or "sp公開" in combined
                or ("sp" in nm.lower() and "skip" not in nm.lower())
            )

        def _score_sched(act: LegalAction) -> int:
            p = act.payload or {}
            nm = str(p.get("title") or p.get("name") or "")
            desc = str(act.description or "")
            combined = f"{nm} {desc}".lower()
            if any(bad in combined for bad in ("ジュエル", "リタイア")):
                return -10_000
            if "トラブル" in combined:
                return -2_000

            is_sp = _is_sp_action(act)
            is_jugyo = ("授業" in combined) and ("レッスン" not in combined)
            attr_bonus = self._stat_attr_bonus(
                obs,
                combined,
                est_gain=self._estimate_schedule_gain(combined, is_sp),
                is_sp=is_sp,
                prefer_non_highest=is_jugyo,
                enforce_floor=is_jugyo,
                p_idol_id=resolved_pid,
                db=self.db,
            )
            if is_sp:
                return 10_000 + attr_bonus

            is_rest = any(k in combined for k in ("休む", "休息", "rest"))
            is_soudan = any(k in combined for k in ("特別指導", "相談"))
            if is_soudan and stamina >= 6 and (has_trouble_sched or (wk_sched >= 5 and p_pts_sched >= 125)):
                # W6 相談 is the critical pre-Battle-1 shop to delete #23 眠気 / trouble cards
                return 6_000 + attr_bonus
            if low_hp and is_rest:
                return 5_000

            base = 300
            if any(k in combined for k in ("追い込み", "公開レッスン", "授業", "ボーカル", "ダンス", "ビジュアル", "レッスン")):
                base = 2_000
            elif is_soudan:
                base = 1_800
            elif any(k in combined for k in ("差し入れ", "活動支給")):
                base = 1_500
            elif is_rest:
                base = 900
            elif "おでかけ" in combined:
                base = 200
            return base + attr_bonus

        best = max(legal_actions, key=_score_sched)
        best_combined = f"{(best.payload or {}).get('title') or (best.payload or {}).get('name') or ''} {best.description}".lower()
        if not any(k in best_combined for k in ("休む", "休息", "rest")):
            self._rest_hp_recheck_confirmed = False
            self._rest_hp_recheck_cancels = 0
        if _is_sp_action(best) and self._stat_attr_bonus(obs, best_combined, est_gain=420, is_sp=True, p_idol_id=resolved_pid, db=self.db) > -5_000:
            return DecisionResult(
                choice_id=best.choice_id,
                reason=f"LocalAgent(schedule): Prioritizing SP course -> {best.description}",
                source="local_hybrid",
            )
        return DecisionResult(
            choice_id=best.choice_id,
            reason=f"LocalAgent(schedule): Chosen {best.description} (HP={stamina}/{max_stam})",
            source="local_hybrid",
        )

    def _decide_produce_event(
        self,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation],
        p_idol_id: Optional[int],
    ) -> DecisionResult:
        self._produce_event_seen_count += 1
        if (
            self._has_seen_produce_setup
            or (obs is not None and int(obs.week or 1) >= 2)
            or self._produce_event_seen_count >= 2
            or bool(self.tracker is not None and self.tracker.produce_deck)
        ):
            self._has_entered_produce = True
        resolved_pid_ev, target_eff, guide = self._resolve_idol_and_effect(obs, p_idol_id=p_idol_id)
        eff_kws = _EFFECT_KEYWORDS.get(target_eff or "goodImpressionTurns", ("好印象",))
        sec_kws = _SECONDARY_EFFECT_KEYWORDS.get(target_eff or "goodImpressionTurns", ("やる気", "元気"))
        branch_map = {
            "goodImpressionTurns": (("好印象",), ("やる気",)),
            "motivation": (("やる気",), ("好印象",)),
            "goodConditionTurns": (("好調",), ("集中",)),
            "concentration": (("集中",), ("好調",)),
            "fullPowerPoint": (("全力",), ("温存", "強気")),
            "preservation": (("温存",), ("全力", "強気")),
            "strength": (("強気",), ("全力", "温存")),
        }
        target_branch_kws, opp_branch_kws = branch_map.get(target_eff or "", ((), ()))
        is_jugyo_event = bool(
            obs and ("授業" in (obs.week_activity or "") or "授業" in (obs.raw_text or ""))
        )
        raw_ev_txt = (obs.raw_text or "") if obs is not None else ""
        is_jugyo_branch_event = is_jugyo_event or any(
            tok in raw_ev_txt
            for tok in ("自信の方はどうですか", "セレクトチェンジ", "余裕です", "不安です", "勝ってきます")
        ) or any(
            any(
                tok in f"{(a.payload or {}).get('name') or ''} {a.description or ''}"
                for tok in ("自信の方はどうですか", "セレクトチェンジ", "余裕です", "不安です", "勝ってきます", "授業选项")
            )
            for a in legal_actions
        )

        def _strip_safe_trouble_phrase(text: str) -> str:
            return (
                text.replace("トラブルカード以外", "")
                .replace("トラプルカード以外", "")
                .replace("トラブルカードをのぞく", "")
            )

        def _has_trouble_penalty(act: LegalAction) -> bool:
            p = act.payload or {}
            t_raw = f"{p.get('title') or p.get('name') or ''} {act.description or ''}".lower()
            t_clean = _strip_safe_trouble_phrase(t_raw)
            return any(tk in t_clean for tk in ("トラブル", "トラプル", "眠気", "睡眠", "干扰卡", "ル追加", "ブル追加"))

        non_trouble_pos: Dict[int, int] = {}
        nt_counter = 0
        for a in legal_actions:
            if not _has_trouble_penalty(a):
                p_a = a.payload or {}
                opt_i = p_a.get("option_index")
                non_trouble_pos[a.choice_id] = int(opt_i) if opt_i is not None else nt_counter
                nt_counter += 1

        def _score_event(act: LegalAction) -> int:
            p = act.payload or {}
            cid = p.get("card_id")
            title = str(p.get("title") or p.get("name") or "")
            desc = str(act.description or "")
            pv_text = str(p.get("preview_text") or "").lower()
            combined = f"{title} {desc} {pv_text}".lower()
            combined_no_safe_trouble = _strip_safe_trouble_phrase(combined)

            if any(bad in combined for bad in ("ジュエル", "リタイア")):
                return -10_000
            if cid:
                return 2_000 + self._score_skill_card(
                    card_id=int(cid),
                    title=title,
                    desc=desc,
                    target_effect=target_eff,
                    guide=guide,
                    plan=(obs.plan if obs else "logic"),
                )
            score = 0
            if any(tk in combined_no_safe_trouble for tk in ("トラブル", "トラプル", "眠気", "睡眠", "干扰卡", "ル追加", "ブル追加")):
                score -= 4_500
            if any(k in combined for k in eff_kws):
                score += 1_200 if target_eff == "goodConditionTurns" else 900
            elif target_eff == "goodConditionTurns" and "集中" in combined:
                # On 好調 idols (e.g. #139 姫崎莉波), avoid picking pure 集中 lesson/event paths over 好調!
                score -= 700
            elif any(k in combined for k in sec_kws):
                score += 400
            if pv_text and target_branch_kws:
                if any(k in pv_text for k in target_branch_kws):
                    score += 3_000
                elif any(k in pv_text for k in opp_branch_kws):
                    score -= 3_500
            elif is_jugyo_branch_event and nt_counter >= 2 and act.choice_id in non_trouble_pos:
                pos_idx = non_trouble_pos[act.choice_id]
                # 授業 events systematically place 好印象/好調 on option #1 (idx 0) and やる気/集中 on option #2 (idx 1)
                if target_eff in ("goodImpressionTurns", "goodConditionTurns") and pos_idx == 0:
                    score += 800
                elif target_eff in ("motivation", "concentration") and pos_idx == 1:
                    score += 800
            attr_bonus = self._stat_attr_bonus(
                obs,
                combined,
                est_gain=80,
                is_sp=False,
                prefer_non_highest=is_jugyo_event,
                enforce_floor=is_jugyo_event,
                p_idol_id=resolved_pid_ev,
                db=self.db,
            )
            if attr_bonus < 0:
                score += attr_bonus
            elif is_jugyo_event:
                if attr_bonus >= 900:
                    score += attr_bonus
                else:
                    score += 180 if attr_bonus >= 120 else (120 if attr_bonus > 0 else 0)
            else:
                score += (attr_bonus // 40) * 60
            if "削除" in combined or "強化" in combined:
                score += 250
            if "スキップ" in combined or "何もしない" in combined:
                score -= 200
            return score

        best = max(legal_actions, key=_score_event)
        best_p = best.payload or {}
        is_sel = bool(best_p.get("is_selected"))
        pv_best = str(best_p.get("preview_text") or "")
        if is_jugyo_branch_event and not is_sel:
            if not any(k in f"{best_p.get('title') or best_p.get('name') or ''} {best.description or ''}" for k in target_branch_kws):
                if best.payload is None:
                    best.payload = {}
                best.payload["preview_only"] = True
        if is_sel and pv_best and any(k in pv_best for k in target_branch_kws):
            ev_reason = f"LocalAgent(event): Confirming matched {target_eff or 'plan'} preview -> {best.description}"
        elif any(
            bool((a.payload or {}).get("is_selected"))
            and any(k in str((a.payload or {}).get("preview_text") or "") for k in opp_branch_kws)
            for a in legal_actions
        ):
            ev_reason = f"LocalAgent(event): Switching away from opposite branch preview -> {best.description}"
        elif best_p.get("preview_only"):
            ev_reason = f"LocalAgent(event): Previewing aligned {target_eff or 'plan'} slot -> {best.description}"
        else:
            ev_reason = f"LocalAgent(event): Matching {target_eff or 'plan'} -> {best.description}"
        return DecisionResult(
            choice_id=best.choice_id,
            reason=ev_reason,
            source="local_hybrid",
        )

    def _is_banned_drink(
        self,
        drink_id: Optional[int],
        title: str = "",
        desc: str = "",
    ) -> bool:
        """Return True for user-banned P-drinks (default #19 初星スペシャル青汁)."""
        if drink_id is not None and int(drink_id) in self._banned_drink_ids:
            return True
        combined = f"{title or ''} {desc or ''}"
        low = combined.lower()
        if 19 in self._banned_drink_ids and (
            "青汁" in combined
            or "ランダムな強化済みスキルカード" in combined
            or "addrandomupgradedssrcardtohand" in low
        ):
            return True
        for bnm in self._banned_drink_names:
            if bnm and bnm in combined:
                return True
        return False

    def _find_banned_drink_discard_action(
        self, legal_actions: Sequence[LegalAction]
    ) -> Optional[LegalAction]:
        for a in legal_actions:
            p = a.payload or {}
            nm = str(p.get("name") or "")
            if not nm.startswith("破棄候補_"):
                continue
            did = p.get("drink_id")
            if did is None:
                m_did = re.match(r"^破棄候補_(\d+)$", nm)
                if m_did:
                    did = int(m_did.group(1))
            # Check only the held slot's drink_id & title; a.description lists reward candidates.
            if self._is_banned_drink(did, str(p.get("title") or nm), ""):
                return a
        return None

    def _maybe_pick_banned_drink_discard(
        self,
        legal_actions: Sequence[LegalAction],
        scene: str,
        week: int,
    ) -> Optional[DecisionResult]:
        discard_act = self._find_banned_drink_discard_action(legal_actions)
        if discard_act is None:
            return None
        ctx = (scene, int(week or 0))
        if self._banned_drink_discard_ctx != ctx:
            self._banned_drink_discard_ctx = ctx
            self._banned_drink_discard_attempts = 0
        if self._banned_drink_discard_attempts >= 2:
            return None
        self._banned_drink_discard_attempts += 1
        self._expecting_drink_discard = True
        return DecisionResult(
            choice_id=discard_act.choice_id,
            reason=f"LocalAgent({scene}): Discarding banned 青汁 via {discard_act.description}",
            source="local_hybrid",
        )

    def _score_drink(
        self,
        *,
        drink_id: Optional[int],
        title: str,
        desc: str,
        target_effect: Optional[str],
        plan: str,
    ) -> int:
        """Score a candidate P-drink by HIF utility, extra-use/retrigger power, and idol recommended_effect."""
        row = self.db.get_drink_by_id(int(drink_id)) if drink_id else None
        if row is None and (title or desc):
            for dr in self.db.get_drinks_for_plan(plan):
                if dr.get("name") and str(dr["name"]) in f"{title} {desc}":
                    row = dr
                    drink_id = int(dr["id"])
                    break
        dname = str(row["name"]) if row else str(title or "")
        ddesc = str(row.get("description") or "") if row else ""
        dsl = f"{row.get('dsl_actions', '')} {row.get('dsl_effects', '')}".lower() if row else ""
        if self._is_banned_drink(drink_id, f"{dname} {title}", f"{ddesc} {desc} {dsl}"):
            return -100_000
        text_l = f"{dname} {title} {ddesc} {desc} {dsl}".lower()

        score = 100
        # Top-tier action-economy & retrigger drinks
        if (drink_id == 25) or ("特製ハツボシエキス" in text_l) or ("再発動" in text_l):
            score += 1350
        elif (drink_id in (18, 23)) or ("ホエイプロテイン" in text_l) or ("パワフル漢方" in text_l) or ("使用数追加" in text_l) or ("cardusesremaining" in text_l):
            score += 1200
        elif (drink_id == 21) or ("厳選初星ティー" in text_l):
            score += 950 if target_effect == "goodImpressionTurns" else 550
        elif (drink_id == 22) or ("初星黒酢" in text_l) or ("手札に移動" in text_l):
            score += 750
        elif (drink_id == 12) or ("ブーストエキス" in text_l):
            score += 700
        elif (drink_id == 14) or ("センブリソーダ" in text_l):
            score += 650

        # Effect alignment
        if target_effect:
            prim_kws = _EFFECT_KEYWORDS.get(target_effect, ())
            if any(kw in text_l for kw in prim_kws):
                score += 600
            sec_kws = _SECONDARY_EFFECT_KEYWORDS.get(target_effect, ())
            if any(kw in text_l for kw in sec_kws):
                score += 240

        if any(k in text_l for k in ("カードを引く", "手札", "コスト", "体力回復", "パラメータ上昇量")):
            score += 220
        if "ホットコーヒー" in text_l or "ミックススムージー" in text_l or "センブリ" in text_l:
            score += 260
        if "元気+" in text_l or "烏龍茶" in text_l or "初星水" in text_l:
            score += 140
        rarity = str(row.get("rarity") or "").upper() if row else ""
        score += {"SSR": 140, "SR": 90, "R": 40}.get(rarity, 0)
        if drink_id is not None:
            did_i = int(drink_id)
            if did_i in self._preferred_drink_ids:
                score += 2_500
            score += int((self.preferences.drink_score_overrides or {}).get(did_i, 0))
        return score

    @staticmethod
    def _extract_reward_slot_idx(act: LegalAction) -> Optional[int]:
        p = act.payload or {}
        if p.get("slot_idx") is not None:
            try:
                return int(p["slot_idx"])
            except Exception:
                pass
        nm = str(p.get("name") or p.get("title") or "")
        m = re.search(r"(\d+)", nm)
        return (int(m.group(1)) - 1) if m else None

    @classmethod
    def _infer_current_reward_slot(cls, preview_acts: List[LegalAction]) -> int:
        prev_slots = [s for s in (cls._extract_reward_slot_idx(p) for p in preview_acts) if s is not None]
        total_slots = max(len(preview_acts) + 1, (max(prev_slots) + 1) if prev_slots else 1)
        missing_slots = sorted(set(range(total_slots)) - set(prev_slots))
        return missing_slots[0] if missing_slots else 0

    def _score_battle_tutor_card(
        self,
        *,
        card_id: Optional[int],
        title: str,
        desc: str,
        obs: Optional[ScreenObservation],
        target_effect: Optional[str],
        guide: Optional[PIdolGuide],
        plan: str,
        p_idol_id: Optional[int],
    ) -> int:
        """Score a candidate in #22 初星黒酢 move-to-hand modal: #814 -> #162 -> #776 -> #812, strictly avoid trouble."""
        cid_i = int(card_id or 0)
        row = self.db.get_card_by_id(cid_i) if cid_i > 0 else None
        cname = str(row.get("name") or "") if row else ""
        ctype = str(row.get("type") or "").lower() if row else ""
        text_all = f"{cname} {title} {desc}".lower()
        if ctype == "trouble" or cid_i == 23 or any(tk in text_all for tk in ("眠気", "トラブル", "トラプル")):
            return -100_000
        base_id = int(row.get("base_id") or cid_i) if row else cid_i
        planner = getattr(self, "battle_planner", None)
        sim = getattr(planner, "_sim", None) if planner is not None else None
        planner_week = getattr(planner, "_last_week", None) if planner is not None else None
        obs_week = int(getattr(obs, "week", 0) or 0) if obs is not None else 0
        same_battle_sim = sim is not None and (planner_week is None or obs_week <= 0 or int(planner_week) == obs_week)
        double_armed = bool(same_battle_sim and getattr(sim, "double_card_effect_cards", 0) > 0)
        rem_turns = int(getattr(obs, "turns_remaining", 10) or 10) if obs is not None else 10
        if same_battle_sim and getattr(sim, "turns_remaining", None) is not None:
            rem_turns = min(rem_turns, int(sim.turns_remaining))
        late_window = rem_turns <= 3 or double_armed
        hand_bases: set[int] = set()
        if obs is not None and getattr(obs, "cards", None):
            for hc in obs.cards:
                hcid = int(getattr(hc, "card_id", 0) or 0)
                if hcid > 0:
                    hrow = self.db.get_card_by_id(hcid)
                    hand_bases.add(int(hrow.get("base_id") or hcid) if hrow else hcid)
        elif same_battle_sim and getattr(sim, "hand", None):
            for hc in sim.hand:
                cfg_h = getattr(hc, "cfg", None)
                if cfg_h is not None:
                    hb = int(getattr(cfg_h, "base_id", 0) or getattr(cfg_h, "card_id", 0) or 0)
                    if hb > 0:
                        hand_bases.add(hb)
        plan_l = str(plan or getattr(self.tracker, "plan", "") or (row or {}).get("plan") or "").strip().lower()
        is_sense_plan = plan_l in ("", "sense")
        is_logic_plan = plan_l == "logic"
        has_813_in_hand = is_sense_plan and (813 in hand_bases)
        has_162_in_hand = is_sense_plan and (162 in hand_bases)
        has_813_in_deck = is_sense_plan and (
            has_813_in_hand or bool(same_battle_sim and sim.has_unexhausted_card_family(813))
        )
        has_zeroing_in_hand = (
            any((b == 618 if is_sense_plan else (b == 152 if is_logic_plan else False)) for b in hand_bases)
            or bool(
                same_battle_sim
                and any(sim.plan_rules.is_resource_zeroing_finisher(hc.cfg) for hc in (getattr(sim, "hand", None) or ()))
            )
        )
        row_dsl = str(row.get("dsl_actions") or "") if row else ""
        if is_sense_plan:
            is_zeroing_fin = (
                base_id == 618
                or "世界一の勇気" in text_all
                or (
                    "score+=" in row_dsl
                    and bool(re.search(r"\b(goodConditionTurns|concentration)\s*=\s*0\b", row_dsl))
                )
            )
        elif is_logic_plan:
            is_zeroing_fin = (
                base_id == 152
                or "届いて" in text_all
                or (
                    "score+=" in row_dsl
                    and bool(re.search(r"\b(genki|goodImpressionTurns)\s*=\s*0\b", row_dsl))
                )
            )
        else:
            is_zeroing_fin = False
        if is_zeroing_fin:
            # Resource-zeroing solo finisher (Sense #618/#619, Logic #152/#153):
            # top priority ONLY on final turn when missing from hand; never tutor early!
            if rem_turns <= 1 and not has_zeroing_in_hand:
                return 22_000
            return -2_500
        if is_sense_plan:
            if base_id == 813 or "自然体の魅力" in text_all:
                if has_813_in_hand and not double_armed and not has_162_in_hand:
                    return 15_500 if late_window else 8_000
                if double_armed or has_162_in_hand:
                    return 22_000 if (late_window or double_armed) else 12_000
                return 20_000 if late_window else 9_000
            if base_id == 162 or "国民的アイドル" in text_all:
                if has_813_in_hand and not double_armed and not has_162_in_hand:
                    return 22_000 if late_window else 12_500
                if has_813_in_deck and not double_armed and not has_162_in_hand:
                    return 18_500 if late_window else 8_500
                if rem_turns <= 1 and has_zeroing_in_hand:
                    return -1_000
            if base_id == 775 or "話題沸騰" in text_all:
                return 14_000
            if base_id == 811 or "お姉さんの感覚" in text_all:
                return 11_500
        elif is_logic_plan:
            if (base_id == 777 or "びしっとキメ顔" in text_all) and rem_turns <= 5:
                return 18_000
        base_s = self._score_skill_card(
            card_id=cid_i if cid_i > 0 else None,
            title=title,
            desc=desc,
            target_effect=target_effect,
            guide=guide,
            plan=plan,
            for_removal=False,
            p_idol_id=p_idol_id,
            check_same_card=False,
        )
        if row:
            dsl_l = str(row.get("dsl_actions") or "").lower()
            rarity = str(row.get("rarity") or "").upper()
            if "cardusesremaining+=" in dsl_l or rarity == "SSR":
                base_s += 1_500
        return base_s

    def _is_non_exhaust_mental_card(
        self,
        card_id: Optional[int],
        title: str = "",
        desc: str = "",
    ) -> bool:
        """Return True for ordinary non-exhaust Mental (M) cards worth rerolling in early pickup/change."""
        row = self.db.get_card_by_id(int(card_id)) if card_id else None
        if row is None:
            return False
        if str(row.get("type") or "").lower() != "mental":
            return False
        text_all = f"{row.get('description') or ''} {title or ''} {desc or ''}"
        if row.get("limit_count") is not None or "レッスン中1回" in text_all:
            return False
        cid_i = int(card_id)
        if self._is_priority_card(cid_i):
            return False
        base_id = int(row.get("base_id") or cid_i)
        dsl = f"{row.get('dsl_actions') or ''} {row.get('dsl_effects') or ''}".lower()
        if (
            cid_i in _CORE_KEEP_CARD_IDS
            or base_id in _CORE_KEEP_BASE_IDS
            or row.get("is_signature")
            or "extra_turn" in dsl
            or "turnsremaining+=" in dsl
            or "追加ターン" in text_all
            or "cardusesremaining+=" in dsl
            or "使用数追加" in text_all
        ):
            return False
        return True

    def _avoided_card_fallback_key(
        self,
        card_id: Optional[int],
        title: str = "",
        desc: str = "",
    ) -> Tuple[int, int, int, int]:
        """Fallback priority when all reward candidates are excluded and no reroll/skip remains:
        1-use exhaustible (`limit_count == 1` / `レッスン中1回`) > Rarity (SSR > SR > R) > Type (M > A) > upgraded.
        """
        row = self.db.get_card_by_id(int(card_id)) if card_id else None
        text_all = f"{(row or {}).get('description') or ''} {title or ''} {desc or ''}"
        is_single_use_exhaust = (
            1
            if ((row is not None and row.get("limit_count") == 1) or ("レッスン中1回" in text_all))
            else 0
        )
        rarity = str((row or {}).get("rarity") or "").strip().upper()
        rarity_rank = {"LEGEND": 5, "L": 5, "SSR": 4, "SR": 3, "R": 2, "N": 1}.get(rarity, 0)
        ctype = str((row or {}).get("type") or "").strip().lower()
        type_rank = 2 if ctype == "mental" else (1 if ctype == "active" else 0)
        upgraded_rank = (
            1
            if ((row is not None and bool(row.get("upgraded"))) or ("+" in (title or "")))
            else 0
        )
        return (is_single_use_exhaust, rarity_rank, type_rank, upgraded_rank)

    def _clear_reward_scan_state(self) -> None:
        self._reward_preview_count = 0
        self._reward_seen_slots.clear()
        self._reward_slot_attempts.clear()
        self._reward_visited_grid_slots.clear()
        self._reward_grid_scores.clear()
        self._last_clicked_grid_slot = None
        self._reward_grid_request_scroll = None

    def _decide_card_reward(
        self,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation],
        p_idol_id: Optional[int],
    ) -> DecisionResult:
        self._has_entered_produce = True
        resolved_pid, target_eff, guide = self._resolve_idol_and_effect(obs, p_idol_id=p_idol_id)
        plan = (obs.plan if obs else "logic").lower()
        raw_txt = (obs.raw_text or "") if obs else ""

        # Handle Customize Menu sub-screen (実行する / menu tiles / 一覧に戻る)
        exec_cust = next((a for a in legal_actions if str((a.payload or {}).get("name") or "") == "カスタマイズ実行"), None)
        if exec_cust is not None:
            self._customize_done_count += 1
            self._customize_clicked_tiles.clear()
            return DecisionResult(
                choice_id=exec_cust.choice_id,
                reason=f"LocalAgent(customize): Executing -> {exec_cust.description}",
                source="local_hybrid",
            )
        cust_tiles = [a for a in legal_actions if (a.payload or {}).get("is_customize_tile")]
        in_cust_menu = bool(cust_tiles) or any(str((a.payload or {}).get("name") or "") == "一覧に戻る" for a in legal_actions)
        if in_cust_menu:
            if cust_tiles:
                unvisited_tiles = [
                    a for a in cust_tiles
                    if str((a.payload or {}).get("customize_tile_key") or (a.payload or {}).get("name") or "") not in self._customize_clicked_tiles
                ]
                if unvisited_tiles:
                    def _cust_tile_rank(a: LegalAction) -> int:
                        nm = str((a.payload or {}).get("customize_tile_key") or (a.payload or {}).get("name") or "")
                        if any(k in nm for k in ("使用数", "回数", "ターン")):
                            return 100
                        if any(k in nm for k in ("条件変更", "条件", "コスト", "消費")):
                            return 90
                        if any(k in nm for k in ("好印象", "好調", "集中", "全力")):
                            return 85 if "+" in nm or "＋" in nm else 80
                        if any(k in nm for k in ("パラメータ", "スコア", "持続", "ドロー", "引く")):
                            return 70
                        if "やる気" in nm:
                            return 50
                        if "元気" in nm:
                            return 30
                        return 40

                    chosen_tile = max(unvisited_tiles, key=_cust_tile_rank)
                    tkey = str((chosen_tile.payload or {}).get("customize_tile_key") or (chosen_tile.payload or {}).get("name") or "")
                    self._customize_clicked_tiles.add(tkey)
                    return DecisionResult(
                        choice_id=chosen_tile.choice_id,
                        reason=f"LocalAgent(customize): Trying option -> {chosen_tile.description}",
                        source="local_hybrid",
                    )
            cust_exit = next(
                (a for a in legal_actions if str((a.payload or {}).get("name") or "") in ("一覧に戻る", "終了", "戻る", "もどる", "キャンセル")),
                None,
            )
            if cust_exit is not None:
                self._customize_clicked_tiles.clear()
                return DecisionResult(
                    choice_id=cust_exit.choice_id,
                    reason=f"LocalAgent(customize): Exiting menu -> {cust_exit.description}",
                    source="local_hybrid",
                )

        # Handle Customize card-selection grid: pick a card before clicking カスタマイズする (only exit if exhausted/done)
        is_customize_grid = ("カスタマイズ" in raw_txt) or any(
            str((a.payload or {}).get("name") or "") in ("カスタマイズする", "もどる") for a in legal_actions
        )
        cust_limit_hit = (
            ("あと0回" in raw_txt)
            or ("あと0枚" in raw_txt)
            or ("あと0教" in raw_txt)
            or (self._customize_done_count >= 2)
            or (self._customize_cards_tried >= 3)
        )
        if is_customize_grid and cust_limit_hit:
            cust_back = next(
                (a for a in legal_actions if str((a.payload or {}).get("name") or "") in ("もどる", "終了", "戻る", "キャンセル")),
                None,
            )
            if cust_back is not None:
                self._customize_clicked_tiles.clear()
                return DecisionResult(
                    choice_id=cust_back.choice_id,
                    reason=f"LocalAgent(customize): Finished/limit reached -> {cust_back.description}",
                    source="local_hybrid",
                )

        sig = tuple((a.choice_id, str((a.payload or {}).get("name") or ""), (a.payload or {}).get("card_id"), (a.payload or {}).get("drink_id")) for a in legal_actions)
        is_preview_tok = lambda nm: any(tok in nm for tok in ("预览", "候选", "待选"))
        is_drink_opt = lambda a: bool(
            (a.payload or {}).get("drink_id") is not None
            or "饮料" in str((a.payload or {}).get("name") or "")
            or "饮料" in str((a.payload or {}).get("title") or "")
            or "ドリンク" in str((a.payload or {}).get("title") or "")
        )
        if sig != self._last_reward_sig:
            if not any(is_preview_tok(str((a.payload or {}).get("name") or "")) for a in legal_actions):
                self._clear_reward_scan_state()
                self._reward_reroll_attempts = 0
        self._last_reward_sig = sig

        # Handle drink cap or already-confirmed "決定"
        for a in legal_actions:
            nm = str((a.payload or {}).get("name") or "")
            if nm == "決定" or "已勾选『受け取らない』" in str((a.payload or {}).get("title") or ""):
                self._reward_seen_slots.clear()
                self._reward_slot_attempts.clear()
                return DecisionResult(choice_id=a.choice_id, reason="LocalAgent(reward): Confirming 決定", source="local_hybrid")

        maybe_discard = self._maybe_pick_banned_drink_discard(
            legal_actions,
            "card_reward",
            int(getattr(obs, "week", 0) or 0) if obs is not None else 0,
        )
        if maybe_discard is not None:
            self._reward_seen_slots.clear()
            self._reward_slot_attempts.clear()
            return maybe_discard

        has_drink_cap = any(str((a.payload or {}).get("name") or "").startswith("破棄候補_") for a in legal_actions)
        if has_drink_cap:
            dont_take = next((a for a in legal_actions if (a.payload or {}).get("name") == "受け取らない"), None)
            if dont_take is not None:
                self._reward_seen_slots.clear()
                self._reward_slot_attempts.clear()
                return DecisionResult(
                    choice_id=dont_take.choice_id,
                    reason="LocalAgent(reward): Drink tray full, choosing 受け取らない",
                    source="local_hybrid",
                )

        claim_act = next(
            (
                a for a in legal_actions
                if str((a.payload or {}).get("name") or "") in ("受け取る", "強化する", "削除する", "チェンジする", "カスタマイズする", "移動する", "次へ", "スキップ")
            ),
            None,
        )
        preview_acts = [
            a for a in legal_actions
            if is_preview_tok(str((a.payload or {}).get("name") or ""))
        ]
        cancel_act = next(
            (a for a in legal_actions if str((a.payload or {}).get("name") or "") in ("キャンセル", "戻る", "もどる")),
            None,
        )
        claim_name_peek = str((claim_act.payload or {}).get("name") or "") if claim_act is not None else ""
        has_grid_previews = any(
            tok in str((a.payload or {}).get("name") or "")
            for a in preview_acts
            for tok in ("待选卡", "换出候选卡")
        )
        has_3choice_previews = any(
            "候选卡" in str((a.payload or {}).get("name") or "")
            and "换出候选卡" not in str((a.payload or {}).get("name") or "")
            for a in preview_acts
        )
        is_removal = (
            claim_name_peek in ("削除する", "チェンジする")
            or (
                has_grid_previews
                and not has_3choice_previews
                and any(k in raw_txt for k in ("削除", "手放す", "チェンジ"))
            )
            or (
                not has_3choice_previews
                and "チェンジで獲得" not in raw_txt
                and claim_name_peek not in ("受け取る", "強化する", "カスタマイズする", "移動する", "次へ")
                and any(k in raw_txt for k in ("削除", "手放す", "チェンジ"))
            )
        )
        is_drink_flow = ("ドリンク" in raw_txt and "スキルカード" not in raw_txt) or any(is_drink_opt(a) for a in legal_actions)
        is_delete_removal = bool(
            claim_name_peek == "削除する"
            or (is_removal and any(k in raw_txt for k in ("削除", "手放す")) and "チェンジ" not in raw_txt)
        )
        is_change_swap = bool(is_removal and not is_delete_removal)
        is_change_pickup = (not is_removal) and (not is_drink_flow) and (
            ("チェンジで獲得" in raw_txt) or (claim_name_peek == "次へ")
        )
        ignore_change_upgrade = bool(is_change_pickup or is_change_swap)
        deck_has_trouble = self._deck_has_trouble_card()
        has_removable_unwanted: Optional[bool] = None
        if (is_removal or is_change_pickup) and not is_drink_flow:
            has_removable_unwanted = self._deck_has_removable_unwanted_card(
                target_effect=target_eff,
                guide=guide,
                plan=plan,
                p_idol_id=resolved_pid,
                ignore_upgrade=ignore_change_upgrade,
            )

        reroll_act = next(
            (
                a for a in legal_actions
                if str((a.payload or {}).get("name") or "") == "再抽選"
                and "ジュエル" not in str((a.payload or {}).get("title") or a.description)
            ),
            None,
        )
        avoid_reroll_ok = bool(
            reroll_act is not None
            and self._reward_reroll_attempts < 3
            and (not is_removal)
            and (not is_drink_flow)
        )
        early_reroll_ok = bool(
            self.preferences.reroll_early_non_exhaust_mental
            and reroll_act is not None
            and self._reward_reroll_attempts < 3
            and (not is_removal)
            and (not is_drink_flow)
            and int(getattr(obs, "week", 0) or 0) <= 6
        )

        # Direct selectable card options (not preview buttons)
        direct_card_opts = [
            a for a in legal_actions
            if (a.payload or {}).get("card_id") is not None and not is_preview_tok(str((a.payload or {}).get("name") or ""))
            and str((a.payload or {}).get("name") or "") not in ("受け取る", "強化する", "削除する", "チェンジする", "カスタマイズする", "移動する", "次へ")
        ]
        if direct_card_opts:
            scored_direct = [
                (
                    self._score_skill_card(
                        card_id=(a.payload or {}).get("card_id"),
                        title=str((a.payload or {}).get("title") or (a.payload or {}).get("name") or ""),
                        desc=str(a.description or ""),
                        target_effect=target_eff,
                        guide=guide,
                        plan=plan,
                        for_removal=is_removal,
                        p_idol_id=resolved_pid,
                        in_change_pickup=is_change_pickup,
                        has_removable_unwanted=has_removable_unwanted,
                        check_same_card=(not is_delete_removal),
                        ignore_upgrade=ignore_change_upgrade,
                    ),
                    a,
                )
                for a in direct_card_opts
            ]
            best_direct_score, best_direct = max(scored_direct, key=lambda pair: pair[0])
            best_cid = (best_direct.payload or {}).get("card_id")
            used_direct_avoid_fallback = False
            if (
                (not is_removal)
                and (not is_drink_flow)
                and claim_name_peek not in ("強化する", "移動する")
                and best_direct_score <= -3_000
            ):
                if avoid_reroll_ok:
                    self._reward_reroll_attempts += 1
                    self._clear_reward_scan_state()
                    return DecisionResult(
                        choice_id=reroll_act.choice_id,
                        reason=(
                            f"LocalAgent(reward): All direct reward candidates avoided/forbidden "
                            f"(best=#{best_cid}, score={best_direct_score}); rerolling via 再抽選"
                        ),
                        source="local_hybrid",
                    )
                if not is_change_pickup:
                    skip_or_reject = next(
                        (
                            a for a in legal_actions
                            if str((a.payload or {}).get("name") or "") in ("スキップ", "受け取らない", "もらわない", "見送る", "キャンセル")
                        ),
                        None,
                    )
                    if skip_or_reject is not None:
                        self._clear_reward_scan_state()
                        self._reward_reroll_attempts = 0
                        return DecisionResult(
                            choice_id=skip_or_reject.choice_id,
                            reason=f"LocalAgent(reward): All direct reward candidates avoided/forbidden (best score={best_direct_score}); skipping",
                            source="local_hybrid",
                        )
                best_direct = max(
                    direct_card_opts,
                    key=lambda a: self._avoided_card_fallback_key(
                        (a.payload or {}).get("card_id"),
                        str((a.payload or {}).get("title") or (a.payload or {}).get("name") or ""),
                        str(a.description or ""),
                    ),
                )
                best_cid = (best_direct.payload or {}).get("card_id")
                used_direct_avoid_fallback = True
            if (not used_direct_avoid_fallback) and early_reroll_ok and self._is_non_exhaust_mental_card(
                best_cid,
                str((best_direct.payload or {}).get("title") or (best_direct.payload or {}).get("name") or ""),
                str(best_direct.description or ""),
            ):
                self._reward_reroll_attempts += 1
                self._clear_reward_scan_state()
                return DecisionResult(
                    choice_id=reroll_act.choice_id,
                    reason=f"LocalAgent(reward): Early card pickup/change best candidate #{best_cid} is non-exhaust M-card; rerolling via 再抽選",
                    source="local_hybrid",
                )
            if is_change_pickup:
                self._came_from_change_pickup = True
                self._pending_change_gain_card_id = int(best_cid) if best_cid else None
                if best_cid:
                    brow = self.db.get_card_by_id(int(best_cid))
                    self._pending_change_gain_base_id = (
                        int(brow.get("base_id") or best_cid) if brow else int(best_cid)
                    )
                else:
                    self._pending_change_gain_base_id = None
            elif is_removal:
                self._came_from_change_pickup = False
                self._pending_change_gain_card_id = None
                self._pending_change_gain_base_id = None
            self._reward_reroll_attempts = 0
            direct_reason = (
                f"LocalAgent(reward): All avoided & no reroll -> fallback 1-use-exhaust>SSR>SR>R>M>A -> {best_direct.description}"
                if used_direct_avoid_fallback
                else f"LocalAgent(reward): Direct card pick ({target_eff}) -> {best_direct.description}"
            )
            return DecisionResult(
                choice_id=best_direct.choice_id,
                reason=direct_reason,
                source="local_hybrid",
            )

        # Direct or multi-slot Drink Reward evaluation: pick globally best drink across all visible slots!
        if is_drink_flow and preview_acts and all(is_drink_opt(p) for p in preview_acts):
            drink_skip_act = next(
                (
                    a for a in legal_actions
                    if str((a.payload or {}).get("name") or "") in ("受け取らない", "スキップ", "見送る", "キャンセル")
                ),
                None,
            )
            if claim_act is None:
                best_drink_prev = max(
                    preview_acts,
                    key=lambda p: self._score_drink(
                        drink_id=(p.payload or {}).get("drink_id"),
                        title=str((p.payload or {}).get("title") or (p.payload or {}).get("name") or ""),
                        desc=str(p.description or ""),
                        target_effect=target_eff,
                        plan=plan,
                    ),
                )
                best_prev_score = self._score_drink(
                    drink_id=(best_drink_prev.payload or {}).get("drink_id"),
                    title=str((best_drink_prev.payload or {}).get("title") or (best_drink_prev.payload or {}).get("name") or ""),
                    desc=str(best_drink_prev.description or ""),
                    target_effect=target_eff,
                    plan=plan,
                )
                if best_prev_score <= -50_000 and drink_skip_act is not None:
                    self._reward_seen_slots.clear()
                    self._reward_slot_attempts.clear()
                    return DecisionResult(
                        choice_id=drink_skip_act.choice_id,
                        reason="LocalAgent(drink reward): All visible candidate drinks banned (青汁); skipping",
                        source="local_hybrid",
                    )
                return DecisionResult(
                    choice_id=best_drink_prev.choice_id,
                    reason=f"LocalAgent(drink reward): Choosing best drink -> {best_drink_prev.description}",
                    source="local_hybrid",
                )
            cur_did = (claim_act.payload or {}).get("drink_id")
            cur_dscore = self._score_drink(
                drink_id=cur_did,
                title=str((claim_act.payload or {}).get("title") or ""),
                desc=str(claim_act.description or ""),
                target_effect=target_eff,
                plan=plan,
            )
            prev_slots = [s for s in (self._extract_reward_slot_idx(p) for p in preview_acts) if s is not None]
            total_slots = max(len(preview_acts) + 1, (max(prev_slots) + 1) if prev_slots else 1)
            missing_slots = sorted(set(range(total_slots)) - set(prev_slots))
            cur_slot = missing_slots[0] if missing_slots else 0
            self._reward_seen_slots[cur_slot] = (cur_dscore, cur_did, str((claim_act.payload or {}).get("title") or ""))

            # Real Gakumas drink pickers have at most 3 candidates; >=4 indicates a single-drink popup banner split.
            if len(preview_acts) >= 4:
                self._reward_seen_slots.clear()
                self._reward_slot_attempts.clear()
                if cur_dscore <= -50_000 and drink_skip_act is not None:
                    return DecisionResult(
                        choice_id=drink_skip_act.choice_id,
                        reason=f"LocalAgent(drink reward): Single drink banned (青汁); choosing {drink_skip_act.description}",
                        source="local_hybrid",
                    )
                return DecisionResult(
                    choice_id=claim_act.choice_id,
                    reason=f"LocalAgent(drink reward): Claiming single-drink banner directly (score={cur_dscore}) -> {claim_act.description}",
                    source="local_hybrid",
                )

            # If any alternative drink thumbnail lacked a template drink_id, inspect unseen slots first
            unseen_unknown = [
                p for p in preview_acts
                if (p.payload or {}).get("drink_id") is None
                and (s := self._extract_reward_slot_idx(p)) is not None
                and s not in self._reward_seen_slots
                and self._reward_slot_attempts.get(s, 0) < 2
            ]
            if unseen_unknown:
                nxt_d = unseen_unknown[0]
                dslot = self._extract_reward_slot_idx(nxt_d)
                if dslot is not None:
                    self._reward_slot_attempts[dslot] = self._reward_slot_attempts.get(dslot, 0) + 1
                return DecisionResult(
                    choice_id=nxt_d.choice_id,
                    reason=f"LocalAgent(drink reward): Inspecting unidentified drink slot -> {nxt_d.description}",
                    source="local_hybrid",
                )

            best_other_prev = max(
                preview_acts,
                key=lambda p: self._score_drink(
                    drink_id=(p.payload or {}).get("drink_id"),
                    title=str((p.payload or {}).get("title") or (p.payload or {}).get("name") or ""),
                    desc=str(p.description or ""),
                    target_effect=target_eff,
                    plan=plan,
                ),
            )
            best_other_score = self._score_drink(
                drink_id=(best_other_prev.payload or {}).get("drink_id"),
                title=str((best_other_prev.payload or {}).get("title") or (best_other_prev.payload or {}).get("name") or ""),
                desc=str(best_other_prev.description or ""),
                target_effect=target_eff,
                plan=plan,
            )
            best_other_slot = self._extract_reward_slot_idx(best_other_prev)
            best_seen_slot = max(self._reward_seen_slots, key=lambda k: self._reward_seen_slots[k][0])
            if (
                best_other_score > cur_dscore
                and best_other_slot not in self._reward_seen_slots
                and (best_other_slot is None or self._reward_slot_attempts.get(best_other_slot, 0) < 2)
            ):
                if best_other_slot is not None:
                    self._reward_slot_attempts[best_other_slot] = self._reward_slot_attempts.get(best_other_slot, 0) + 1
                return DecisionResult(
                    choice_id=best_other_prev.choice_id,
                    reason=f"LocalAgent(drink reward): Switching to higher-value drink ({best_other_score} > {cur_dscore}) -> {best_other_prev.description}",
                    source="local_hybrid",
                )
            if cur_slot != best_seen_slot and self._reward_slot_attempts.get(best_seen_slot, 0) < 3:
                best_prev = next(
                    (p for p in preview_acts if self._extract_reward_slot_idx(p) == best_seen_slot),
                    None,
                )
                if best_prev is not None:
                    self._reward_slot_attempts[best_seen_slot] = self._reward_slot_attempts.get(best_seen_slot, 0) + 1
                    return DecisionResult(
                        choice_id=best_prev.choice_id,
                        reason=f"LocalAgent(drink reward): Returning to best drink slot #{best_seen_slot + 1} (score={self._reward_seen_slots[best_seen_slot][0]})",
                        source="local_hybrid",
                    )
            best_seen_score = self._reward_seen_slots[best_seen_slot][0]
            self._reward_seen_slots.clear()
            self._reward_slot_attempts.clear()
            if cur_dscore <= -50_000 and best_seen_score <= -50_000 and drink_skip_act is not None:
                return DecisionResult(
                    choice_id=drink_skip_act.choice_id,
                    reason=f"LocalAgent(drink reward): Refusing banned drink (青汁); choosing {drink_skip_act.description}",
                    source="local_hybrid",
                )
            return DecisionResult(
                choice_id=claim_act.choice_id,
                reason=f"LocalAgent(drink reward): Claiming best drink (score={cur_dscore}) -> {claim_act.description}",
                source="local_hybrid",
            )

        if claim_act is not None:
            claim_name = str((claim_act.payload or {}).get("name") or "")
            claim_cid = (claim_act.payload or {}).get("card_id")
            claim_title = str((claim_act.payload or {}).get("title") or "")

            if claim_name == "カスタマイズする":
                self._customize_cards_tried += 1
                self._customize_clicked_tiles.clear()
                return DecisionResult(
                    choice_id=claim_act.choice_id,
                    reason=f"LocalAgent(customize): Opening customize menu -> {claim_act.description}",
                    source="local_hybrid",
                )

            if is_removal and preview_acts:
                claim_slot = (claim_act.payload or {}).get("slot_idx")
                cur_slot = (
                    int(claim_slot)
                    if claim_slot is not None
                    else (
                        int(self._last_clicked_grid_slot)
                        if self._last_clicked_grid_slot is not None
                        else self._infer_current_reward_slot(preview_acts)
                    )
                )
                claim_is_custom = bool((claim_act.payload or {}).get("is_customized"))
                cur_score = (
                    self._score_skill_card(
                        card_id=int(claim_cid),
                        title=claim_title,
                        desc=str(claim_act.description or ""),
                        target_effect=target_eff,
                        guide=guide,
                        plan=plan,
                        for_removal=True,
                        p_idol_id=resolved_pid,
                        has_removable_unwanted=has_removable_unwanted,
                        check_same_card=(not is_delete_removal),
                        ignore_upgrade=is_change_swap,
                        is_customized=claim_is_custom,
                    )
                    if claim_cid
                    else (-18_000 if claim_is_custom else -500)
                )
                self._reward_grid_scores[cur_slot] = cur_score
                self._reward_visited_grid_slots.add(cur_slot)
                unvisited_grid = [
                    p for p in preview_acts
                    if (
                        (s := self._extract_reward_slot_idx(p)) is None
                        or (s != cur_slot and s not in self._reward_visited_grid_slots)
                    )
                ]
                # When deleting in shop with a Trouble (T) card in deck (or on first shop probe where W1-W5
                # events may have injected an untracked #23 眠気), keep scanning until we locate the T card (>=4000).
                # Similarly, during チェンジ/削除, if the deck contains user-configured avoid_card_ids or forbidden cards,
                # keep scanning the visible deck grid so we ALWAYS swap/delete those avoided cards first!
                deck_has_avoid_or_forbidden = False
                if self.tracker is not None and self.tracker.produce_deck:
                    for _dcid in self.tracker.produce_deck:
                        _drow = self.db.get_card_by_id(int(_dcid))
                        _dbase = int(_drow.get("base_id") or _dcid) if _drow else int(_dcid)
                        if (
                            self._is_avoid_card(int(_dcid), base_id=_dbase)
                            or self._is_forbidden_card(
                                card_id=int(_dcid),
                                base_id=_dbase,
                                cname=str((_drow or {}).get("name") or ""),
                                p_idol_id=resolved_pid,
                                guide=guide,
                            )
                        ):
                            deck_has_avoid_or_forbidden = True
                            break
                prefer_trouble_scan = bool(
                    is_delete_removal
                    and (deck_has_trouble or (self._in_shop_visit and not self._first_shop_delete_done))
                    and cur_score < 4000
                    and unvisited_grid
                    and self._reward_preview_count < 16
                )
                prefer_avoid_scan = bool(
                    is_removal
                    and deck_has_avoid_or_forbidden
                    and cur_score < 2500
                    and unvisited_grid
                    and self._reward_preview_count < 14
                )
                prefer_high_removal_scan = prefer_trouble_scan or prefer_avoid_scan
                if (
                    cur_score >= 4000
                    or (cur_score >= 2500 and not prefer_trouble_scan)
                    or (cur_score > 0 and not prefer_high_removal_scan)
                ):
                    if is_delete_removal and self._shop_visit_count <= 1:
                        self._first_shop_delete_done = True
                    self._came_from_change_pickup = False
                    self._pending_change_gain_card_id = None
                    self._pending_change_gain_base_id = None
                    self._clear_reward_scan_state()
                    return DecisionResult(
                        choice_id=claim_act.choice_id,
                        reason=f"LocalAgent(removal): Confirming weak/trouble card (score={cur_score}) -> {claim_act.description}",
                        source="local_hybrid",
                    )
                max_grid_scans = 14 if prefer_high_removal_scan else (12 if cur_score <= -1500 else 6)
                non_custom_unvisited = [
                    p for p in unvisited_grid
                    if not (p.payload or {}).get("is_customized")
                ]
                scan_pool = non_custom_unvisited or unvisited_grid
                if scan_pool and self._reward_preview_count < max_grid_scans:
                    self._reward_preview_count += 1
                    nxt = max(
                        scan_pool,
                        key=lambda a: (
                            -1 if (a.payload or {}).get("is_customized") else (
                                1 if ("暗灰" in a.description or "基本" in a.description or "暗灰" in str((a.payload or {}).get("title") or "")) else 0
                            ),
                            self._extract_reward_slot_idx(a) or 0,
                        ),
                    )
                    nslot = self._extract_reward_slot_idx(nxt)
                    if nslot is not None:
                        self._reward_visited_grid_slots.add(nslot)
                        self._last_clicked_grid_slot = nslot
                    return DecisionResult(
                        choice_id=nxt.choice_id,
                        reason=f"LocalAgent(removal): Inspecting deck slot (cur_score={cur_score}) -> {nxt.description}",
                        source="local_hybrid",
                    )
                if (
                    self._reward_grid_scores
                    and max(self._reward_grid_scores.values()) < 0
                    and getattr(self, "_reward_grid_can_flip_scroll", False)
                ):
                    self._reward_grid_request_scroll = "flip"
                    return DecisionResult(
                        choice_id=claim_act.choice_id,
                        reason="LocalAgent(removal): Requesting scroll to opposite end of deck to find weaker card",
                        source="local_hybrid",
                    )
                if claim_is_custom and claim_name == "チェンジする":
                    # On mandatory H.I.F チェンジ, cancelling only loops back to Page 1 (which has no skip);
                    # immediately switch to the best non-customized deck slot instead!
                    non_cust_alts = [
                        p for p in preview_acts
                        if not (p.payload or {}).get("is_customized")
                        and self._extract_reward_slot_idx(p) != cur_slot
                    ]
                    if non_cust_alts:
                        best_nc = max(
                            non_cust_alts,
                            key=lambda p: self._reward_grid_scores.get(self._extract_reward_slot_idx(p) or -1, -9_999),
                        )
                        nc_slot = self._extract_reward_slot_idx(best_nc)
                        if nc_slot is not None:
                            self._last_clicked_grid_slot = nc_slot
                        return DecisionResult(
                            choice_id=best_nc.choice_id,
                            reason=f"LocalAgent(removal): Protecting customized slot #{cur_slot + 1}; switching to non-customized slot -> {best_nc.description}",
                            source="local_hybrid",
                        )
                if (
                    cancel_act is not None
                    and (
                        (
                            claim_name == "チェンジする"
                            and (not self._came_from_change_pickup)
                            and self._reward_cancel_attempts == 0
                            and cur_score < -250
                        )
                        or (
                            claim_name == "削除する"
                            and (
                                (cur_score < -250 and (not self._came_from_change_pickup) and self._reward_cancel_attempts == 0)
                                or (claim_is_custom and self._reward_cancel_attempts < 2)
                            )
                        )
                    )
                ):
                    self._reward_cancel_attempts += 1
                    self._pending_change_gain_card_id = None
                    self._pending_change_gain_base_id = None
                    self._clear_reward_scan_state()
                    return DecisionResult(
                        choice_id=cancel_act.choice_id,
                        reason=f"LocalAgent(removal): Cancelling {claim_name} to protect {'customized' if claim_is_custom else 'strong'} card (score={cur_score})",
                        source="local_hybrid",
                    )
                if self._reward_grid_scores:
                    best_grid_slot = max(self._reward_grid_scores, key=lambda k: self._reward_grid_scores[k])
                    if best_grid_slot != cur_slot and self._reward_grid_scores[best_grid_slot] > cur_score:
                        best_prev = next(
                            (p for p in preview_acts if self._extract_reward_slot_idx(p) == best_grid_slot),
                            None,
                        )
                        if best_prev is not None:
                            self._last_clicked_grid_slot = best_grid_slot
                            return DecisionResult(
                                choice_id=best_prev.choice_id,
                                reason=f"LocalAgent(removal): Returning to least-costly swap slot #{best_grid_slot + 1} (score={self._reward_grid_scores[best_grid_slot]} > {cur_score})",
                                source="local_hybrid",
                            )
                if is_delete_removal and self._shop_visit_count <= 1:
                    self._first_shop_delete_done = True
                self._came_from_change_pickup = False
                self._pending_change_gain_card_id = None
                self._pending_change_gain_base_id = None
                self._clear_reward_scan_state()
                return DecisionResult(
                    choice_id=claim_act.choice_id,
                    reason=f"LocalAgent(reward): Claiming/confirming {claim_name} ({claim_act.description})",
                    source="local_hybrid",
                )

            # Skill card 3-choice reward / Upgrade / #22 Tutor: inspect candidates cleanly
            if preview_acts and not is_removal:
                parsed_cid = int(claim_cid) if claim_cid else None
                in_chg_pick = is_change_pickup and (claim_name not in ("強化する", "移動する"))
                if claim_name == "移動する":
                    cur_score = self._score_battle_tutor_card(
                        card_id=parsed_cid,
                        title=claim_title,
                        desc=str(claim_act.description or ""),
                        obs=obs,
                        target_effect=target_eff,
                        guide=guide,
                        plan=plan,
                        p_idol_id=resolved_pid,
                    )
                else:
                    cur_score = self._score_skill_card(
                        card_id=parsed_cid,
                        title=claim_title,
                        desc=str(claim_act.description or ""),
                        target_effect=target_eff,
                        guide=guide,
                        plan=plan,
                        for_removal=False,
                        p_idol_id=resolved_pid,
                        in_change_pickup=in_chg_pick,
                        has_removable_unwanted=has_removable_unwanted if in_chg_pick else None,
                        ignore_upgrade=in_chg_pick,
                    )
                claim_slot = (claim_act.payload or {}).get("slot_idx")
                prev_slots = [s for s in (self._extract_reward_slot_idx(p) for p in preview_acts) if s is not None]
                is_full_grid_previews = has_grid_previews or (
                    len(prev_slots) >= 4 and set(prev_slots) == set(range(len(prev_slots)))
                )
                if claim_slot is not None:
                    cur_slot = int(claim_slot)
                elif is_full_grid_previews:
                    cur_slot = int(self._last_clicked_grid_slot) if self._last_clicked_grid_slot is not None else 0
                else:
                    cur_slot = self._infer_current_reward_slot(preview_acts)
                self._reward_seen_slots[cur_slot] = (cur_score, parsed_cid, claim_title)

                if claim_name == "移動する" and cur_score >= 18_000:
                    self._clear_reward_scan_state()
                    return DecisionResult(
                        choice_id=claim_act.choice_id,
                        reason=f"LocalAgent(tutor): Found priority combo card (score={cur_score}) -> {claim_act.description}",
                        source="local_hybrid",
                    )

                unvisited_prevs = [
                    p for p in preview_acts
                    if (s := self._extract_reward_slot_idx(p)) is not None
                    and s not in self._reward_seen_slots
                    and self._reward_slot_attempts.get(s, 0) < 2
                ]
                if claim_name == "強化する" and len(self._reward_seen_slots) >= 4:
                    unvisited_prevs = []
                elif claim_name == "移動する":
                    max_seen_score = max(v[0] for v in self._reward_seen_slots.values())
                    max_tutor_scans = 16 if max_seen_score < 18_000 else 8
                    if len(self._reward_seen_slots) >= max_tutor_scans:
                        unvisited_prevs = []
                if unvisited_prevs:
                    nxt = unvisited_prevs[0]
                    nslot = self._extract_reward_slot_idx(nxt)
                    if nslot is not None:
                        self._reward_slot_attempts[nslot] = self._reward_slot_attempts.get(nslot, 0) + 1
                        self._last_clicked_grid_slot = nslot
                    return DecisionResult(
                        choice_id=nxt.choice_id,
                        reason=f"LocalAgent(reward): Scanning unseen card slot (cur={cur_slot}, score={cur_score}, seen={sorted(self._reward_seen_slots)})",
                        source="local_hybrid",
                    )

                used_avoid_fallback = False
                if self._reward_seen_slots:
                    best_slot = max(self._reward_seen_slots, key=lambda k: self._reward_seen_slots[k][0])
                    best_seen_score, best_seen_cid, best_seen_title = self._reward_seen_slots[best_slot]
                    if claim_name not in ("強化する", "移動する") and best_seen_score <= -3_000:
                        if avoid_reroll_ok and claim_name in ("受け取る", "次へ"):
                            self._reward_reroll_attempts += 1
                            self._clear_reward_scan_state()
                            return DecisionResult(
                                choice_id=reroll_act.choice_id,
                                reason=(
                                    f"LocalAgent(reward): All scanned cards avoided/forbidden "
                                    f"(best score={best_seen_score}); rerolling via 再抽選"
                                ),
                                source="local_hybrid",
                            )
                        if not in_chg_pick:
                            skip_or_reject = next(
                                (
                                    a for a in legal_actions
                                    if a is not claim_act
                                    and str((a.payload or {}).get("name") or "") in ("スキップ", "受け取らない", "もらわない", "見送る", "キャンセル")
                                ),
                                None,
                            )
                            if skip_or_reject is not None:
                                self._clear_reward_scan_state()
                                self._reward_reroll_attempts = 0
                                return DecisionResult(
                                    choice_id=skip_or_reject.choice_id,
                                    reason=f"LocalAgent(reward): All scanned cards harmful/forbidden (best score={best_seen_score}); skipping",
                                    source="local_hybrid",
                                )
                        best_slot = max(
                            self._reward_seen_slots,
                            key=lambda k: self._avoided_card_fallback_key(
                                self._reward_seen_slots[k][1],
                                self._reward_seen_slots[k][2],
                            ),
                        )
                        best_seen_score, best_seen_cid, best_seen_title = self._reward_seen_slots[best_slot]
                        used_avoid_fallback = True
                    if (
                        (not used_avoid_fallback)
                        and early_reroll_ok
                        and claim_name in ("受け取る", "次へ")
                        and self._is_non_exhaust_mental_card(best_seen_cid, best_seen_title, "")
                    ):
                        self._reward_reroll_attempts += 1
                        self._clear_reward_scan_state()
                        return DecisionResult(
                            choice_id=reroll_act.choice_id,
                            reason=(
                                f"LocalAgent(reward): Early pickup/change best candidate #{best_seen_cid} "
                                f"({best_seen_title}) is non-exhaust M-card; rerolling via 再抽選"
                            ),
                            source="local_hybrid",
                        )
                    if cur_slot != best_slot and self._reward_slot_attempts.get(best_slot, 0) < 3:
                        best_prev = next(
                            (p for p in preview_acts if self._extract_reward_slot_idx(p) == best_slot),
                            None,
                        )
                        if best_prev is not None:
                            self._reward_slot_attempts[best_slot] = self._reward_slot_attempts.get(best_slot, 0) + 1
                            self._last_clicked_grid_slot = best_slot
                            sw_reason = (
                                (
                                    f"LocalAgent(reward): All avoided & no reroll -> switching to fallback slot #{best_slot + 1} "
                                    f"(#{best_seen_cid} {best_seen_title}, 1-use-exhaust>SSR>SR>R>M>A)"
                                )
                                if used_avoid_fallback
                                else (
                                    f"LocalAgent(reward): Switching to best card slot #{best_slot + 1} "
                                    f"(score={self._reward_seen_slots[best_slot][0]} >= {cur_score})"
                                )
                            )
                            return DecisionResult(
                                choice_id=best_prev.choice_id,
                                reason=sw_reason,
                                source="local_hybrid",
                            )
                if (not used_avoid_fallback) and (not in_chg_pick) and claim_name not in ("強化する", "移動する") and cur_score <= -3_000:
                    if avoid_reroll_ok and claim_name in ("受け取る", "次へ"):
                        self._reward_reroll_attempts += 1
                        self._clear_reward_scan_state()
                        return DecisionResult(
                            choice_id=reroll_act.choice_id,
                            reason=(
                                f"LocalAgent(reward): All scanned cards avoided/forbidden "
                                f"(best score={cur_score}); rerolling via 再抽選"
                            ),
                            source="local_hybrid",
                        )
                    skip_or_reject = next(
                        (
                            a for a in legal_actions
                            if a is not claim_act
                            and str((a.payload or {}).get("name") or "") in ("スキップ", "受け取らない", "もらわない", "見送る", "キャンセル")
                        ),
                        None,
                    )
                    if skip_or_reject is not None:
                        self._clear_reward_scan_state()
                        self._reward_reroll_attempts = 0
                        return DecisionResult(
                            choice_id=skip_or_reject.choice_id,
                            reason=f"LocalAgent(reward): All scanned cards harmful/forbidden (best score={cur_score}); skipping",
                            source="local_hybrid",
                        )
                if in_chg_pick or ("チェンジで獲得" in raw_txt) or (claim_name == "次へ"):
                    self._came_from_change_pickup = True
                    self._pending_change_gain_card_id = parsed_cid
                    if parsed_cid:
                        crow_pg = self.db.get_card_by_id(parsed_cid)
                        self._pending_change_gain_base_id = (
                            int(crow_pg.get("base_id") or parsed_cid) if crow_pg else parsed_cid
                        )
                    else:
                        self._pending_change_gain_base_id = None
                self._clear_reward_scan_state()
                self._reward_reroll_attempts = 0
                claim_reason = (
                    f"LocalAgent(reward): All avoided & no reroll -> fallback 1-use-exhaust>SSR>SR>R>M>A (score={cur_score}) -> {claim_act.description}"
                    if used_avoid_fallback
                    else f"LocalAgent(reward): Claiming best evaluated card (score={cur_score}) -> {claim_act.description}"
                )
                return DecisionResult(
                    choice_id=claim_act.choice_id,
                    reason=claim_reason,
                    source="local_hybrid",
                )

            claim_did = (claim_act.payload or {}).get("drink_id")
            if (not is_removal) and claim_name == "受け取る" and self._is_banned_drink(
                claim_did,
                claim_title,
                str(claim_act.description or ""),
            ):
                skip_banned_drink = next(
                    (
                        a for a in legal_actions
                        if a is not claim_act
                        and str((a.payload or {}).get("name") or "") in ("受け取らない", "スキップ", "見送る", "キャンセル")
                    ),
                    None,
                )
                if skip_banned_drink is not None:
                    self._clear_reward_scan_state()
                    self._reward_reroll_attempts = 0
                    return DecisionResult(
                        choice_id=skip_banned_drink.choice_id,
                        reason=f"LocalAgent(drink reward): Refusing banned drink (青汁); choosing {skip_banned_drink.description}",
                        source="local_hybrid",
                    )

            if (not is_removal) and (not is_change_pickup) and claim_cid and claim_name == "受け取る":
                single_score = self._score_skill_card(
                    card_id=int(claim_cid),
                    title=claim_title,
                    desc=str(claim_act.description or ""),
                    target_effect=target_eff,
                    guide=guide,
                    plan=plan,
                    for_removal=False,
                    p_idol_id=resolved_pid,
                )
                if single_score <= -3_000:
                    if avoid_reroll_ok and claim_name in ("受け取る", "次へ"):
                        self._reward_reroll_attempts += 1
                        self._clear_reward_scan_state()
                        return DecisionResult(
                            choice_id=reroll_act.choice_id,
                            reason=(
                                f"LocalAgent(reward): Single card avoided/forbidden "
                                f"(score={single_score}); rerolling via 再抽選"
                            ),
                            source="local_hybrid",
                        )
                    skip_or_reject = next(
                        (
                            a for a in legal_actions
                            if a is not claim_act
                            and str((a.payload or {}).get("name") or "") in ("スキップ", "受け取らない", "もらわない", "見送る", "キャンセル")
                        ),
                        None,
                    )
                    if skip_or_reject is not None:
                        self._clear_reward_scan_state()
                        self._reward_reroll_attempts = 0
                        return DecisionResult(
                            choice_id=skip_or_reject.choice_id,
                            reason=f"LocalAgent(reward): Single card harmful/forbidden (score={single_score}); skipping",
                            source="local_hybrid",
                        )

            if (
                early_reroll_ok
                and claim_name in ("受け取る", "次へ")
                and claim_cid
                and self._is_non_exhaust_mental_card(int(claim_cid), claim_title, str(claim_act.description or ""))
            ):
                self._reward_reroll_attempts += 1
                self._clear_reward_scan_state()
                return DecisionResult(
                    choice_id=reroll_act.choice_id,
                    reason=(
                        f"LocalAgent(reward): Early pickup/change card #{claim_cid} "
                        f"({claim_title}) is non-exhaust M-card; rerolling via 再抽選"
                    ),
                    source="local_hybrid",
                )

            if is_removal:
                if is_delete_removal and self._shop_visit_count <= 1:
                    self._first_shop_delete_done = True
                self._came_from_change_pickup = False
                self._pending_change_gain_card_id = None
                self._pending_change_gain_base_id = None
            elif is_change_pickup or ("チェンジで獲得" in raw_txt) or (claim_name == "次へ"):
                self._came_from_change_pickup = True
                parsed_cid = int(claim_cid) if claim_cid else None
                self._pending_change_gain_card_id = parsed_cid
                if parsed_cid:
                    crow_pg = self.db.get_card_by_id(parsed_cid)
                    self._pending_change_gain_base_id = (
                        int(crow_pg.get("base_id") or parsed_cid) if crow_pg else parsed_cid
                    )
                else:
                    self._pending_change_gain_base_id = None
            self._clear_reward_scan_state()
            self._reward_reroll_attempts = 0
            return DecisionResult(
                choice_id=claim_act.choice_id,
                reason=f"LocalAgent(reward): Claiming/confirming {claim_name} ({claim_act.description})",
                source="local_hybrid",
            )

        if preview_acts:
            unvisited_prev = [
                p
                for p in preview_acts
                if (self._extract_reward_slot_idx(p) is None)
                or (self._extract_reward_slot_idx(p) not in self._reward_visited_grid_slots)
            ]
            if not unvisited_prev:
                if (
                    is_removal
                    and cancel_act is not None
                    and (not self._came_from_change_pickup)
                    and self._reward_cancel_attempts == 0
                ):
                    self._reward_cancel_attempts += 1
                    self._clear_reward_scan_state()
                    return DecisionResult(
                        choice_id=cancel_act.choice_id,
                        reason="LocalAgent(reward): All visible slots failed to select; cancelling once",
                        source="local_hybrid",
                    )
                slot_idx = self._reward_preview_count % len(preview_acts)
                self._reward_preview_count += 1
                chosen_prev = preview_acts[slot_idx]
            else:
                pool_acts = unvisited_prev
                if is_removal and any(("暗灰" in a.description or "基本" in a.description or "暗灰" in str((a.payload or {}).get("title") or "")) for a in pool_acts):
                    chosen_prev = max(
                        pool_acts,
                        key=lambda a: 500 if ("暗灰" in a.description or "基本" in a.description or "暗灰" in str((a.payload or {}).get("title") or "")) else 0,
                    )
                elif is_drink_flow:
                    chosen_prev = max(
                        pool_acts,
                        key=lambda p: self._score_drink(
                            drink_id=(p.payload or {}).get("drink_id"),
                            title=str((p.payload or {}).get("title") or (p.payload or {}).get("name") or ""),
                            desc=str(p.description or ""),
                            target_effect=target_eff,
                            plan=plan,
                        ),
                    )
                else:
                    slot_idx = self._reward_preview_count % len(pool_acts)
                    self._reward_preview_count += 1
                    chosen_prev = pool_acts[slot_idx]
            chosen_slot = self._extract_reward_slot_idx(chosen_prev)
            if chosen_slot is not None:
                self._reward_visited_grid_slots.add(chosen_slot)
                self._last_clicked_grid_slot = chosen_slot
            return DecisionResult(
                choice_id=chosen_prev.choice_id,
                reason=f"LocalAgent(reward): Selecting slot {chosen_prev.description}",
                source="local_hybrid",
            )

        # Fallback: score all options
        best = max(
            legal_actions,
            key=lambda a: (
                -100_000
                if self._is_banned_drink(
                    (a.payload or {}).get("drink_id"),
                    str((a.payload or {}).get("title") or (a.payload or {}).get("name") or ""),
                    str(a.description or ""),
                )
                else (
                    -10_000
                    if any(bad in str((a.payload or {}).get("name") or a.description) for bad in ("ジュエル", "再抽選"))
                    else self._score_skill_card(
                        card_id=(a.payload or {}).get("card_id"),
                        title=str((a.payload or {}).get("title") or (a.payload or {}).get("name") or ""),
                        desc=str(a.description or ""),
                        target_effect=target_eff,
                        guide=guide,
                        plan=plan,
                        for_removal=is_removal,
                        ignore_upgrade=ignore_change_upgrade,
                    )
                )
            ),
        )
        return DecisionResult(
            choice_id=best.choice_id,
            reason=f"LocalAgent(reward): Fallback pick -> {best.description}",
            source="local_hybrid",
        )

    def _deck_has_card_family(self, card_id: Optional[int]) -> bool:
        """Return True if tracker.produce_deck already owns any copy (base or +) of `card_id`."""
        if not card_id or self.tracker is None:
            return False
        deck_ids = list(getattr(self.tracker, "produce_deck", []) or [])
        if not deck_ids:
            return False
        cid_i = int(card_id)
        row = self.db.get_card_by_id(cid_i)
        target_base = int(row.get("base_id") or cid_i) if row else cid_i
        target_upg = int(row["upgraded_id"]) if (row and row.get("upgraded_id")) else 0
        for dcid in deck_ids:
            di = int(dcid)
            if di == cid_i or di == target_base or (target_upg and di == target_upg):
                return True
            drow = self.db.get_card_by_id(di)
            if drow and int(drow.get("base_id") or di) == target_base:
                return True
        return False

    def _is_shop_buyable_priority_card(self, act: LegalAction, p_pts: int) -> bool:
        """Return True iff `act` previews or exchanges a user `priority_card_ids` card not yet in deck and affordable."""
        if not self._priority_card_ids and not self._priority_card_base_ids:
            return False
        p = act.payload or {}
        cid = p.get("card_id")
        if cid is None:
            return False
        try:
            cid_i = int(cid)
        except (TypeError, ValueError):
            return False
        if cid_i <= 0:
            return False
        row = self.db.get_card_by_id(cid_i)
        base_id = int(row.get("base_id") or cid_i) if row else cid_i
        if not self._is_priority_card(cid_i, base_id=base_id):
            return False
        if cid_i in self._shop_attempted_card_ids or base_id in self._shop_attempted_card_ids:
            return False
        if self._deck_has_card_family(cid_i):
            return False
        price = int(p.get("price") or 0)
        if price > p_pts:
            return False
        return True

    @staticmethod
    def _is_shop_skill_card_action(act: LegalAction) -> bool:
        """Strictly return True if selecting/clicking this shop action would inspect or buy a new skill card."""
        p = act.payload or {}
        slot_key = str(p.get("shop_slot_key") or "")
        nm = str(p.get("name") or "")
        title = str(p.get("title") or "")
        if slot_key.startswith("スキルカード:") or nm.startswith("スキルカード"):
            return True
        if nm in ("交換する", "チェンジする", "チェンジ") and (p.get("card_id") is not None or (p.get("drink_id") is None and "スキルカード" in title)):
            return True
        return False

    def _is_shop_banned_drink_action(self, act: LegalAction) -> bool:
        """Return True if a shop preview or exchange option targets any user-banned drink."""
        p = act.payload or {}
        nm = str(p.get("name") or "")
        if nm.startswith("破棄候補_"):
            return False
        slot_key = str(p.get("shop_slot_key") or "")
        is_drink_shop_opt = (
            p.get("drink_id") is not None
            or nm.startswith("Pドリンク")
            or slot_key.startswith("Pドリンク:")
            or "ドリンク" in str(p.get("title") or "")
        )
        if not is_drink_shop_opt:
            return False
        return self._is_banned_drink(
            p.get("drink_id"),
            str(p.get("title") or nm),
            str(act.description or ""),
        )

    def _decide_shop(
        self,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation],
        p_idol_id: Optional[int] = None,
    ) -> DecisionResult:
        """
        Shop policy:
        STRICTLY NEVER buy or add new skill cards (`_is_shop_skill_card_action`) or banned drinks (青汁).
        Allowed: upgrade (`強化`/`カスタマイズ`), delete weak/trouble card (`削除`),
        1-for-1 trade (`セレクトチェンジ`) when trouble card exists, affordable non-banned P-drinks
        when tray isn't full, else exit (`終了`/`次へ`).
        """
        self._has_entered_produce = True
        if not self._in_shop_visit:
            self._in_shop_visit = True
            self._shop_visit_count += 1
        p_pts = int(getattr(obs, "p_points", 0) or 0) if obs is not None else 0
        wk = int(getattr(obs, "week", 0) or 0) if obs is not None else 0
        if wk > 0 and self._shop_last_week is not None and wk != self._shop_last_week:
            self._shop_noop_features.clear()
            self._shop_clicked_slots.clear()
            self._shop_refresh_noop_streak = 0
            self._shop_refresh_since_purchase = 0
            self._shop_last_clicked_feature = None
        if wk > 0:
            self._shop_last_week = wk
        maybe_discard = self._maybe_pick_banned_drink_discard(legal_actions, "shop", wk)
        if maybe_discard is not None:
            self._shop_card_popup_guard = False
            self._shop_expecting_drink_confirm = False
            return maybe_discard
        tot_wk = int(getattr(obs, "total_weeks", 0) or 0) if obs is not None else 0
        scen = str(getattr(obs, "scenario_id", "") or "") if obs is not None else ""
        is_final_shop = bool(
            (scen in ("hif_honsen", "hif") and wk >= 8)
            or (tot_wk >= 2 and wk >= tot_wk - 1)
        )
        is_first_shop = (not is_final_shop) and self._shop_visit_count <= 1
        if (
            self._shop_last_clicked_feature in ("強化", "削除", "カスタマイズ", "セレクトチェンジ")
            and self._shop_last_points is not None
            and p_pts == self._shop_last_points
        ):
            self._shop_noop_features.add(self._shop_last_clicked_feature)
        elif self._shop_last_clicked_feature == "リフレッシュ" and self._shop_last_points is not None:
            if p_pts < self._shop_last_points:
                # Refresh succeeded and actually spent P-points -> unlock refreshed shop actions
                self._shop_refresh_noop_streak = 0
                self._shop_clicked_slots.clear()
                self._shop_noop_features.clear()
                self._shop_attempted_drink_ids.clear()
                self._shop_attempted_card_ids.clear()
                self._shop_expecting_priority_card_confirm = None
                self._shop_expecting_priority_card_is_change = False
                self._customize_done_count = 0
                self._customize_cards_tried = 0
            elif p_pts == self._shop_last_points:
                self._shop_refresh_noop_streak += 1
                if self._shop_refresh_noop_streak >= 3:
                    self._shop_noop_features.add("リフレッシュ")
        self._shop_last_clicked_feature = None
        if self._shop_last_points is not None and p_pts != self._shop_last_points:
            self._shop_clicked_slots.clear()
        held_drinks = list(getattr(obs, "drinks", []) or [])
        cap_drinks = 4 if (obs and obs.scenario_id in ("hif_honsen", "hif")) else 3
        deck_ids = list(getattr(self.tracker, "produce_deck", []) or [])
        has_trouble = any(self._is_trouble_card_id(cid_d) for cid_d in deck_ids)
        resolved_pid, _, _ = self._resolve_idol_and_effect(obs, p_idol_id=p_idol_id)
        forb_bases = _FORBIDDEN_BASE_IDS_BY_P_IDOL.get(int(resolved_pid or 0), frozenset())
        has_forbidden_in_deck = False
        has_basic_n_or_avoided = False
        if deck_ids:
            for cid_d in deck_ids:
                cid_di = int(cid_d)
                drow = self.db.get_card_by_id(cid_di)
                dbase = int(drow.get("base_id") or cid_di) if drow else cid_di
                if forb_bases and (
                    cid_di in forb_bases
                    or dbase in forb_bases
                    or (resolved_pid == 163 and cid_di in (152, 153, 426, 427))
                ):
                    has_forbidden_in_deck = True
                if (
                    self._is_avoid_card(cid_di, base_id=dbase)
                    or (drow is not None and str(drow.get("rarity") or "").upper() == "N")
                ):
                    has_basic_n_or_avoided = True
        has_upgradable_card = True
        if deck_ids:
            has_upgradable_card = False
            for cid_d in deck_ids:
                crow = self.db.get_card_by_id(int(cid_d))
                if crow and not crow.get("upgraded") and str(crow.get("type") or "") != "trouble" and crow.get("upgraded_id"):
                    has_upgradable_card = True
                    break

        # Filter out non-priority skill cards unconditionally; allow user `priority_card_ids` if not yet owned & affordable!
        def _slot_id(act: LegalAction) -> str:
            p = act.payload or {}
            return str(p.get("shop_slot_key") or p.get("name") or "")

        allowed = [
            a for a in legal_actions
            if (not self._is_shop_skill_card_action(a) or self._is_shop_buyable_priority_card(a, p_pts))
            and not self._is_shop_banned_drink_action(a)
            and not str((a.payload or {}).get("name") or "").startswith("破棄候補_")
            and (
                _slot_id(a) not in self._shop_clicked_slots
                or str((a.payload or {}).get("name") or "") in ("終了", "次へ", "交換する", "チェンジする", "チェンジ", "リフレッシュ")
            )
            and _slot_id(a) not in self._shop_noop_features
        ]
        if not allowed:
            exit_fallback = [
                a for a in legal_actions
                if str((a.payload or {}).get("name") or "") in ("終了", "次へ", "スキップ", "閉じる")
            ]
            allowed = [
                a for a in legal_actions
                if (not self._is_shop_skill_card_action(a) or self._is_shop_buyable_priority_card(a, p_pts))
                and not self._is_shop_banned_drink_action(a)
                and not str((a.payload or {}).get("name") or "").startswith("破棄候補_")
            ] or exit_fallback or [
                a for a in legal_actions
                if not self._is_shop_banned_drink_action(a)
            ] or legal_actions

        exit_act = next(
            (
                a for a in allowed
                if str((a.payload or {}).get("name") or "") in ("終了", "次へ", "スキップ", "閉じる")
                or any(k in str((a.payload or {}).get("title") or a.description) for k in ("离开商店", "終了", "次へ"))
            ),
            None,
        )

        # Top priority: if an active `交換する` or `チェンジする` matches an unowned user `priority_card_ids` card, buy/exchange it BEFORE drinks!
        priority_card_exchange = next(
            (
                a for a in allowed
                if str((a.payload or {}).get("name") or "") in ("交換する", "チェンジする", "チェンジ")
                and (a.payload or {}).get("card_id") is not None
                and self._is_shop_buyable_priority_card(a, p_pts)
            ),
            None,
        )
        if priority_card_exchange is not None:
            pcid = int((priority_card_exchange.payload or {}).get("card_id") or 0)
            prow = self.db.get_card_by_id(pcid)
            pbase = int(prow.get("base_id") or pcid) if prow else pcid
            ppay = priority_card_exchange.payload or {}
            is_chg_prio = bool(
                ppay.get("is_shop_change_card")
                or ppay.get("shop_action_type") == "change"
                or str(ppay.get("name") or "") in ("チェンジする", "チェンジ")
            )
            self._shop_card_popup_guard = False
            self._shop_expecting_drink_confirm = False
            self._shop_expecting_priority_card_confirm = pcid
            self._shop_expecting_priority_card_is_change = is_chg_prio
            if is_chg_prio:
                self._came_from_change_pickup = True
                self._pending_change_gain_card_id = pcid
                self._pending_change_gain_base_id = pbase
            self._shop_attempted_card_ids.add(pcid)
            self._shop_attempted_card_ids.add(pbase)
            self._shop_refresh_since_purchase = 0
            verb_cn = "Exchanging (チェンジ)" if is_chg_prio else "Purchasing"
            return DecisionResult(
                choice_id=priority_card_exchange.choice_id,
                reason=f"LocalAgent(shop): {verb_cn} priority skill card #{pcid} -> {priority_card_exchange.description}",
                source="local_hybrid",
            )
        self._shop_expecting_priority_card_confirm = None
        self._shop_expecting_priority_card_is_change = False

        has_buyable_priority_card_slot = any(
            self._is_shop_buyable_priority_card(a, p_pts) for a in allowed
        )
        # Next: if an active non-banned drink exchange ("交換する" with drink_id or Pドリンク slot) is ready and affordable, buy it,
        # unless an unowned affordable ★ priority skill card is waiting in the shop!
        drink_exchange = (
            None
            if has_buyable_priority_card_slot
            else next(
                (
                    a for a in allowed
                    if str((a.payload or {}).get("name") or "") == "交換する"
                    and (
                        (a.payload or {}).get("drink_id") is not None
                        or str((a.payload or {}).get("shop_slot_key") or "").startswith("Pドリンク:")
                    )
                    and not self._is_shop_banned_drink_action(a)
                    and (
                        int((a.payload or {}).get("drink_id") or 0) <= 0
                        or int((a.payload or {}).get("drink_id") or 0) not in self._shop_attempted_drink_ids
                    )
                    and len(held_drinks) < cap_drinks
                    and int((a.payload or {}).get("price") or 0) <= p_pts
                ),
                None,
            )
        )
        if drink_exchange is not None:
            self._shop_card_popup_guard = False
            self._shop_expecting_drink_confirm = True
            ex_did = int((drink_exchange.payload or {}).get("drink_id") or 0)
            if ex_did > 0:
                self._shop_attempted_drink_ids.add(ex_did)
            self._shop_refresh_since_purchase = 0
            return DecisionResult(
                choice_id=drink_exchange.choice_id,
                reason=f"LocalAgent(shop): Purchasing P-drink -> {drink_exchange.description}",
                source="local_hybrid",
            )
        self._shop_expecting_drink_confirm = False

        has_affordable_drink_opt = any(
            (
                str((a.payload or {}).get("name") or "").startswith("Pドリンク")
                or str((a.payload or {}).get("shop_slot_key") or "").startswith("Pドリンク:")
            )
            and int((a.payload or {}).get("price") or 0) <= p_pts
            for a in allowed
        )
        refresh_since_purchase = int(getattr(self, "_shop_refresh_since_purchase", 0) or 0)
        can_refresh_for_drinks = any(
            (
                str((a.payload or {}).get("name") or "") == "リフレッシュ"
                or str((a.payload or {}).get("shop_slot_key") or "") == "リフレッシュ"
            )
            and "リフレッシュ" not in self._shop_noop_features
            and self.preferences.spend_all_final_shop
            and self._shop_refresh_count < 8
            and refresh_since_purchase < 2
            and int((a.payload or {}).get("price") or 0) <= 90
            and p_pts >= int((a.payload or {}).get("price") or 0) + 75
            for a in allowed
        )
        # Iron rule in W8 インターバル: while held drinks < 4 and a drink or refresh is available,
        # strictly fill all 4 drink slots BEFORE allowing any card customization (加点) or upgrade!
        need_fill_drinks_first = bool(
            is_final_shop
            and len(held_drinks) < cap_drinks
            and (has_affordable_drink_opt or can_refresh_for_drinks)
        )

        def _score_shop_allowed(act: LegalAction) -> int:
            p = act.payload or {}
            nm = str(p.get("name") or "")
            slot_key = str(p.get("shop_slot_key") or "")
            price = int(p.get("price") or 0)
            if nm in self._shop_noop_features or slot_key in self._shop_noop_features:
                return -5_000
            if price > p_pts and nm not in ("終了", "次へ"):
                return -5_000
            if self._is_shop_buyable_priority_card(act, p_pts):
                # User-configured ★ 高优先必选卡 (not yet in deck): top priority in shop, ABOVE drinks!
                return 2_600
            if nm == "削除" or slot_key == "削除":
                if has_forbidden_in_deck:
                    return 2_400
                if is_first_shop and not self._first_shop_delete_done:
                    # First shop (W6 相談): always inspect 削除 once to clear any W1-W5 Trouble (T / #23 眠気)
                    # card even if it was injected by an event/授業 without updating tracker.produce_deck.
                    # Inspecting 削除 costs 0P (cancels cleanly if no T/weak card is present).
                    return 2_400
                if self._first_shop_delete_done:
                    return -5_000
                if has_trouble:
                    return 2_400
                return 1_200 if ((not deck_ids) or has_basic_n_or_avoided) else -5_000
            if nm == "強化" or slot_key == "強化":
                if not has_upgradable_card:
                    return -5_000
                if need_fill_drinks_first:
                    return -5_000
                # Post-Battle-1 final shop (W8 インターバル): low priority for 強化; prefer Pドリンク (full 4) -> カスタマイズ first!
                return 950 if is_final_shop else 2_100
            if (nm.startswith("Pドリンク") or slot_key.startswith("Pドリンク:")) and len(held_drinks) < cap_drinks:
                return 2_400 if is_final_shop else 1_950
            if nm == "カスタマイズ" or slot_key == "カスタマイズ":
                # Iron rule in W8 インターバル: MUST fill all 4 drink slots before any card customization (加点)!
                if need_fill_drinks_first:
                    return -5_000
                if is_final_shop and self._customize_done_count < 3 and p_pts >= max(70, price):
                    return 2_250
                if self._customize_done_count < 2 and p_pts >= 70:
                    return 1_800
            if (nm == "セレクトチェンジ" or slot_key == "セレクトチェンジ"):
                if has_forbidden_in_deck:
                    return 1_900
                if has_trouble:
                    return 1_100
            if nm == "リフレッシュ" or slot_key == "リフレッシュ":
                noop_probe = self._shop_refresh_noop_streak > 0
                if is_final_shop:
                    if (
                        self.preferences.spend_all_final_shop
                        and self._shop_refresh_count < 8
                        and (refresh_since_purchase < 2 or noop_probe)
                        and price <= 200
                        and ((len(held_drinks) < cap_drinks and p_pts >= price + 60) or p_pts >= price + 70)
                    ):
                        # When <4 drinks and no affordable drink is on shelf, refresh aggressively to fill 4 drinks!
                        return 2_100 if need_fill_drinks_first else 1_350
                elif (
                    self._shop_refresh_count < 2
                    and (refresh_since_purchase < 2 or noop_probe)
                    and price <= 30
                    and p_pts >= price + 100
                    and len(held_drinks) < cap_drinks
                ):
                    return 1_050
            if act is exit_act or nm in ("終了", "次へ"):
                return 800
            return -100

        best = max(allowed, key=_score_shop_allowed)
        if _score_shop_allowed(best) < 0 and exit_act is not None:
            best = exit_act
        self._shop_last_points = p_pts
        best_key = _slot_id(best)
        best_name = str((best.payload or {}).get("name") or "")
        if best_key == "リフレッシュ":
            self._shop_refresh_count += 1
            self._shop_refresh_since_purchase = int(getattr(self, "_shop_refresh_since_purchase", 0) or 0) + 1
            self._shop_last_clicked_feature = "リフレッシュ"
        elif best is not exit_act and best_key and best_name not in ("終了", "次へ", "交換する", "チェンジする", "チェンジ"):
            self._shop_clicked_slots.add(best_key)
            if best_key in ("強化", "削除", "カスタマイズ", "セレクトチェンジ"):
                self._shop_last_clicked_feature = best_key
        return DecisionResult(
            choice_id=best.choice_id,
            reason=f"LocalAgent(shop no-new-card): Chosen {best.description}",
            source="local_hybrid",
        )

    def _decide_popup(
        self,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation],
    ) -> DecisionResult:
        raw_l = (obs.raw_text or "") if obs is not None else ""
        # Never confirm buying a skill card or spending Jewels / retiring!
        forbid_tokens = ("ジュエル", "リタイア", "プロデュース終了", "ジュエル消耗停止")
        safe_acts = [
            a for a in legal_actions
            if not any(fk in str((a.payload or {}).get("name") or a.description) for fk in forbid_tokens)
        ] or legal_actions

        is_retire_or_quit = any(
            k in raw_l for k in ("リタイア", "中断", "あきらめる", "プロデュース終了", "プロデュースを", "データ削除")
        )
        has_banned_name_in_raw = any(bnm and bnm in raw_l for bnm in self._banned_drink_names)
        is_discard_confirm = ("捨てますか" in raw_l) or ("廃棄確認" in raw_l) or ("破棄しますか" in raw_l)
        if (not is_retire_or_quit) and is_discard_confirm:
            confirm_del = next(
                (a for a in safe_acts if str((a.payload or {}).get("name") or "") in ("決定", "はい", "OK", "確認")),
                None,
            )
            cancel_del = next(
                (
                    a for a in safe_acts
                    if str((a.payload or {}).get("name") or "") in ("いいえ", "キャンセル", "閉じる", "とじる")
                ),
                None,
            )
            if self._expecting_drink_discard or has_banned_name_in_raw:
                if confirm_del is not None:
                    self._expecting_drink_discard = False
                    return DecisionResult(
                        choice_id=confirm_del.choice_id,
                        reason=f"LocalAgent(popup): Confirming P-drink discard via {confirm_del.description}",
                        source="local_hybrid",
                    )
            elif cancel_del is not None:
                return DecisionResult(
                    choice_id=cancel_del.choice_id,
                    reason=f"LocalAgent(popup): Cancelling unintended drink discard via {cancel_del.description}",
                    source="local_hybrid",
                )

        if (not is_retire_or_quit) and (not is_discard_confirm) and (
            self._expecting_drink_discard or has_banned_name_in_raw
        ):
            discard_btn = next(
                (a for a in safe_acts if str((a.payload or {}).get("name") or "") in ("捨てる", "破棄する", "破棄")),
                None,
            )
            if discard_btn is not None:
                return DecisionResult(
                    choice_id=discard_btn.choice_id,
                    reason=f"LocalAgent(popup): Discarding P-drink via {discard_btn.description}",
                    source="local_hybrid",
                )
            has_use_or_buy = any(
                str((a.payload or {}).get("name") or "") in ("使う", "使用する", "交換", "交換する")
                for a in safe_acts
            )
            if has_use_or_buy:
                close_or_cancel = next(
                    (
                        a for a in safe_acts
                        if any(
                            k in str((a.payload or {}).get("name") or a.description)
                            for k in ("キャンセル", "いいえ", "閉じる", "とじる")
                        )
                    ),
                    None,
                )
                if close_or_cancel is not None:
                    self._expecting_drink_discard = False
                    return DecisionResult(
                        choice_id=close_or_cancel.choice_id,
                        reason=f"LocalAgent(popup): Refusing banned drink action via {close_or_cancel.description}",
                        source="local_hybrid",
                    )

        expecting_any_shop_confirm = bool(
            self._shop_expecting_drink_confirm or self._shop_expecting_priority_card_confirm
        )
        if ("交換しますか" in raw_l or "交換確認" in raw_l) and ("ドリンク" not in raw_l) and (not expecting_any_shop_confirm):
            cancel = next(
                (a for a in safe_acts if any(k in str((a.payload or {}).get("name") or a.description) for k in ("キャンセル", "いいえ", "閉じる"))),
                None,
            )
            if cancel is not None:
                return DecisionResult(
                    choice_id=cancel.choice_id,
                    reason="LocalAgent(popup): Cancelling unexpected card exchange confirmation",
                    source="local_hybrid",
                )
        cancel_rest = next(
            (
                a for a in safe_acts
                if any(k in str((a.payload or {}).get("name") or a.description) for k in ("キャンセル", "いいえ", "閉じる", "とじる"))
            ),
            None,
        )
        is_rest_confirm_modal = bool(
            ("休み確認" in raw_l)
            or ("休みますか" in raw_l)
            or (
                cancel_rest is not None
                and any(
                    "休む" in f"{(a.payload or {}).get('name') or ''} {a.description or ''}"
                    for a in safe_acts
                )
            )
        )
        if is_rest_confirm_modal and cancel_rest is not None:
            if (
                obs is not None
                and obs.scenario_id in ("hif_honsen", "hif")
                and int(obs.week or 0) == 6
            ):
                return DecisionResult(
                    choice_id=cancel_rest.choice_id,
                    reason="LocalAgent(popup): Cancelling W6 rest modal to inspect 相談",
                    source="local_hybrid",
                )
            cur_wk = int(getattr(obs, "week", 0) or 0) if obs is not None else 0
            if self._rest_hp_recheck_week != cur_wk:
                self._rest_hp_recheck_confirmed = False
                self._rest_hp_recheck_cancels = 0
                self._rest_hp_recheck_week = cur_wk
            if self._rest_hp_recheck_confirmed:
                self._rest_hp_recheck_confirmed = False
                self._pending_rest_hp_recheck = False
            elif self._rest_hp_recheck_cancels < 2:
                self._pending_rest_hp_recheck = True
                self._rest_hp_recheck_week = cur_wk
                self._rest_hp_recheck_cancels += 1
                return DecisionResult(
                    choice_id=cancel_rest.choice_id,
                    reason="LocalAgent(popup): Exiting rest modal once to re-verify true HP on schedule",
                    source="local_hybrid",
                )
        elif not is_rest_confirm_modal:
            self._pending_rest_hp_recheck = False
            self._rest_hp_recheck_confirmed = False
        if self._shop_expecting_priority_card_confirm and (
            "交換しますか" in raw_l
            or "交換確認" in raw_l
            or "チェンジしますか" in raw_l
            or "チェンジ確認" in raw_l
            or any(
                k in str((a.payload or {}).get("name") or a.description)
                for a in safe_acts
                for k in ("交換", "チェンジ")
            )
        ):
            bought_cid = int(self._shop_expecting_priority_card_confirm)
            is_chg_prio = bool(self._shop_expecting_priority_card_is_change)
            self._shop_expecting_priority_card_confirm = None
            self._shop_expecting_priority_card_is_change = False
            if is_chg_prio:
                crow_b = self.db.get_card_by_id(bought_cid)
                self._came_from_change_pickup = True
                self._pending_change_gain_card_id = bought_cid
                self._pending_change_gain_base_id = (
                    int(crow_b.get("base_id") or bought_cid) if crow_b else bought_cid
                )
            elif self.tracker is not None and not self._deck_has_card_family(bought_cid):
                self.tracker.produce_deck.append(bought_cid)
        if self._shop_expecting_drink_confirm and ("交換しますか" in raw_l or "交換確認" in raw_l or any("交換" in str((a.payload or {}).get("name") or a.description) for a in safe_acts)):
            self._shop_expecting_drink_confirm = False

        # In H.I.F retry modal, pick 再挑戦 only when explicitly offered safely
        confirm_order = (
            "再開する",
            "再開",
            "続きから",
            "決定",
            "交換する",
            "交換",
            "強化する",
            "削除する",
            "カスタマイズする",
            "強化",
            "削除",
            "チェンジする",
            "チェンジ",
            "はい",
            "確認",
            "OK",
            "使う",
            "使用する",
            "次へ",
            "閉じる",
            "とじる",
            "再挑戦",
        )

        def _score_popup(act: LegalAction) -> int:
            nm = str((act.payload or {}).get("name") or "")
            desc = str(act.description or "")
            combined = f"{nm} {desc}"
            if any(bad in combined for bad in forbid_tokens):
                return -10_000
            if nm in ("捨てる", "破棄する", "破棄") or "捨てる" in combined or "破棄" in combined:
                return -5_000
            for idx, tok in enumerate(confirm_order):
                if tok in nm or tok in desc:
                    return 2_000 - idx * 25
            if "キャンセル" in combined or "いいえ" in combined:
                return 200
            return 500

        best = max(safe_acts, key=_score_popup)
        return DecisionResult(
            choice_id=best.choice_id,
            reason=f"LocalAgent(popup): Selected {best.description}",
            source="local_hybrid",
        )

    def decide(
        self,
        prompt: str,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation] = None,
        use_session: bool = False,
        scenario_id: Optional[str] = None,
        p_idol_id: Optional[int] = None,
        **_: Any,
    ) -> DecisionResult:
        """Select a legal action locally across every Gakumas scene with zero LLM calls."""
        if not legal_actions:
            return DecisionResult(
                choice_id=0,
                reason="LocalAgent: No legal actions provided; waiting.",
                source="local_hybrid",
            )

        scene = obs.scene_type if obs is not None else "unknown"
        if scene not in ("popup", "loading", "skip"):
            self._expecting_drink_discard = False
            if obs is not None and not any(
                self._is_banned_drink(
                    getattr(d, "drink_id", None),
                    getattr(d, "name", ""),
                    getattr(d, "dsl_summary", ""),
                )
                for d in (obs.drinks or ())
            ):
                self._banned_drink_discard_attempts = 0
                self._banned_drink_discard_ctx = None
        if scene != "battle_turn":
            # Reset battle-specific caches once out of battle scenes (NEVER on transient popups OR mid-battle #22 移動 modal!)
            raw_obs_txt = (obs.raw_text or "") if obs is not None else ""
            is_mid_battle_move_modal = (
                scene == "card_reward"
                and (
                    any(
                        str((a.payload or {}).get("name") or "") == "移動する"
                        or "手札に移動" in str(a.description or "")
                        or "加入手牌" in str(a.description or "")
                        for a in legal_actions
                    )
                    or (
                        ("移動" in raw_obs_txt or "手札" in raw_obs_txt)
                        and not any(k in raw_obs_txt for k in ("受け取る", "チェンジ", "削除", "強化", "カスタマイズ"))
                    )
                )
            )
            _act_blob = " ".join(
                f"{(a.payload or {}).get('name') or ''} {a.description or ''}" for a in legal_actions
            )
            has_post_battle_reward_cue = any(
                k in raw_obs_txt or k in _act_blob
                for k in (
                    "受け取る",
                    "受け取らない",
                    "再抽選",
                    "獲得",
                    "報酬",
                    "CLEAR",
                    "PERFECT",
                    "削除",
                    "強化",
                    "カスタマイズ",
                    "セレクトチェンジ",
                    "特別指導",
                )
            )
            in_active_battle = bool(self.tracker is not None and getattr(self.tracker, "in_battle", False))
            should_reset_on_reward = (
                scene == "card_reward"
                and not is_mid_battle_move_modal
                and (not in_active_battle or has_post_battle_reward_cue)
            )
            if scene in ("produce_schedule", "shop", "home", "produce_setup") or should_reset_on_reward:
                self.battle_planner.reset_battle()
        if scene in ("produce_schedule", "battle_turn", "home", "produce_setup"):
            self._in_shop_visit = False
            self._shop_clicked_slots.clear()
            self._shop_noop_features.clear()
            self._shop_last_clicked_feature = None
            self._shop_last_points = None
            self._shop_refresh_count = 0
            self._shop_refresh_noop_streak = 0
            self._shop_expecting_drink_confirm = False
            self._shop_expecting_priority_card_confirm = None
            self._shop_expecting_priority_card_is_change = False
            self._shop_attempted_drink_ids.clear()
            self._shop_attempted_card_ids.clear()
            self._reward_seen_slots.clear()
            self._reward_slot_attempts.clear()
            self._reward_visited_grid_slots.clear()
            self._reward_cancel_attempts = 0
            self._reward_grid_scores.clear()
            self._came_from_change_pickup = False
            self._pending_change_gain_card_id = None
            self._pending_change_gain_base_id = None
            self._customize_clicked_tiles.clear()
            self._customize_cards_tried = 0
            self._customize_done_count = 0
        elif scene in ("produce_event", "shop"):
            self._reward_seen_slots.clear()
            self._reward_slot_attempts.clear()
            self._reward_visited_grid_slots.clear()
            self._reward_cancel_attempts = 0
            self._reward_grid_scores.clear()
            if scene == "produce_event" or not self._shop_expecting_priority_card_confirm:
                self._came_from_change_pickup = False
                self._pending_change_gain_card_id = None
                self._pending_change_gain_base_id = None
            if scene == "produce_event":
                self._shop_clicked_slots.clear()
                self._shop_last_clicked_feature = None
                self._shop_last_points = None
                self._shop_attempted_drink_ids.clear()
                self._shop_attempted_card_ids.clear()
                self._shop_expecting_priority_card_confirm = None
                self._shop_expecting_priority_card_is_change = False
                self._shop_refresh_count = 0
                self._shop_refresh_noop_streak = 0
                self._customize_clicked_tiles.clear()
                self._customize_cards_tried = 0
                self._customize_done_count = 0
        elif scene in ("card_reward", "popup"):
            self._shop_last_clicked_feature = None

        if scene == "battle_turn" and obs is not None:
            self._has_entered_produce = True
            cid, reason, _stats = self.battle_planner.decide_battle_action(obs, legal_actions, tracker=self.tracker)
            return DecisionResult(choice_id=cid, reason=reason, source="local_hybrid")

        if scene == "home":
            return self._decide_home(legal_actions)
        if scene == "produce_setup":
            return self._decide_produce_setup(legal_actions)
        if scene == "produce_schedule":
            return self._decide_produce_schedule(legal_actions, obs, p_idol_id=p_idol_id)
        if scene == "produce_event":
            return self._decide_produce_event(legal_actions, obs, p_idol_id=p_idol_id)
        if scene == "card_reward":
            return self._decide_card_reward(legal_actions, obs, p_idol_id=p_idol_id)
        if scene == "shop":
            return self._decide_shop(legal_actions, obs, p_idol_id=p_idol_id)
        if scene == "popup":
            return self._decide_popup(legal_actions, obs)
        if scene == "loading":
            return DecisionResult(
                choice_id=legal_actions[0].choice_id,
                reason="LocalAgent(loading): Waiting for transition",
                source="rule_skip",
            )

        # "skip" or unknown generic continuation
        raw_skip = (obs.raw_text or "") if obs is not None else ""
        is_pre_produce_setup_text = any(
            sk in raw_skip
            for sk in (
                "開始確認",
                "獲得量上昇",
                "選抜試験メモリー",
                "プロデュース開始",
                "ース開始",
                "育成情報",
                "Pアイドル",
                "アイドル選択",
                "TrueEnd条件:達成",
                "TrueEnd条件：達成",
                "True End条件:達成",
                "True End条件：達成",
            )
        )
        has_true_end_result = (
            ("TrueEnd" in raw_skip or "True End" in raw_skip)
            and ("TrueEnd条件" not in raw_skip)
            and ("True End条件" not in raw_skip)
        )
        if scene == "skip" and obs is not None and not is_pre_produce_setup_text and (
            has_true_end_result
            or any(tok in raw_skip for tok in ("最終評価", "総合評価", "プロデュース完了"))
        ):
            self._has_entered_produce = True
        first = legal_actions[0]
        return DecisionResult(
            choice_id=first.choice_id,
            reason=f"LocalAgent({scene}): Proceeding via {first.description}",
            source="local_hybrid",
        )
