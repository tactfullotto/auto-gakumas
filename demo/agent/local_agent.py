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
        self._shop_attempted_drink_ids: Set[int] = set()
        self._shop_clicked_slots: Set[str] = set()
        self._shop_noop_features: Set[str] = set()
        self._shop_last_clicked_feature: Optional[str] = None
        self._shop_last_points: Optional[int] = None
        self._shop_refresh_count: int = 0
        self._expecting_drink_discard: bool = False
        self._banned_drink_discard_attempts: int = 0
        self._banned_drink_discard_ctx: Optional[Tuple[str, int]] = None
        self.set_preferences(preferences)

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
                if row.get("upgraded_id"):
                    cids.add(int(row["upgraded_id"]))
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
        b_i = int(base_id or cid_i or 0)
        if not b_i and cid_i:
            row = self.db.get_card_by_id(cid_i)
            b_i = int(row.get("base_id") or cid_i) if row else cid_i
        return (cid_i > 0 and cid_i in self._priority_card_ids) or (
            b_i > 0 and b_i in self._priority_card_base_ids
        )

    def _is_keep_card(self, card_id: Optional[int], base_id: int = 0) -> bool:
        cid_i = int(card_id or 0)
        b_i = int(base_id or cid_i or 0)
        return (cid_i > 0 and cid_i in self._keep_card_ids) or (
            b_i > 0 and b_i in self._keep_card_base_ids
        )

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
        self._shop_attempted_drink_ids.clear()
        self._shop_clicked_slots.clear()
        self._shop_noop_features.clear()
        self._shop_last_clicked_feature = None
        self._shop_last_points = None
        self._shop_refresh_count = 0
        self._expecting_drink_discard = False
        self._banned_drink_discard_attempts = 0
        self._banned_drink_discard_ctx = None
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
        b_i = int(base_id or cid_i or 0)
        if not b_i and cid_i:
            row = self.db.get_card_by_id(cid_i)
            b_i = int(row.get("base_id") or cid_i) if row else cid_i
        if (cid_i > 0 and cid_i in self._avoid_card_ids) or (b_i > 0 and b_i in self._avoid_card_base_ids):
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

    def _deck_has_removable_unwanted_card(
        self,
        *,
        target_effect: Optional[str],
        guide: Optional[PIdolGuide],
        plan: str,
        p_idol_id: Optional[int] = None,
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
            )
            if rem_score >= 100:
                return True
        return False

    @staticmethod
    def _stat_values(obs: Optional[ScreenObservation]) -> Dict[str, int]:
        stats = (obs.idol_stats if obs and obs.idol_stats else None) or {"vocal": 2400, "dance": 1400, "visual": 1800}
        return {
            "vo": min(STAT_CAP, max(0, int(stats.get("vocal", 180) or 180))),
            "da": min(STAT_CAP, max(0, int(stats.get("dance", 160) or 160))),
            "vi": min(STAT_CAP, max(0, int(stats.get("visual", 150) or 150))),
        }

    @classmethod
    def _stat_rank_map(cls, obs: Optional[ScreenObservation]) -> Dict[str, int]:
        vals = cls._stat_values(obs)
        ordered = sorted(
            vals.items(),
            key=lambda kv: kv[1],
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
        vals = cls._stat_values(obs)
        ranks = cls._stat_rank_map(obs)
        week = int(getattr(obs, "week", 1) or 1)
        reserve = 140 if (attr == "vo" and week <= 6) else 60
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
    ) -> int:
        """Score a skill card by idol `recommended_effect`, PIdolGuide, rarity, and deck quality."""
        row = self.db.get_card_by_id(int(card_id)) if card_id else None
        cname = str(row["name"]) if row else str(title or "")
        ctype = str(row.get("type") or "") if row else ""
        rarity = str(row.get("rarity") or "").upper() if row else ""
        dsl = f"{row.get('dsl_actions', '')} {row.get('dsl_effects', '')}".lower() if row else ""
        c_rec = str(row.get("recommended_effect") or "") if row else ""
        text_l = f"{cname} {title} {desc} {dsl}".lower()
        upgraded = bool((row and row.get("upgraded")) or "+" in cname or "+" in title)
        base_id = int(row.get("base_id") or card_id) if (row is not None and card_id) else (int(card_id) if card_id else 0)

        # Trouble cards (`眠気` / `トラブル`): terrible to pick, top priority to remove
        if card_id == 23 or ctype == "trouble" or "眠気" in cname or "トラブル" in cname:
            return 5000 if for_removal else -10_000

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
        if (card_id and int(card_id) in self._avoid_card_ids) or (
            base_id and int(base_id) in self._avoid_card_base_ids
        ):
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

        # Primary `recommended_effect` alignment
        if target_effect:
            if c_rec == target_effect:
                score += 650
            elif target_effect.lower() in dsl:
                score += 500
            prim_kws = _EFFECT_KEYWORDS.get(target_effect, ())
            if any(kw in text_l for kw in prim_kws):
                score += 380
            sec_kws = _SECONDARY_EFFECT_KEYWORDS.get(target_effect, ())
            if any(kw in text_l for kw in sec_kws):
                score += 120

        # PIdolGuide affinity / anti-synergy
        if guide is not None:
            if card_id and int(card_id) in guide.recommended_cards:
                score += 420
            elif any(k in cname for k in guide.recommended_card_names):
                score += 360
            if card_id and int(card_id) in guide.caution_cards:
                score -= 350
            elif any(k in cname for k in guide.caution_card_names):
                score -= 280

        if row is not None and row.get("is_signature"):
            score += 350
        if upgraded:
            score += 90
        if "cardusesremaining+=" in dsl or "使用数追加" in text_l:
            score += 140

        # Protect multiplicative 好印象 scaling (#521 夢色リップ+) & GI effect-buff engines (#920/#617/#778)
        if "goodimpressionturns*=" in dsl or "setgoodimpressionturnseffectbuff" in dsl:
            score += 2_200

        deck_count = 0
        family_in_deck = False
        if self.tracker is not None and card_id:
            deck_ids_tr = self.tracker.produce_deck or []
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
        is_logic_gi_build = (plan.lower() == "logic") or (target_effect == "goodImpressionTurns")
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
        # prefer an already-owned family so Page 2 can swap X -> X cleanly without losing core cards.
        if (not for_removal) and in_change_pickup and family_in_deck and (has_removable_unwanted is False):
            score += 4_000 if upgraded else 3_500
            if not is_unique_card:
                score += 400

        tracker_hooks = (getattr(self.tracker, "card_memory_hooks", None) or {}) if self.tracker is not None else {}
        hook_meta = (
            tracker_hooks.get(int(card_id))
            or tracker_hooks.get(str(card_id))
            or tracker_hooks.get(base_id)
            or tracker_hooks.get(str(base_id))
            or {}
        ) if card_id else {}
        has_real_hook = bool(
            hook_meta.get("force_initial_hand")
            or hook_meta.get("free_extra_use_first_play")
            or (hook_meta.get("exiles_trouble_on_play") and base_id != 426)
            or base_id in (126, 128, 136, 296, 298)
        )
        if has_real_hook and not (for_removal and deck_count >= 2):
            score += 1_200

        rarity_pts = {"LEGEND": 240, "SSR": 200, "SR": 120, "R": 60, "N": 10}
        score += rarity_pts.get(rarity, 30)

        # Protect core Logic/H.I.F loop & finisher cards from swaps/removals
        if is_core_keep:
            score += 2_000

        # Apply user-configured card priorities, locks, and manual numeric score overrides
        user_priority = self._is_priority_card(card_id, base_id=base_id)
        user_keep = self._is_keep_card(card_id, base_id=base_id)
        card_overrides = self.preferences.card_score_overrides or {}
        override_delta = 0
        if card_id and int(card_id) in card_overrides:
            override_delta = int(card_overrides[int(card_id)])
        elif base_id and int(base_id) in card_overrides:
            override_delta = int(card_overrides[int(base_id)])

        if user_priority:
            score += 20_000
        elif user_keep and for_removal:
            score += 15_000
        if override_delta:
            score += override_delta

        has_strong_user_protection = (
            user_priority
            or (for_removal and user_keep)
            or (for_removal and override_delta >= 500)
        )

        is_ordinary_unupgraded_active = (
            ctype == "active"
            and not upgraded
            and not is_elite_active
            and not is_core_keep
            and not has_strong_user_protection
        )
        is_non_elite_active = (
            ctype == "active"
            and not is_elite_active
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
            (rarity == "R" and not upgraded) or (card_id in (68, 69, 94) and not upgraded)
        ):
            score -= 600
            if for_removal:
                return max(450, -score)
        elif rarity == "SR" and not upgraded and not is_core_keep and not has_strong_user_protection and ctype != "mental":
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
    ) -> DecisionResult:
        self._has_entered_produce = True
        self._reward_preview_count = 0
        self._last_reward_sig = None

        maybe_discard = self._maybe_pick_banned_drink_discard(
            legal_actions,
            "produce_schedule",
            int(getattr(obs, "week", 0) or 0) if obs is not None else 0,
        )
        if maybe_discard is not None:
            return maybe_discard

        stamina = int(getattr(obs, "stamina", 30) or 30)
        max_stam = max(1, int(getattr(obs, "max_stamina", 44) or 44))
        low_hp = stamina < max(8, int(round(max_stam * 0.22)))

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
            )
            if is_sp:
                return 10_000 + attr_bonus

            is_rest = any(k in combined for k in ("休む", "休息", "rest"))
            if low_hp and is_rest:
                return 5_000

            base = 300
            if any(k in combined for k in ("追い込み", "公開レッスン", "授業", "ボーカル", "ダンス", "ビジュアル", "レッスン")):
                base = 2_000
            elif any(k in combined for k in ("特別指導", "相談")):
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
        if _is_sp_action(best) and self._stat_attr_bonus(obs, best_combined, est_gain=420, is_sp=True) > -5_000:
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
        _, target_eff, guide = self._resolve_idol_and_effect(obs, p_idol_id=p_idol_id)
        eff_kws = _EFFECT_KEYWORDS.get(target_eff or "goodImpressionTurns", ("好印象",))
        sec_kws = _SECONDARY_EFFECT_KEYWORDS.get(target_eff or "goodImpressionTurns", ("やる気", "元気"))
        is_jugyo_event = bool(
            obs and ("授業" in (obs.week_activity or "") or "授業" in (obs.raw_text or ""))
        )

        def _score_event(act: LegalAction) -> int:
            p = act.payload or {}
            cid = p.get("card_id")
            title = str(p.get("title") or p.get("name") or "")
            desc = str(act.description or "")
            combined = f"{title} {desc}".lower()

            if any(bad in combined for bad in ("ジュエル", "リタイア")):
                return -10_000
            score = 0
            if cid:
                return 2_000 + self._score_skill_card(
                    card_id=int(cid),
                    title=title,
                    desc=desc,
                    target_effect=target_eff,
                    guide=guide,
                    plan=(obs.plan if obs else "logic"),
                )
            if any(tk in combined for tk in ("トラブル", "トラプル", "眠気", "睡眠", "干扰卡")):
                score -= 4_500
            if any(k in combined for k in eff_kws):
                score += 900
            elif any(k in combined for k in sec_kws):
                score += 400
            attr_bonus = self._stat_attr_bonus(
                obs,
                combined,
                est_gain=80,
                is_sp=False,
                prefer_non_highest=is_jugyo_event,
            )
            if attr_bonus < 0:
                score += attr_bonus
            elif is_jugyo_event:
                score += 180 if attr_bonus >= 120 else (120 if attr_bonus > 0 else 0)
            else:
                score += (attr_bonus // 40) * 60
            if "削除" in combined or "強化" in combined:
                score += 250
            if "スキップ" in combined or "何もしない" in combined:
                score -= 200
            return score

        best = max(legal_actions, key=_score_event)
        return DecisionResult(
            choice_id=best.choice_id,
            reason=f"LocalAgent(event): Matching {target_eff or 'plan'} -> {best.description}",
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
            if nm.startswith("破棄候補_") and self._is_banned_drink(
                p.get("drink_id"),
                str(p.get("title") or nm),
                str(a.description or ""),
            ):
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
        dsl = str(row.get("effect_dsl") or "").lower()
        if (
            cid_i in _CORE_KEEP_CARD_IDS
            or base_id in _CORE_KEEP_BASE_IDS
            or row.get("is_signature")
            or "extra_turn" in dsl
            or "追加ターン" in text_all
            or "cardusesremaining+=" in dsl
            or "使用数追加" in text_all
        ):
            return False
        return True

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
                if str((a.payload or {}).get("name") or "") in ("受け取る", "強化する", "削除する", "チェンジする", "カスタマイズする", "次へ", "スキップ")
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
                and claim_name_peek not in ("受け取る", "強化する", "カスタマイズする", "次へ")
                and any(k in raw_txt for k in ("削除", "手放す", "チェンジ"))
            )
        )
        is_drink_flow = ("ドリンク" in raw_txt and "スキルカード" not in raw_txt) or any(is_drink_opt(a) for a in legal_actions)
        is_change_pickup = (not is_removal) and (not is_drink_flow) and (
            ("チェンジで獲得" in raw_txt) or (claim_name_peek == "次へ")
        )
        has_removable_unwanted: Optional[bool] = None
        if (is_removal or is_change_pickup) and not is_drink_flow:
            has_removable_unwanted = self._deck_has_removable_unwanted_card(
                target_effect=target_eff,
                guide=guide,
                plan=plan,
                p_idol_id=resolved_pid,
            )

        reroll_act = next(
            (
                a for a in legal_actions
                if str((a.payload or {}).get("name") or "") == "再抽選"
                and "ジュエル" not in str((a.payload or {}).get("title") or a.description)
            ),
            None,
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
            and str((a.payload or {}).get("name") or "") not in ("受け取る", "強化する", "削除する", "チェンジする", "カスタマイズする", "次へ")
        ]
        if direct_card_opts:
            best_direct = max(
                direct_card_opts,
                key=lambda a: self._score_skill_card(
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
                    check_same_card=(claim_name_peek != "削除する"),
                ),
            )
            best_cid = (best_direct.payload or {}).get("card_id")
            if early_reroll_ok and self._is_non_exhaust_mental_card(
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
            return DecisionResult(
                choice_id=best_direct.choice_id,
                reason=f"LocalAgent(reward): Direct card pick ({target_eff}) -> {best_direct.description}",
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
                    int(self._last_clicked_grid_slot)
                    if self._last_clicked_grid_slot is not None
                    else (int(claim_slot) if claim_slot is not None else self._infer_current_reward_slot(preview_acts))
                )
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
                        check_same_card=(claim_name != "削除する"),
                    )
                    if claim_cid
                    else -500
                )
                self._reward_grid_scores[cur_slot] = cur_score
                self._reward_visited_grid_slots.add(cur_slot)
                if cur_score >= 4000 or cur_score > 0:
                    self._came_from_change_pickup = False
                    self._pending_change_gain_card_id = None
                    self._pending_change_gain_base_id = None
                    self._clear_reward_scan_state()
                    return DecisionResult(
                        choice_id=claim_act.choice_id,
                        reason=f"LocalAgent(removal): Confirming weak/trouble card (score={cur_score}) -> {claim_act.description}",
                        source="local_hybrid",
                    )
                unvisited_grid = [
                    p for p in preview_acts
                    if (
                        (s := self._extract_reward_slot_idx(p)) is None
                        or (s != cur_slot and s not in self._reward_visited_grid_slots)
                    )
                ]
                max_grid_scans = 12 if cur_score <= -1500 else 6
                if unvisited_grid and self._reward_preview_count < max_grid_scans:
                    self._reward_preview_count += 1
                    nxt = max(
                        unvisited_grid,
                        key=lambda a: (
                            1 if ("暗灰" in a.description or "基本" in a.description or "暗灰" in str((a.payload or {}).get("title") or "")) else 0,
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
                if (
                    cancel_act is not None
                    and claim_name == "チェンジする"
                    and cur_score < -250
                    and (not self._came_from_change_pickup)
                    and self._reward_cancel_attempts == 0
                ):
                    self._reward_cancel_attempts += 1
                    self._pending_change_gain_card_id = None
                    self._pending_change_gain_base_id = None
                    self._clear_reward_scan_state()
                    return DecisionResult(
                        choice_id=cancel_act.choice_id,
                        reason=f"LocalAgent(removal): Cancelling swap once to avoid losing strong card (score={cur_score})",
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
                self._came_from_change_pickup = False
                self._pending_change_gain_card_id = None
                self._pending_change_gain_base_id = None
                self._clear_reward_scan_state()
                return DecisionResult(
                    choice_id=claim_act.choice_id,
                    reason=f"LocalAgent(reward): Claiming/confirming {claim_name} ({claim_act.description})",
                    source="local_hybrid",
                )

            # Skill card 3-choice reward: inspect all candidates before picking globally highest score
            if preview_acts and not is_removal:
                parsed_cid = int(claim_cid) if claim_cid else None
                in_chg_pick = is_change_pickup and (claim_name != "強化する")
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
                )
                cur_slot = self._infer_current_reward_slot(preview_acts)
                self._reward_seen_slots[cur_slot] = (cur_score, parsed_cid, claim_title)

                unvisited_prevs = [
                    p for p in preview_acts
                    if (s := self._extract_reward_slot_idx(p)) is not None
                    and s not in self._reward_seen_slots
                    and self._reward_slot_attempts.get(s, 0) < 2
                ]
                if claim_name == "強化する" and len(self._reward_seen_slots) >= 4:
                    unvisited_prevs = []
                if unvisited_prevs:
                    nxt = unvisited_prevs[0]
                    nslot = self._extract_reward_slot_idx(nxt)
                    if nslot is not None:
                        self._reward_slot_attempts[nslot] = self._reward_slot_attempts.get(nslot, 0) + 1
                    return DecisionResult(
                        choice_id=nxt.choice_id,
                        reason=f"LocalAgent(reward): Scanning unseen card slot (cur={cur_slot}, score={cur_score}, seen={sorted(self._reward_seen_slots)})",
                        source="local_hybrid",
                    )

                if self._reward_seen_slots:
                    best_slot = max(self._reward_seen_slots, key=lambda k: self._reward_seen_slots[k][0])
                    best_seen_score, best_seen_cid, best_seen_title = self._reward_seen_slots[best_slot]
                    if (not in_chg_pick) and claim_name != "強化する" and best_seen_score <= -3_000:
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
                    if (
                        early_reroll_ok
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
                            return DecisionResult(
                                choice_id=best_prev.choice_id,
                                reason=(
                                    f"LocalAgent(reward): Switching to best card slot #{best_slot + 1} "
                                    f"(score={self._reward_seen_slots[best_slot][0]} >= {cur_score})"
                                ),
                                source="local_hybrid",
                            )
                if (not in_chg_pick) and claim_name != "強化する" and cur_score <= -3_000:
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
                return DecisionResult(
                    choice_id=claim_act.choice_id,
                    reason=f"LocalAgent(reward): Claiming best evaluated card (score={cur_score}) -> {claim_act.description}",
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
                    )
                )
            ),
        )
        return DecisionResult(
            choice_id=best.choice_id,
            reason=f"LocalAgent(reward): Fallback pick -> {best.description}",
            source="local_hybrid",
        )

    @staticmethod
    def _is_shop_skill_card_action(act: LegalAction) -> bool:
        """Strictly return True if selecting/clicking this shop action would inspect or buy a new skill card."""
        p = act.payload or {}
        slot_key = str(p.get("shop_slot_key") or "")
        nm = str(p.get("name") or "")
        title = str(p.get("title") or "")
        if slot_key.startswith("スキルカード:") or nm.startswith("スキルカード"):
            return True
        if nm == "交換する" and (p.get("card_id") is not None or (p.get("drink_id") is None and "スキルカード" in title)):
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
        p_pts = int(getattr(obs, "p_points", 0) or 0) if obs is not None else 0
        wk = int(getattr(obs, "week", 0) or 0) if obs is not None else 0
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
        if (
            self._shop_last_clicked_feature in ("強化", "削除", "カスタマイズ", "セレクトチェンジ")
            and self._shop_last_points is not None
            and p_pts == self._shop_last_points
        ):
            self._shop_noop_features.add(self._shop_last_clicked_feature)
        self._shop_last_clicked_feature = None
        if self._shop_last_points is not None and p_pts != self._shop_last_points:
            self._shop_clicked_slots.clear()
        held_drinks = list(getattr(obs, "drinks", []) or [])
        cap_drinks = 4 if (obs and obs.scenario_id in ("hif_honsen", "hif")) else 3
        deck_ids = list(getattr(self.tracker, "produce_deck", []) or [])
        has_trouble = 23 in deck_ids
        resolved_pid, _, _ = self._resolve_idol_and_effect(obs, p_idol_id=p_idol_id)
        forb_bases = _FORBIDDEN_BASE_IDS_BY_P_IDOL.get(int(resolved_pid or 0), frozenset())
        has_forbidden_in_deck = False
        if forb_bases and deck_ids:
            for cid_d in deck_ids:
                cid_di = int(cid_d)
                drow = self.db.get_card_by_id(cid_di)
                dbase = int(drow.get("base_id") or cid_di) if drow else cid_di
                if cid_di in forb_bases or dbase in forb_bases or (resolved_pid == 163 and cid_di in (152, 153, 426, 427)):
                    has_forbidden_in_deck = True
                    break
        has_upgradable_card = True
        if deck_ids:
            has_upgradable_card = False
            for cid_d in deck_ids:
                crow = self.db.get_card_by_id(int(cid_d))
                if crow and not crow.get("upgraded") and str(crow.get("type") or "") != "trouble" and crow.get("upgraded_id"):
                    has_upgradable_card = True
                    break

        # Filter out ALL skill-card buying/previewing options unconditionally, plus already-attempted no-op shop buttons
        def _slot_id(act: LegalAction) -> str:
            p = act.payload or {}
            return str(p.get("shop_slot_key") or p.get("name") or "")

        allowed = [
            a for a in legal_actions
            if not self._is_shop_skill_card_action(a)
            and not self._is_shop_banned_drink_action(a)
            and not str((a.payload or {}).get("name") or "").startswith("破棄候補_")
            and (_slot_id(a) not in self._shop_clicked_slots or _slot_id(a) in ("終了", "次へ", "交換する", "リフレッシュ"))
            and _slot_id(a) not in self._shop_noop_features
        ]
        if not allowed:
            allowed = [
                a for a in legal_actions
                if not self._is_shop_skill_card_action(a)
                and not self._is_shop_banned_drink_action(a)
                and not str((a.payload or {}).get("name") or "").startswith("破棄候補_")
            ] or [
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

        # If an active non-banned drink exchange ("交換する" with drink_id) is ready and affordable, buy it!
        drink_exchange = next(
            (
                a for a in allowed
                if str((a.payload or {}).get("name") or "") == "交換する"
                and (a.payload or {}).get("drink_id") is not None
                and not self._is_shop_banned_drink_action(a)
                and int((a.payload or {}).get("drink_id") or 0) not in self._shop_attempted_drink_ids
                and len(held_drinks) < cap_drinks
                and int((a.payload or {}).get("price") or 0) <= p_pts
            ),
            None,
        )
        if drink_exchange is not None:
            self._shop_card_popup_guard = False
            self._shop_expecting_drink_confirm = True
            self._shop_attempted_drink_ids.add(int((drink_exchange.payload or {}).get("drink_id") or 0))
            return DecisionResult(
                choice_id=drink_exchange.choice_id,
                reason=f"LocalAgent(shop): Purchasing P-drink -> {drink_exchange.description}",
                source="local_hybrid",
            )
        self._shop_expecting_drink_confirm = False

        def _score_shop_allowed(act: LegalAction) -> int:
            p = act.payload or {}
            nm = str(p.get("name") or "")
            slot_key = str(p.get("shop_slot_key") or "")
            price = int(p.get("price") or 0)
            if nm in self._shop_noop_features or slot_key in self._shop_noop_features:
                return -5_000
            if price > p_pts and nm not in ("終了", "次へ"):
                return -5_000
            if nm == "削除" or slot_key == "削除":
                return 2_400 if (has_trouble or has_forbidden_in_deck) else 1_200
            if nm == "強化" or slot_key == "強化":
                return 2_100 if has_upgradable_card else -5_000
            if (nm.startswith("Pドリンク") or slot_key.startswith("Pドリンク:")) and len(held_drinks) < cap_drinks:
                return 1_950
            if nm == "カスタマイズ" or slot_key == "カスタマイズ":
                if is_final_shop and self._customize_done_count < 3 and p_pts >= max(70, price):
                    return 1_850
                if self._customize_done_count < 2 and p_pts >= 70:
                    return 1_800
            if (nm == "セレクトチェンジ" or slot_key == "セレクトチェンジ"):
                if has_forbidden_in_deck:
                    return 1_900
                if has_trouble:
                    return 1_100
            if nm == "リフレッシュ" or slot_key == "リフレッシュ":
                if is_final_shop:
                    if (
                        self.preferences.spend_all_final_shop
                        and self._shop_refresh_count < 8
                        and price <= 200
                        and ((len(held_drinks) < cap_drinks and p_pts >= price + 60) or p_pts >= price + 70)
                    ):
                        return 1_050
                elif (
                    self._shop_refresh_count < 2
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
        if best_key == "リフレッシュ":
            self._shop_refresh_count += 1
            self._shop_clicked_slots.clear()
            self._shop_noop_features.clear()
            self._shop_last_clicked_feature = None
            self._shop_attempted_drink_ids.clear()
            self._customize_done_count = 0
            self._customize_cards_tried = 0
        elif best is not exit_act and best_key and best_key not in ("終了", "次へ", "交換する"):
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
        if (not is_retire_or_quit) and (
            self._expecting_drink_discard or has_banned_name_in_raw or ("捨てますか" in raw_l)
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
            if ("捨てますか" in raw_l) or ("破棄" in raw_l):
                confirm_del = next(
                    (a for a in safe_acts if str((a.payload or {}).get("name") or "") in ("決定", "はい", "OK", "確認")),
                    None,
                )
                if confirm_del is not None:
                    self._expecting_drink_discard = False
                    return DecisionResult(
                        choice_id=confirm_del.choice_id,
                        reason=f"LocalAgent(popup): Confirming P-drink discard via {confirm_del.description}",
                        source="local_hybrid",
                    )
            has_use_or_buy = any(
                str((a.payload or {}).get("name") or "") in ("使う", "使用する", "交換", "交換する")
                for a in safe_acts
            )
            if has_use_or_buy and (has_banned_name_in_raw or self._expecting_drink_discard):
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

        if ("交換しますか" in raw_l or "交換確認" in raw_l) and ("ドリンク" not in raw_l) and (not self._shop_expecting_drink_confirm):
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
        if self._shop_expecting_drink_confirm and ("交換しますか" in raw_l or "交換確認" in raw_l or any("交換" in str((a.payload or {}).get("name") or a.description) for a in safe_acts)):
            self._shop_expecting_drink_confirm = False

        # In H.I.F retry modal, pick 再挑戦 only when explicitly offered safely
        confirm_order = (
            "再開する",
            "再開",
            "続きから",
            "決定",
            "交換する",
            "強化する",
            "削除する",
            "カスタマイズする",
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
        if scene != "popup":
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
            # Reset battle-specific caches once out of battle scenes (except transient popup)
            if scene in ("produce_schedule", "shop", "card_reward", "home", "produce_setup"):
                self.battle_planner.reset_battle()
        if scene in ("produce_schedule", "battle_turn", "home", "produce_setup"):
            self._shop_clicked_slots.clear()
            self._shop_noop_features.clear()
            self._shop_last_clicked_feature = None
            self._shop_last_points = None
            self._shop_refresh_count = 0
            self._shop_expecting_drink_confirm = False
            self._shop_attempted_drink_ids.clear()
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
            self._came_from_change_pickup = False
            self._pending_change_gain_card_id = None
            self._pending_change_gain_base_id = None
            if scene == "produce_event":
                self._shop_clicked_slots.clear()
                self._shop_last_clicked_feature = None
                self._shop_last_points = None
                self._shop_attempted_drink_ids.clear()
                self._shop_refresh_count = 0
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
            return self._decide_produce_schedule(legal_actions, obs)
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
