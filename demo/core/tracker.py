"""
Cross-turn game state tracker for Gakumas battle stage.
"""

from collections import Counter
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from demo.schemas import ScreenObservation


class GameStateTracker:
    """
    Maintains cross-turn and cross-week state during a stage battle or produce run:
    - Persists produce_deck across produce weeks (adds/upgrades/removals + auto hand sync)
    - Tracks multi-copy single-use card exhaustion (limit_count=1) & battle-generated cards
    - Tracks turns, card uses, survival stats, and remaining draw pool
    """

    def __init__(self, plan: str = "sense", p_idol_id: Optional[int] = None, scenario_id: str = "hajime_regular"):
        self.plan: str = plan.lower()
        self.p_idol_id: Optional[int] = p_idol_id
        self.scenario_id: str = (scenario_id or "hajime_regular").lower()
        self.produce_deck: List[int] = []
        self.equipped_p_items: List[Dict[str, Any]] = []
        self.card_memory_hooks: Dict[int, Dict[str, Any]] = {}
        self.card_customizations: Dict[int, Dict[str, Any]] = {}
        self.memory_slots: List[Dict[str, Any]] = []
        self.memory_passives: List[Dict[str, Any]] = []
        self.recommended_effect: Optional[str] = None
        self.initial_loadout_captured: bool = False
        self.produce_library_log: List[str] = []

        self.used_limit_cards: Set[int] = set()
        self.battle_played_counts: Dict[int, int] = {}
        self.battle_exhausted_counts: Dict[int, int] = {}
        self.battle_generated_cards: List[int] = []
        self.pending_top_deck: List[int] = []
        self.signature_genki_trigger_active: bool = False
        self.signature_genki_counter: int = 0
        self.good_impression_effect_buff: float = 0.0
        self.signature_gi_recast_counter: int = 0
        self.signature_gi_recasts_remaining: int = 2
        self.signature_gi_recast_used_turn: bool = False
        self.in_battle: bool = False

        self.turn: int = 1
        self.turns_remaining: int = 10
        self.card_uses_remaining: int = 1
        self.stamina: int = 30
        self.max_stamina: int = 30
        self.genki: int = 0
        self.score: int = 0
        self.p_points: int = 0
        self.idol_stats: Dict[str, int] = {}
        self.buffs: Dict[str, int] = {}
        self.cards_played_history: List[Dict[str, Any]] = []
        self.drinks_used_history: List[int] = []
        self.shield_history: List[str] = []

        # Per-battle four-zone deck state (Hand / Draw Pile / Discard Pile / Exhausted)
        self.battle_hand: List[int] = []
        self.battle_draw_pile: List[int] = []
        self.battle_discard_pile: List[int] = []
        self.battle_exhaust_pile: List[int] = []
        self.battle_reshuffle_count: int = 0
        self.battle_zone_events: List[str] = []
        self._battle_tracked_turn: Optional[int] = None
        self._battle_tracked_turns_remaining: Optional[int] = None

        # Single-produce persistent LLM session sync registry
        self.session_bootstrapped: bool = False
        self.llm_synced_card_ids: Set[int] = set()
        self.llm_synced_drink_ids: Set[int] = set()
        self.llm_synced_p_items: Set[str] = set()
        self.llm_last_deck_snapshot: List[int] = []
        self.llm_last_week: Optional[int] = None
        self.llm_last_scene: Optional[str] = None
        self.llm_last_in_battle: bool = False
        self.llm_pending_events: List[str] = []

    def init_produce_deck(
        self,
        db: Optional[Any] = None,
        plan: Optional[str] = None,
        p_idol_id: Optional[int] = None,
        scenario_id: str = "hajime_regular",
    ) -> List[int]:
        """Initialize produce_deck from starter templates (or empty base in H.I.F 本戦 memory mode) & signature cards."""
        from demo.core.scenarios import (
            IDOL_EXTRA_SIGNATURE_CARDS,
            IDOL_SIGNATURE_START_CARD,
            STARTER_DECK_BY_PLAN,
        )

        active_plan = (plan or self.plan or "logic").lower()
        self.plan = active_plan
        self.scenario_id = (scenario_id or self.scenario_id or "hajime_regular").lower()
        if p_idol_id is not None:
            self.p_idol_id = p_idol_id

        is_hif = (scenario_id or "").lower() in ("hif_honsen", "hif")
        deck = [] if is_hif else list(STARTER_DECK_BY_PLAN.get(active_plan, STARTER_DECK_BY_PLAN["sense"]))

        sig_ids: List[int] = []
        if self.p_idol_id is not None:
            if self.p_idol_id in IDOL_EXTRA_SIGNATURE_CARDS:
                sig_ids.extend(IDOL_EXTRA_SIGNATURE_CARDS[self.p_idol_id])
            elif self.p_idol_id in IDOL_SIGNATURE_START_CARD:
                sig_ids.append(IDOL_SIGNATURE_START_CARD[self.p_idol_id])
            elif db is not None:
                try:
                    for c in db.get_cards_for_plan(active_plan):
                        if c.get("p_idol_id") == self.p_idol_id and c.get("is_signature") == 1 and c.get("upgraded") == 0:
                            sig_ids.append(int(c["id"]))
                except Exception:
                    pass
        elif active_plan == "logic" and not is_hif:
            sig_ids.append(IDOL_SIGNATURE_START_CARD[126])

        for sig_id in sig_ids:
            if sig_id not in deck:
                deck.append(sig_id)
        self.produce_deck = deck
        return list(self.produce_deck)

    def set_initial_loadout(
        self,
        *,
        deck_ids: List[int],
        p_items: List[Dict[str, Any]],
        p_idol_id: Optional[int] = None,
        plan: Optional[str] = None,
        recommended_effect: Optional[str] = None,
        card_memory_hooks: Optional[Dict[int, Dict[str, Any]]] = None,
        card_customizations: Optional[Dict[Any, Dict[str, Any]]] = None,
        memory_slots: Optional[List[Dict[str, Any]]] = None,
        memory_passives: Optional[List[Dict[str, Any]]] = None,
        db: Optional[Any] = None,
    ) -> None:
        """Persist scanned initial cards, P-items, and idol effect preferences for this produce run."""
        if plan:
            self.plan = str(plan).lower()
        if p_idol_id is not None:
            self.p_idol_id = int(p_idol_id)
        self.produce_deck = [int(cid) for cid in deck_ids if cid]
        self.equipped_p_items = [dict(it) for it in (p_items or [])]
        self.card_memory_hooks = {
            int(k): dict(v) for k, v in (card_memory_hooks or {}).items()
        }
        if card_customizations is not None:
            self.card_customizations = {
                int(k): dict(v)
                for k, v in card_customizations.items()
                if isinstance(v, dict)
            }
        if memory_slots is not None:
            self.memory_slots = [dict(s) for s in memory_slots if isinstance(s, dict)]
        if memory_passives is not None:
            self.memory_passives = [dict(p) for p in memory_passives if isinstance(p, dict)]
        resolved_rec = recommended_effect
        if resolved_rec is None and self.p_idol_id is not None and db is not None:
            try:
                pid_row = (
                    db.get_p_idol_by_id(self.p_idol_id)
                    if hasattr(db, "get_p_idol_by_id")
                    else None
                )
                if pid_row and pid_row.get("recommended_effect"):
                    resolved_rec = str(pid_row["recommended_effect"])
            except Exception:
                pass
        self.recommended_effect = resolved_rec
        self.initial_loadout_captured = True
        self.produce_library_log.append(
            f"Captured loadout: p_idol={self.p_idol_id}, plan={self.plan}, "
            f"rec={self.recommended_effect}, deck={len(self.produce_deck)}, p_items={len(self.equipped_p_items)}"
        )

    def merge_manual_memory_hooks(
        self,
        hooks: Optional[Dict[Any, Dict[str, Any]]] = None,
        memory_slots: Optional[List[Dict[str, Any]]] = None,
        memory_passives: Optional[List[Dict[str, Any]]] = None,
        card_customizations: Optional[Dict[Any, Dict[str, Any]]] = None,
    ) -> None:
        """Merge user-entered memory hooks, customizations, and passive slots onto scanned initial loadout."""
        for k, v in (hooks or {}).items():
            if not isinstance(v, dict):
                continue
            try:
                cid = int(k)
            except (TypeError, ValueError):
                continue
            merged = dict(self.card_memory_hooks.get(cid) or {})
            merged.update(v)
            self.card_memory_hooks[cid] = merged
        for k, v in (card_customizations or {}).items():
            if not isinstance(v, dict):
                continue
            try:
                cid = int(k)
            except (TypeError, ValueError):
                continue
            merged_c = dict(self.card_customizations.get(cid) or {})
            merged_c.update(v)
            self.card_customizations[cid] = merged_c
        if memory_slots:
            self.memory_slots = [dict(s) for s in memory_slots if isinstance(s, dict)]
        if memory_passives:
            self.memory_passives = [dict(p) for p in memory_passives if isinstance(p, dict)]

    def export_produce_library(self, db: Optional[Any] = None) -> Dict[str, Any]:
        """Return a structured snapshot of this run's initial + current cards, P-items, and idol metadata."""
        card_entries: List[Dict[str, Any]] = []
        for cid in self.produce_deck:
            row = db.get_card_by_id(cid) if db is not None else None
            card_entries.append({
                "card_id": int(cid),
                "name": str(row["name"]) if row else f"Card#{cid}",
                "card_type": str(row.get("type") or "") if row else "",
                "rarity": str(row.get("rarity") or "") if row else "",
                "upgraded": bool(row.get("upgraded")) if row else False,
                "memory_hooks": dict(self.card_memory_hooks.get(int(cid), {})),
                "customizations": dict(self.card_customizations.get(int(cid), {})),
            })
        return {
            "p_idol_id": self.p_idol_id,
            "plan": self.plan,
            "scenario_id": self.scenario_id,
            "recommended_effect": self.recommended_effect,
            "initial_loadout_captured": self.initial_loadout_captured,
            "p_points": self.p_points,
            "idol_stats": dict(self.idol_stats),
            "deck_count": len(self.produce_deck),
            "deck_ids": list(self.produce_deck),
            "cards": card_entries,
            "p_items": [dict(it) for it in self.equipped_p_items],
            "card_memory_hooks": {str(k): dict(v) for k, v in self.card_memory_hooks.items()},
            "card_customizations": {str(k): dict(v) for k, v in self.card_customizations.items()},
            "memory_slots": [dict(s) for s in self.memory_slots],
            "memory_passives": [dict(p) for p in self.memory_passives],
            "log": list(self.produce_library_log),
        }

    def sanitize_produce_deck(self, db: Optional[Any] = None) -> None:
        """Remove temporary battle-generated token cards (e.g. #757/#758) from persistent produce_deck."""
        if not self.produce_deck:
            return
        cleaned: List[int] = []
        for cid in self.produce_deck:
            cid_i = int(cid)
            if cid_i in (757, 758):
                continue
            if db is not None:
                try:
                    row = db.get_card_by_id(cid_i)
                    if row and row.get("base_id") == 757:
                        continue
                except Exception:
                    pass
            cleaned.append(cid_i)
        self.produce_deck = cleaned

    def has_card_family(self, card_id: int, db: Optional[Any] = None) -> bool:
        if not card_id:
            return False
        target_fam = self._family_id(int(card_id), db=db)
        return any(self._family_id(int(cid), db=db) == target_fam for cid in self.produce_deck)

    def add_card_to_deck(self, card_id: int, db: Optional[Any] = None) -> bool:
        if not card_id:
            return False
        cid_int = int(card_id)
        if db is not None:
            row = db.get_card_by_id(cid_int)
            is_unique_card = bool(
                row is not None
                and (
                    row.get("is_unique_in_deck")
                    or row.get("is_unique")
                    or ("重複不可" in str(row.get("description") or ""))
                )
            )
            if is_unique_card and self.has_card_family(cid_int, db=db):
                return False
        self.produce_deck.append(cid_int)
        return True

    def remove_card_from_deck(self, card_id: int) -> bool:
        for i, cid in enumerate(self.produce_deck):
            if cid == card_id:
                del self.produce_deck[i]
                return True
        return False

    def upgrade_card_in_deck(self, old_card_id: int, new_card_id: int) -> bool:
        for i, cid in enumerate(self.produce_deck):
            if cid == old_card_id:
                self.produce_deck[i] = int(new_card_id)
                if int(old_card_id) in self.card_customizations and int(new_card_id) not in self.card_customizations:
                    self.card_customizations[int(new_card_id)] = dict(self.card_customizations[int(old_card_id)])
                return True
        return False

    def _family_id(self, card_id: int, base_id: Optional[int] = None, db: Optional[Any] = None) -> int:
        if base_id:
            return int(base_id)
        if db is not None:
            try:
                row = db.get_card_by_id(card_id)
                if row and row.get("base_id"):
                    return int(row["base_id"])
            except Exception:
                pass
        return int(card_id)

    def _total_battle_copies(self, card_id: int, base_id: Optional[int] = None, db: Optional[Any] = None) -> int:
        fam = self._family_id(card_id, base_id=base_id, db=db)
        total = 0
        for cid in self.produce_deck + self.battle_generated_cards:
            if cid == card_id or (base_id and cid == base_id) or self._family_id(cid, db=db) == fam:
                total += 1
        return total

    def _card_label(self, card_id: int, db: Optional[Any] = None) -> str:
        row = db.get_card_by_id(card_id) if db is not None else None
        name = (row.get("name") if row else None) or f"#{card_id}"
        return f"《{name}》(#{card_id})"

    def _record_zone_event(self, note: str) -> None:
        clean = (note or "").strip()
        if not clean:
            return
        if not self.battle_zone_events or self.battle_zone_events[-1] != clean:
            self.battle_zone_events.append(clean)
        if len(self.battle_zone_events) > 12:
            self.battle_zone_events = self.battle_zone_events[-12:]

    def _zones_empty(self) -> bool:
        return not (
            self.battle_hand
            or self.battle_draw_pile
            or self.battle_discard_pile
            or self.battle_exhaust_pile
        )

    def _remove_one_card(
        self,
        pile: List[int],
        card_id: int,
        base_id: Optional[int] = None,
        db: Optional[Any] = None,
    ) -> Optional[int]:
        """Remove and return 1 card from `pile` preferring exact ID, then base_id, then card family."""
        for idx, cid in enumerate(pile):
            if cid == card_id:
                return pile.pop(idx)
        if base_id is not None:
            for idx, cid in enumerate(pile):
                if cid == base_id:
                    return pile.pop(idx)
        target_fam = self._family_id(card_id, base_id=base_id, db=db)
        for idx, cid in enumerate(pile):
            if self._family_id(cid, db=db) == target_fam:
                return pile.pop(idx)
        return None

    @staticmethod
    def _replace_one_card(pile: List[int], old_id: int, new_id: int) -> bool:
        for idx, cid in enumerate(pile):
            if cid == old_id:
                pile[idx] = int(new_id)
                return True
        return False

    def _replace_in_battle_zones(self, old_id: int, new_id: int) -> bool:
        for zone in (self.battle_hand, self.battle_draw_pile, self.battle_discard_pile):
            if self._replace_one_card(zone, old_id, new_id):
                return True
        return False

    def _ensure_battle_zones_initialized(self, db: Optional[Any] = None) -> None:
        """Lazily initialize the 4 battle zones from produce_deck when empty."""
        if not self.produce_deck or not self._zones_empty():
            return
        draw = list(self.produce_deck)
        exhaust: List[int] = []
        if self.battle_exhausted_counts:
            for fam, ex_n in self.battle_exhausted_counts.items():
                for _ in range(max(0, int(ex_n))):
                    removed = self._remove_one_card(draw, fam, base_id=fam, db=db)
                    if removed is not None:
                        exhaust.append(removed)
        for gid in self.battle_generated_cards:
            if gid not in draw and gid not in exhaust:
                draw.insert(0, gid)
        self.battle_draw_pile = draw
        self.battle_exhaust_pile = exhaust
        self.battle_hand.clear()
        self.battle_discard_pile.clear()
        self.battle_reshuffle_count = 0

    def _reconcile_deck_additions_into_battle(self, db: Optional[Any] = None) -> None:
        """Ensure cards appended to produce_deck outside start_new_battle appear in battle_draw_pile."""
        if not self.produce_deck:
            return
        all_zone_cards = (
            list(self.battle_hand)
            + list(self.battle_draw_pile)
            + list(self.battle_discard_pile)
            + list(self.battle_exhaust_pile)
        )
        tracked_fams = Counter(self._family_id(cid, db=db) for cid in all_zone_cards)
        for gid in self.battle_generated_cards:
            gfam = self._family_id(gid, db=db)
            if tracked_fams.get(gfam, 0) > 0:
                tracked_fams[gfam] -= 1
        deck_fams = Counter(self._family_id(cid, db=db) for cid in self.produce_deck)
        missing_fams = deck_fams - tracked_fams
        if not missing_fams:
            return
        for cid in self.produce_deck:
            fam = self._family_id(cid, db=db)
            if missing_fams.get(fam, 0) > 0:
                self.battle_draw_pile.append(cid)
                missing_fams[fam] -= 1

    def _reshuffle_discard_into_draw(self) -> int:
        """Transfer all cards from Discard Pile into Draw Pile only after Draw Pile empties."""
        if not self.battle_discard_pile:
            return 0
        moved = len(self.battle_discard_pile)
        self.battle_draw_pile.extend(self.battle_discard_pile)
        self.battle_discard_pile.clear()
        self.battle_reshuffle_count += 1
        self._record_zone_event(
            f"【牌堆抽空触发重洗(第{self.battle_reshuffle_count}次)】将【弃牌】中 {moved} 张牌洗回【牌堆】"
        )
        return moved

    def _is_same_hand_with_upgrades(
        self,
        prev_hand: List[int],
        new_hand: List[int],
        db: Optional[Any] = None,
    ) -> bool:
        """Detect in-battle hand upgrades (e.g. フレッシュビネガー) without false turn transitions."""
        if not prev_hand or len(prev_hand) != len(new_hand) or prev_hand == new_hand:
            return False
        prev_fams = Counter(self._family_id(cid, db=db) for cid in prev_hand)
        new_fams = Counter(self._family_id(cid, db=db) for cid in new_hand)
        if prev_fams != new_fams:
            return False
        for cid in new_hand:
            if cid not in prev_hand and db is not None:
                row = db.get_card_by_id(cid)
                if row and row.get("upgraded") == 1:
                    return True
        return False

    def _sync_draw_cards(self, needed_cids: List[int], db: Optional[Any] = None) -> None:
        """Draw `needed_cids` from Draw Pile first; only reshuffle Discard -> Draw once Draw Pile empties."""
        if not needed_cids:
            return
        remaining: List[int] = []
        for cid in needed_cids:
            row = db.get_card_by_id(cid) if db is not None else None
            base_id = int(row["base_id"]) if (row and row.get("base_id")) else None
            if self._remove_one_card(self.battle_draw_pile, cid, base_id=base_id, db=db) is None:
                remaining.append(cid)

        if remaining and self.battle_discard_pile:
            discard_fams = {self._family_id(dcid, db=db) for dcid in self.battle_discard_pile}
            needs_discard_reshuffle = (len(self.battle_draw_pile) == 0) or any(
                self._family_id(rcid, db=db) in discard_fams for rcid in remaining
            )
            if needs_discard_reshuffle:
                if self.battle_draw_pile:
                    leftover_n = len(self.battle_draw_pile)
                    self.battle_discard_pile.extend(self.battle_draw_pile)
                    self.battle_draw_pile.clear()
                    self._record_zone_event(f"前序牌堆余牌({leftover_n}张)归入【弃牌】并抽空")
                self._reshuffle_discard_into_draw()

                still_missing: List[int] = []
                for cid in remaining:
                    row = db.get_card_by_id(cid) if db is not None else None
                    base_id = int(row["base_id"]) if (row and row.get("base_id")) else None
                    if self._remove_one_card(self.battle_draw_pile, cid, base_id=base_id, db=db) is None:
                        still_missing.append(cid)
                remaining = still_missing

        # Token cards inserted straight into battle_generated_cards in tests need no error

    def end_current_turn_hand(
        self,
        db: Optional[Any] = None,
        reason: str = "回合结束未打出",
    ) -> List[int]:
        """Move all remaining unplayed cards in Hand into Discard Pile."""
        if not self.battle_hand:
            return []
        moved = list(self.battle_hand)
        self.battle_discard_pile.extend(moved)
        self.battle_hand.clear()
        labels = "、".join(self._card_label(cid, db=db) for cid in moved)
        self._record_zone_event(f"{reason}：{labels} ({len(moved)}张) → 进入【弃牌】")
        return moved

    def record_turn_end(self, db: Optional[Any] = None) -> List[int]:
        """Record an explicit End Turn / SKIP action, discarding remaining hand cards."""
        moved = self.end_current_turn_hand(db=db, reason="结束回合，手牌未打出")
        self.card_uses_remaining = 0
        self.signature_gi_recast_used_turn = False
        for tk in ("halfCostTurns", "doubleCostTurns", "goodConditionTurns", "goodImpressionTurns"):
            if int(self.buffs.get(tk, 0) or 0) > 0:
                self.buffs[tk] = max(0, int(self.buffs[tk]) - 1)
        return moved

    def _record_shield_event(self, note: str) -> None:
        clean = (note or "").strip()
        if not clean:
            return
        self.shield_history.append(clean)
        if len(self.shield_history) > 8:
            self.shield_history = self.shield_history[-8:]

    def _restore_limit_card_if_still_in_hand(
        self,
        obs_cards: List[Any],
        db: Optional[Any] = None,
    ) -> None:
        """Self-heal if a limit_count=1 card (e.g. #757) failed to register on device and stayed in hand."""
        obs_by_fam: Dict[int, List[Tuple[int, Optional[int]]]] = {}
        for c in obs_cards:
            cid = int(getattr(c, "card_id", 0) or 0)
            if not cid:
                continue
            row = db.get_card_by_id(cid) if db is not None else None
            limit_cnt = row.get("limit_count") if row else None
            if limit_cnt != 1:
                continue
            base_id = int(row["base_id"]) if (row and row.get("base_id")) else None
            fam = self._family_id(cid, base_id=base_id, db=db)
            obs_by_fam.setdefault(fam, []).append((cid, base_id))

        for fam, pairs in obs_by_fam.items():
            seen_fam_n = len(pairs)
            cid, base_id = pairs[0]
            cid_row = db.get_card_by_id(cid) if db is not None else None
            total_copies = self._total_battle_copies(cid, base_id=base_id, db=db) if (self.produce_deck or fam in (757, 758)) else 1
            ex_n = int(self.battle_exhausted_counts.get(fam, 0) or 0)
            available = max(0, total_copies - ex_n)
            while seen_fam_n > available and ex_n > 0:
                restored = self._remove_one_card(self.battle_exhaust_pile, cid, base_id=base_id, db=db)
                if restored is None:
                    break
                self.battle_draw_pile.insert(0, restored)
                ex_n -= 1
                self.battle_exhausted_counts[fam] = ex_n
                if fam == 757 and restored in (757, 758):
                    self.used_limit_cards.discard(757)
                    self.used_limit_cards.discard(758)
                else:
                    self.used_limit_cards.discard(cid)
                    self.used_limit_cards.discard(restored)
                    if base_id is not None:
                        self.used_limit_cards.discard(base_id)
                    self.used_limit_cards.discard(fam)
                    if cid_row and cid_row.get("upgraded_id"):
                        self.used_limit_cards.discard(int(cid_row["upgraded_id"]))
                available = max(0, total_copies - ex_n)
                self._record_zone_event(
                    f"检测到{self._card_label(restored, db=db)}仍在手牌（上步未真正打出），已从【除外】恢复可用"
                )
            if seen_fam_n > 0 and ex_n == 0 and available >= seen_fam_n:
                self.used_limit_cards.discard(cid)
                if base_id is not None:
                    self.used_limit_cards.discard(base_id)
                self.used_limit_cards.discard(fam)
                if cid_row and cid_row.get("upgraded_id"):
                    self.used_limit_cards.discard(int(cid_row["upgraded_id"]))

    def sync_observed_hand(
        self,
        obs_cards: List[Any],
        db: Optional[Any] = None,
        turn: Optional[int] = None,
        turns_remaining: Optional[int] = None,
    ) -> None:
        """Keep produce_deck & per-battle four-zone state aligned with cards observed on screen."""
        self._ensure_battle_zones_initialized(db=db)
        if not obs_cards:
            return
        self._reconcile_deck_additions_into_battle(db=db)
        self._restore_limit_card_if_still_in_hand(obs_cards, db=db)

        obs_cids: List[int] = [int(c.card_id) for c in obs_cards if getattr(c, "card_id", None)]
        first_obs = (
            self._battle_tracked_turn is None
            and self._battle_tracked_turns_remaining is None
            and not self.battle_hand
        )
        turn_changed = False
        if not first_obs:
            if turn is not None and self._battle_tracked_turn is not None and int(turn) != self._battle_tracked_turn:
                turn_changed = True
            elif (
                turns_remaining is not None
                and self._battle_tracked_turns_remaining is not None
                and int(turns_remaining) != self._battle_tracked_turns_remaining
            ):
                turn_changed = True
            elif (
                self.card_uses_remaining <= 0
                and Counter(obs_cids) != Counter(self.battle_hand)
                and not self._is_same_hand_with_upgrades(self.battle_hand, obs_cids, db)
            ):
                turn_changed = True

        # In-battle temporary hand upgrade (e.g. フレッシュビネガー): update battle_hand in place without altering produce_deck
        if (
            self.produce_deck
            and not first_obs
            and not turn_changed
            and self._is_same_hand_with_upgrades(self.battle_hand, obs_cids, db)
        ):
            self.battle_hand = list(obs_cids)
            self._record_zone_event("【手牌】触发局内强化（未消耗/未进弃牌）")
            if turn is not None:
                self._battle_tracked_turn = int(turn)
            if turns_remaining is not None:
                self._battle_tracked_turns_remaining = int(turns_remaining)
            return

        hand_counts = Counter(obs_cids)
        for cid, seen_n in hand_counts.items():
            # Ignore temporary battle-generated token cards (e.g. 757/758) when checking produce_deck additions
            row = db.get_card_by_id(cid) if db is not None else None
            base_id = int(row["base_id"]) if (row and row.get("base_id")) else None
            if cid in (757, 758) or base_id == 757:
                # Consume matching pending top-deck entries once drawn into hand
                while cid in self.pending_top_deck:
                    self.pending_top_deck.remove(cid)
                while base_id and base_id in self.pending_top_deck:
                    self.pending_top_deck.remove(base_id)
                exhausted_tokens = sum(1 for x in self.battle_exhaust_pile if x in (cid, base_id))
                available_tokens = max(0, sum(1 for x in self.battle_generated_cards if x in (cid, base_id)) - exhausted_tokens)
                if available_tokens < seen_n:
                    self.battle_generated_cards.append(cid)
                    self.battle_draw_pile.insert(0, cid)
                continue

            if not self.produce_deck:
                continue

            fam = self._family_id(cid, base_id=base_id, db=db)
            fam_total = sum(1 for d in self.produce_deck if self._family_id(d, db=db) == fam)
            fam_exhausted = int(self.battle_exhausted_counts.get(fam, 0) or 0)
            if fam_exhausted > 0 and (fam_total - fam_exhausted) < seen_n:
                recover_n = min(fam_exhausted, seen_n - max(0, fam_total - fam_exhausted))
                recovered = 0
                for _ in range(recover_n):
                    popped = self._remove_one_card(self.battle_exhaust_pile, cid, base_id=base_id, db=db)
                    if popped is not None:
                        self.battle_draw_pile.append(popped)
                    recovered += 1
                self.battle_exhausted_counts[fam] = max(0, fam_exhausted - recover_n)
                self.used_limit_cards.discard(cid)
                self.used_limit_cards.discard(fam)
                if base_id is not None:
                    self.used_limit_cards.discard(base_id)
                if row and row.get("upgraded_id"):
                    self.used_limit_cards.discard(int(row["upgraded_id"]))
                if recovered > 0:
                    self._record_zone_event(f"检测到{self._card_label(cid, db=db)}仍在手牌，撤销误判除外记录")

            zone_pool = Counter(self.battle_hand) + Counter(self.battle_draw_pile) + Counter(self.battle_discard_pile)
            if zone_pool.get(cid, 0) >= seen_n:
                continue

            # Keep unupgraded/upgraded variants aligned without ever downgrading upgraded cards from battle OCR
            if row and row.get("upgraded") == 1 and base_id and cid not in self.produce_deck:
                if self.upgrade_card_in_deck(base_id, cid):
                    self._replace_in_battle_zones(base_id, cid)
                    continue
            if row and row.get("upgraded") == 0 and row.get("upgraded_id"):
                upg_id_i = int(row["upgraded_id"])
                if zone_pool.get(cid, 0) + zone_pool.get(upg_id_i, 0) >= seen_n:
                    continue
                if upg_id_i in self.produce_deck and fam_total >= seen_n:
                    continue

            total_known = sum(1 for d in self.produce_deck if self._family_id(d, db=db) == fam)
            exhausted_n = int(self.battle_exhausted_counts.get(fam, 0) or 0)
            available_known = max(0, total_known - exhausted_n)
            is_unique_card = bool(
                row is not None
                and (
                    row.get("is_unique_in_deck")
                    or row.get("is_unique")
                    or ("重複不可" in str(row.get("description") or ""))
                )
            )
            while seen_n > available_known:
                if is_unique_card and self.has_card_family(cid, db=db):
                    break
                self.produce_deck.append(cid)
                self.battle_draw_pile.append(cid)
                available_known += 1

        if not self.produce_deck:
            self.battle_hand = list(obs_cids)
            if turn is not None:
                self._battle_tracked_turn = int(turn)
            if turns_remaining is not None:
                self._battle_tracked_turns_remaining = int(turns_remaining)
            return

        if first_obs:
            self._sync_draw_cards(obs_cids, db=db)
            self.battle_hand = list(obs_cids)
            drawn_lbl = "、".join(self._card_label(cid, db=db) for cid in obs_cids)
            self._record_zone_event(f"起手从【牌堆】抽取 {len(obs_cids)} 张至【手牌】：{drawn_lbl}")
        elif turn_changed:
            self.end_current_turn_hand(db=db, reason="上回合结束，未打出手牌")
            self._sync_draw_cards(obs_cids, db=db)
            self.battle_hand = list(obs_cids)
            drawn_lbl = "、".join(self._card_label(cid, db=db) for cid in obs_cids)
            self._record_zone_event(f"新回合从【牌堆】抽取 {len(obs_cids)} 张至【手牌】：{drawn_lbl}")
        else:
            if self._is_same_hand_with_upgrades(self.battle_hand, obs_cids, db):
                self.battle_hand = list(obs_cids)
                self._record_zone_event("【手牌】触发局内强化（未消耗/未进弃牌）")
            elif Counter(obs_cids) != Counter(self.battle_hand):
                cur_cnt = Counter(self.battle_hand)
                obs_cnt = Counter(obs_cids)
                gone_cids: List[int] = []
                for cid, cnt in (cur_cnt - obs_cnt).items():
                    gone_cids.extend([cid] * cnt)
                added_cids: List[int] = []
                for cid, cnt in (obs_cnt - cur_cnt).items():
                    added_cids.extend([cid] * cnt)
                if gone_cids:
                    if self.scenario_id in ("hif_honsen", "hif") and 23 in gone_cids:
                        ex_trouble: List[int] = []
                        other_gone: List[int] = []
                        for gcid in gone_cids:
                            if gcid == 23 and (sum(1 for x in self.battle_exhaust_pile if x == 23) + len(ex_trouble)) < self._total_battle_copies(23, base_id=23, db=db):
                                ex_trouble.append(gcid)
                            else:
                                other_gone.append(gcid)
                        if ex_trouble:
                            self.battle_exhaust_pile.extend(ex_trouble)
                            self.battle_exhausted_counts[23] = self.battle_exhausted_counts.get(23, 0) + len(ex_trouble)
                            self.used_limit_cards.add(23)
                            self._record_zone_event(
                                f"同回合メモリー効果将《眠気》(#23) ×{len(ex_trouble)} 移入【除外】"
                            )
                        if other_gone:
                            self.battle_discard_pile.extend(other_gone)
                            gone_lbl = "、".join(self._card_label(cid, db=db) for cid in other_gone)
                            self._record_zone_event(f"同回合置换手牌 {gone_lbl} ({len(other_gone)}张) → 进入【弃牌】")
                    else:
                        self.battle_discard_pile.extend(gone_cids)
                        gone_lbl = "、".join(self._card_label(cid, db=db) for cid in gone_cids)
                        self._record_zone_event(f"同回合置换手牌 {gone_lbl} ({len(gone_cids)}张) → 进入【弃牌】")
                if added_cids:
                    self._sync_draw_cards(added_cids, db=db)
                    add_lbl = "、".join(self._card_label(cid, db=db) for cid in added_cids)
                    self._record_zone_event(f"同回合从【牌堆】补抽 {len(added_cids)} 张至【手牌】：{add_lbl}")
                self.battle_hand = list(obs_cids)

        if turn is not None:
            self._battle_tracked_turn = int(turn)
        if turns_remaining is not None:
            self._battle_tracked_turns_remaining = int(turns_remaining)

    def record_card_play(
        self,
        card_id: int,
        limit_count: Optional[int] = None,
        base_id: Optional[int] = None,
        dsl_actions: str = "",
        db: Optional[Any] = None,
        dsl_cost: Optional[str] = None,
    ) -> None:
        """
        Records a played card.
        Tracks per-family limit_count=1 exhaustion (Discard vs Exhausted zones),
        repeatable cards entering Discard Pile, backend Shield/HP ledger, and top-deck generation.
        """
        self._ensure_battle_zones_initialized(db=db)
        row = db.get_card_by_id(card_id) if db is not None else None
        if limit_count is None and row is not None:
            limit_count = row.get("limit_count")
        if base_id is None and row is not None:
            base_id = row.get("base_id")
        cust_meta = self.card_customizations.get(int(card_id)) or (
            self.card_customizations.get(int(base_id)) if base_id else None
        )
        if isinstance(cust_meta, dict):
            mods = cust_meta.get("modifiers") if isinstance(cust_meta.get("modifiers"), dict) else cust_meta
            lvls = cust_meta.get("levels") if isinstance(cust_meta.get("levels"), dict) else {}
            if bool(mods.get("remove_limit")) or int(lvls.get("37", lvls.get(37, 0)) or 0) > 0:
                limit_count = None
            elif lvls and db is not None and hasattr(db, "get_customization_by_id"):
                for ck, cv in lvls.items():
                    if int(cv or 0) > 0:
                        crow = db.get_customization_by_id(int(ck))
                        if crow is not None and crow.get("limit_count") == 0:
                            limit_count = None
                            break
        if not dsl_actions and row is not None:
            dsl_actions = row.get("dsl_actions") or ""
        if dsl_cost is None and row is not None:
            dsl_cost = row.get("dsl_cost") or ""
        dsl_cost = (dsl_cost or "").strip()

        from demo.core.idol_guides import get_memory_card_custom_meta

        fam = self._family_id(card_id, base_id=base_id, db=db)
        prev_fam_plays = self.battle_played_counts.get(fam, 0)
        custom_meta = get_memory_card_custom_meta(
            self.scenario_id,
            self.p_idol_id,
            card_id,
            battle_played_counts={fam: prev_fam_plays},
        )
        self.battle_played_counts[fam] = prev_fam_plays + 1

        # Move played card out of Hand (or Draw Pile fallback) into Exhausted (limit=1) or Discard (reusable)
        removed_cid = self._remove_one_card(self.battle_hand, card_id, base_id=base_id, db=db)
        if removed_cid is None and self.produce_deck:
            removed_cid = self._remove_one_card(self.battle_draw_pile, card_id, base_id=base_id, db=db)
        played_cid = removed_cid if removed_cid is not None else int(card_id)

        if limit_count == 1:
            if self.produce_deck or removed_cid is not None or self.in_battle:
                self.battle_exhaust_pile.append(played_cid)
                self._record_zone_event(f"打出 {self._card_label(played_cid, db=db)} [限1] → 进入【除外】(本场不可再用)")
        else:
            if self.produce_deck or removed_cid is not None or self.in_battle:
                self.battle_discard_pile.append(played_cid)
                self._record_zone_event(f"打出 {self._card_label(played_cid, db=db)} [可重复] → 进入【弃牌】")

        # H.I.F memory ability: free exile 1 Trouble (眠気 #23) BEFORE evaluating countCards[removed] thresholds!
        if custom_meta.get("exiles_trouble"):
            for zone_pile in (self.battle_hand, self.battle_draw_pile, self.battle_discard_pile):
                exiled_t = self._remove_one_card(zone_pile, 23, base_id=23, db=db)
                if exiled_t is not None:
                    self.battle_exhaust_pile.append(exiled_t)
                    self.battle_exhausted_counts[23] = self.battle_exhausted_counts.get(23, 0) + 1
                    self.used_limit_cards.add(23)
                    self._record_zone_event("触发メモリー効果：免费将《眠気》(#23)移动至【除外】")
                    break

        # Signature trigger: 私を超えて (#755/#756) spawns #757 every 2 Genki cards used (starts on 1st)
        if card_id in (755, 756) or fam == 755 or "addCardToTopOfDeck(757)" in (dsl_actions or ""):
            self.signature_genki_trigger_active = True
            self.signature_genki_counter = 0
        elif self.signature_genki_trigger_active and "genki+=" in (dsl_actions or "").lower():
            self.signature_genki_counter += 1
            if self.signature_genki_counter % 2 == 1:
                self.battle_generated_cards.append(757)
                self.pending_top_deck.append(757)
                self.battle_draw_pile.insert(0, 757)
                self._record_zone_event("衍生生成《私を超えて(翔)》(#757) → 置入【牌堆】顶")

        if card_id in self.pending_top_deck:
            self.pending_top_deck.remove(card_id)
        elif fam in self.pending_top_deck:
            self.pending_top_deck.remove(fam)

        if limit_count == 1:
            self.battle_exhausted_counts[fam] = self.battle_exhausted_counts.get(fam, 0) + 1
            total_copies = self._total_battle_copies(card_id, base_id=base_id, db=db) if self.produce_deck else 1
            if not self.produce_deck or self.battle_exhausted_counts[fam] >= max(1, total_copies):
                self.used_limit_cards.add(card_id)
                if base_id is not None:
                    self.used_limit_cards.add(base_id)

        if custom_meta.get("grant_extra_use_now"):
            self._record_zone_event("触发《やる気は満点》メモリー効果：首次打出免费再动(+1使用次数)")
        elif "cardusesremaining+=" not in (dsl_actions or "").lower():
            self.card_uses_remaining = max(0, self.card_uses_remaining - 1)

        # Simulate backend Shield (元気) / Green-HP / Red-HP & Motivation / Good Impression ledger
        shield_before = self.genki
        hp_before = self.stamina
        if dsl_cost:
            m_cost = re.match(r"^([a-zA-Z]+)-=(\d+)$", dsl_cost)
            if m_cost:
                cvar, camt = m_cost.group(1), int(m_cost.group(2))
                if cvar == "cost":
                    if int(self.buffs.get("nullifyCostCards", 0) or 0) > 0:
                        eff = 0
                    elif int(self.buffs.get("halfCostTurns", 0) or 0) > 0:
                        eff = math.ceil(camt / 2)
                    elif int(self.buffs.get("doubleCostTurns", 0) or 0) > 0:
                        eff = camt * 2
                    else:
                        eff = camt
                    shield_used = min(self.genki, eff)
                    hp_used = max(0, eff - self.genki)
                    self.genki = max(0, self.genki - shield_used)
                    if hp_used > 0:
                        self.stamina = max(1, self.stamina - hp_used)
                elif cvar == "stamina":
                    self.stamina = max(1, self.stamina - camt)
                elif cvar in ("motivation", "goodImpressionTurns", "goodConditionTurns", "concentration"):
                    self.buffs[cvar] = max(0, int(self.buffs.get(cvar, 0) or 0) - camt)

        # Apply instant action effects in sequence (top-level clauses & countCards[removed] bonuses)
        mot_now = int(self.buffs.get("motivation", 0) or 0)
        exhaust_n_now = len(self.battle_exhaust_pile)
        direct_gi_increases = 0

        def _add_good_impression(base_amt: int) -> None:
            nonlocal direct_gi_increases
            if base_amt <= 0:
                return
            mult = 1.0 + max(0.0, self.good_impression_effect_buff)
            gain = int(math.floor(base_amt * mult + 1e-6))
            self.buffs["goodImpressionTurns"] = int(self.buffs.get("goodImpressionTurns", 0) or 0) + max(base_amt, gain)
            direct_gi_increases += 1

        for clause in [cl.strip() for cl in (dsl_actions or "").split(";") if cl.strip()]:
            m_rem_gi = re.match(r"^if:countCards\[removed\]>=(\d+)\s*\{\s*goodImpressionTurns\+=(\d+)\s*\}$", clause)
            if m_rem_gi:
                if exhaust_n_now >= int(m_rem_gi.group(1)):
                    _add_good_impression(int(m_rem_gi.group(2)))
                continue
            if clause.startswith("at:") or "{" in clause:
                continue
            if re.match(r"^genki/=2\b", clause):
                self.genki //= 2
            elif re.match(r"^genki\+=(\d+)", clause):
                base_g = int(re.match(r"^genki\+=(\d+)", clause).group(1))
                self.genki += base_g + max(0, mot_now)
            elif re.match(r"^fixedGenki\+=(\d+)", clause):
                self.genki += int(re.match(r"^fixedGenki\+=(\d+)", clause).group(1))
            elif re.match(r"^motivation\+=(\d+)", clause):
                self.buffs["motivation"] = int(self.buffs.get("motivation", 0) or 0) + int(
                    re.match(r"^motivation\+=(\d+)", clause).group(1)
                )
            elif re.match(r"^goodImpressionTurns\+=(\d+)", clause):
                _add_good_impression(int(re.match(r"^goodImpressionTurns\+=(\d+)", clause).group(1)))
            elif re.match(r"^goodImpressionTurns\*=([\d\.]+)", clause):
                cur_gi = int(self.buffs.get("goodImpressionTurns", 0) or 0)
                fac_gi = float(re.match(r"^goodImpressionTurns\*=([\d\.]+)", clause).group(1))
                self.buffs["goodImpressionTurns"] = int(math.floor(cur_gi * fac_gi + 1e-6))
            elif re.match(r"^setGoodImpressionTurnsEffectBuff\(([\d\.]+)", clause):
                b_val = float(re.match(r"^setGoodImpressionTurnsEffectBuff\(([\d\.]+)", clause).group(1))
                self.good_impression_effect_buff += b_val
            elif re.match(r"^halfCostTurns\+=(\d+)", clause):
                self.buffs["halfCostTurns"] = int(self.buffs.get("halfCostTurns", 0) or 0) + int(
                    re.match(r"^halfCostTurns\+=(\d+)", clause).group(1)
                )
            elif re.match(r"^fixedStamina\+=(\d+)", clause):
                self.stamina = min(self.max_stamina, self.stamina + int(re.match(r"^fixedStamina\+=(\d+)", clause).group(1)))
            elif re.match(r"^fixedStamina-=(\d+)", clause):
                self.stamina = max(1, self.stamina - int(re.match(r"^fixedStamina-=(\d+)", clause).group(1)))

        if direct_gi_increases > 0 and any(eid in (919, 920) for eid in self.battle_exhaust_pile):
            for _ in range(direct_gi_increases):
                self.signature_gi_recast_counter += 1
                if (
                    self.signature_gi_recast_counter % 3 == 0
                    and self.signature_gi_recasts_remaining > 0
                    and not self.signature_gi_recast_used_turn
                ):
                    self.signature_gi_recasts_remaining -= 1
                    self.signature_gi_recast_used_turn = True
                    self.good_impression_effect_buff += 0.5
                    recast_g = 4 if 920 in self.battle_exhaust_pile else 3
                    self.genki += recast_g + max(0, mot_now)
                    self._record_zone_event("触发《ありのままのわたし》【再演】：免费再放(好印象强化+50%)")

        if self.genki != shield_before or self.stamina != hp_before or "genki" in (dsl_actions or "") or dsl_cost:
            cname = (row.get("name") if row else None) or f"#{card_id}"
            self._record_shield_event(
                f"打出《{cname}》: 盾 {shield_before}->{self.genki}, 血 {hp_before}->{self.stamina}"
            )

        self.cards_played_history.append(
            {
                "turn": self.turn,
                "card_id": card_id,
                "base_id": base_id,
                "limit_count": limit_count,
            }
        )

    def record_drink_use(
        self,
        drink_id: int,
        dsl_actions: str = "",
        db: Optional[Any] = None,
    ) -> None:
        """Records drink consumption in the current stage, including exchangeHand discards and shield/HP buffs."""
        self.drinks_used_history.append(drink_id)
        dsl = dsl_actions or ""
        d_row = db.get_drink_by_id(drink_id) if db is not None else None
        if not dsl and d_row:
            dsl = d_row.get("dsl_actions") or ""
        shield_before = self.genki
        hp_before = self.stamina
        for clause in [cl.strip() for cl in dsl.split(";") if cl.strip()]:
            if clause.startswith("at:") or "{" in clause:
                continue
            m_g = re.match(r"^(?:fixedGenki|genki)\+=(\d+)", clause)
            if m_g:
                self.genki += int(m_g.group(1))
                continue
            m_m = re.match(r"^motivation\+=(\d+)", clause)
            if m_m:
                self.buffs["motivation"] = int(self.buffs.get("motivation", 0) or 0) + int(m_m.group(1))
                continue
            m_gi = re.match(r"^goodImpressionTurns\+=(\d+)", clause)
            if m_gi:
                base_gi = int(m_gi.group(1))
                mult = 1.0 + max(0.0, self.good_impression_effect_buff)
                gain = int(math.floor(base_gi * mult + 1e-6))
                self.buffs["goodImpressionTurns"] = int(self.buffs.get("goodImpressionTurns", 0) or 0) + max(base_gi, gain)
                continue
            m_s = re.match(r"^fixedStamina\+=(\d+)", clause)
            if m_s:
                self.stamina = min(self.max_stamina, self.stamina + int(m_s.group(1)))
        if self.genki != shield_before or self.stamina != hp_before:
            dname = (d_row.get("name") if d_row else None) or f"#{drink_id}"
            self._record_shield_event(f"喝《{dname}》: 盾 {shield_before}->{self.genki}, 血 {hp_before}->{self.stamina}")
        if "exchangehand" in dsl.lower():
            self.end_current_turn_hand(db=db, reason="使用换牌饮料，原手牌弃置")

    def start_new_battle(self, db: Optional[Any] = None) -> None:
        """Resets battle-local state for a new stage and places produce_deck into Draw Pile."""
        self.sanitize_produce_deck(db=db)
        self.used_limit_cards.clear()
        self.battle_played_counts.clear()
        self.battle_exhausted_counts.clear()
        self.battle_generated_cards.clear()
        self.pending_top_deck.clear()
        self.cards_played_history.clear()
        self.drinks_used_history.clear()
        self.signature_genki_trigger_active = False
        self.signature_genki_counter = 0
        self.good_impression_effect_buff = 0.0
        self.signature_gi_recast_counter = 0
        self.signature_gi_recasts_remaining = 2
        self.signature_gi_recast_used_turn = False
        self.in_battle = True
        self.turn = 1
        self.turns_remaining = 10
        self.card_uses_remaining = 1
        self.genki = 0
        self.shield_history.clear()

        self.battle_hand.clear()
        self.battle_draw_pile = list(self.produce_deck)
        self.battle_discard_pile.clear()
        self.battle_exhaust_pile.clear()
        self.battle_reshuffle_count = 0
        self.battle_zone_events.clear()
        self._battle_tracked_turn = None
        self._battle_tracked_turns_remaining = None
        if self.battle_draw_pile:
            self._record_zone_event(
                f"【课程/考试开局建库】整局卡库 {len(self.battle_draw_pile)} 张全部置入【牌堆】，弃牌/除外区为空"
            )

    def start_new_turn(self, turns_remaining: int, db: Optional[Any] = None) -> None:
        """Advances the turn counter, discards unplayed hand cards, and resets per-turn card usage."""
        self.end_current_turn_hand(db=db, reason="回合推进，未打出手牌")
        self.turn += 1
        self.turns_remaining = turns_remaining
        self.card_uses_remaining = 1
        self.signature_gi_recast_used_turn = False
        for tk in ("halfCostTurns", "doubleCostTurns", "goodConditionTurns", "goodImpressionTurns"):
            if int(self.buffs.get(tk, 0) or 0) > 0:
                self.buffs[tk] = max(0, int(self.buffs[tk]) - 1)

    def reconcile_live_battle_state(self, obs: ScreenObservation) -> None:
        """Backstop transient OCR dropouts in live battle frames using backend shield/motivation/goodImpression ledger."""
        if getattr(obs, "p_points", 0) > 0 or obs.scene_type == "shop":
            self.p_points = int(getattr(obs, "p_points", 0) or 0)
        if obs.scene_type != "battle_turn":
            return
        if obs.genki > 0:
            self.genki = int(obs.genki)
        elif self.genki > 0:
            obs.genki = int(self.genki)

        if self.p_idol_id == 163:
            prev_gi = int(self.buffs.get("goodImpressionTurns", 0) or 0)
            obs_gi = int(obs.buffs.get("goodImpressionTurns", 0) or 0)
            obs_mot = int(obs.buffs.get("motivation", 0) or 0)
            if prev_gi >= 20 and obs_gi == 0 and obs_mot >= prev_gi - 3:
                obs.buffs["goodImpressionTurns"] = obs_mot
                prev_mot = int(self.buffs.get("motivation", 0) or 0)
                if prev_mot > 0:
                    obs.buffs["motivation"] = prev_mot
                else:
                    obs.buffs.pop("motivation", None)

        for bk in ("motivation", "goodImpressionTurns", "halfCostTurns"):
            val = int(obs.buffs.get(bk, 0) or 0)
            if val > 0:
                self.buffs[bk] = val
            elif int(self.buffs.get(bk, 0) or 0) > 0:
                obs.buffs[bk] = int(self.buffs[bk])

    def get_shield_prompt_summary(self) -> str:
        """Format live backend Shield (盾/元气), HP, Good Impression, Motivation, Exhaust count and ledger."""
        mot = int(self.buffs.get("motivation", 0) or 0)
        gi = int(self.buffs.get("goodImpressionTurns", 0) or 0)
        half_t = int(self.buffs.get("halfCostTurns", 0) or 0)
        ex_n = len(self.battle_exhaust_pile)
        gi_buff_pct = int(round(self.good_impression_effect_buff * 100))
        gi_extra = f"(强化+{gi_buff_pct}%)" if gi_buff_pct > 0 else ""
        base = (
            f"🛡️当前护盾(元気): {self.genki} | ❤️真实体力: {self.stamina}/{self.max_stamina} "
            f"| 👍好印象: {gi}层{gi_extra} | 🔥干劲(やる気): {mot} | 🚫除外区: {ex_n}张 | 🪶绿血减半: {half_t}回合"
        )
        if self.shield_history:
            return f"{base}\n  - 盾血流水账: " + " | ".join(self.shield_history[-3:])
        return base

    def update_observation(self, obs: ScreenObservation) -> None:
        """Synchronizes internal state with perception observation."""
        self.turn = obs.turn
        self.turns_remaining = obs.turns_remaining
        self.stamina = obs.stamina
        self.max_stamina = obs.max_stamina
        self.genki = obs.genki
        self.score = obs.score
        if getattr(obs, "p_points", 0) > 0 or obs.scene_type == "shop":
            self.p_points = int(getattr(obs, "p_points", 0) or 0)
        obs_stats = getattr(obs, "idol_stats", None) or {}
        if obs_stats and max((int(v or 0) for v in obs_stats.values()), default=0) > 300:
            self.idol_stats = {str(k): int(v or 0) for k, v in obs_stats.items()}
        self.card_uses_remaining = obs.card_uses_remaining
        self.buffs = dict(obs.buffs)
        if obs.plan:
            self.plan = obs.plan.lower()
        if getattr(obs, "scenario_id", None):
            self.scenario_id = str(obs.scenario_id).lower()

    def is_card_exhausted(self, card_id: int, base_id: Optional[int] = None) -> bool:
        """Checks if all known copies of a limit_count=1 card have already been exhausted in this battle."""
        return card_id in self.used_limit_cards or (
            base_id is not None and base_id in self.used_limit_cards
        )

    def _format_zone_card_list(
        self,
        card_ids: List[int],
        db: Optional[Any] = None,
        show_limit_tag: bool = False,
        unplayed_limit_note: bool = False,
    ) -> str:
        if not card_ids:
            return "空"
        counts = Counter(card_ids)
        parts: List[str] = []
        for cid, cnt in counts.items():
            row = db.get_card_by_id(cid) if db is not None else None
            cname = (row.get("name") if row else None) or f"#{cid}"
            is_limit = bool(row and row.get("limit_count") == 1)
            if show_limit_tag:
                suffix = "[限1]" if is_limit else "[可重复]"
            elif unplayed_limit_note and is_limit:
                suffix = "[限1·未打出仍可再用]"
            else:
                suffix = ""
            parts.append(f"《{cname}》{suffix}×{cnt}")
        return "、".join(parts)

    def get_deck_prompt_summary(
        self,
        db: Optional[Any] = None,
        hand_cards: Optional[List[Any]] = None,
    ) -> str:
        """Format full produce_deck + live per-battle 4-zone piles (Hand/Draw/Discard/Exhausted)."""
        if not self.produce_deck:
            return "  - 尚未初始化卡库记录"

        deck_counts = Counter(self.produce_deck)
        active_n, mental_n, trouble_n = 0, 0, 0
        deck_items: List[str] = []
        for cid, cnt in deck_counts.items():
            row = db.get_card_by_id(cid) if db is not None else None
            name = (row.get("name") if row else None) or f"#{cid}"
            ctype = (row.get("type") if row else "active") or "active"
            is_sig = bool(row and row.get("is_signature") == 1)
            is_limit = bool(row and row.get("limit_count") == 1)
            if ctype == "active":
                active_n += cnt
            elif ctype == "mental":
                mental_n += cnt
            else:
                trouble_n += cnt
            role_tag = "[固有⭐]" if is_sig else f"[{ctype[0].upper()}]"
            reuse_tag = "[限1]" if is_limit else "[可重复]"
            deck_items.append(f"《{name}》(#{cid}){role_tag}{reuse_tag}×{cnt}")

        lines = [
            f"  - 整局卡库总计 {len(self.produce_deck)} 张 (主动A:{active_n}张 / 精神M:{mental_n}张 / 干扰:{trouble_n}张): "
            + "、".join(deck_items)
        ]

        if hand_cards is not None:
            self._ensure_battle_zones_initialized(db=db)
            hand_ids = list(self.battle_hand)
            draw_ids = list(self.battle_draw_pile)
            discard_ids = list(self.battle_discard_pile)
            exhaust_ids = list(self.battle_exhaust_pile)

            # Read-only fallback if called directly without prior sync_observed_hand()
            obs_ids = [int(c.card_id) for c in hand_cards if getattr(c, "card_id", None)]
            if not hand_ids and obs_ids:
                hand_ids = obs_ids
                for hcid in obs_ids:
                    hrow = db.get_card_by_id(hcid) if db is not None else None
                    hbase = int(hrow["base_id"]) if (hrow and hrow.get("base_id")) else None
                    self._remove_one_card(draw_ids, hcid, base_id=hbase, db=db)

            hand_txt = self._format_zone_card_list(hand_ids, db=db, show_limit_tag=True)
            draw_txt = self._format_zone_card_list(draw_ids, db=db, show_limit_tag=True)
            discard_txt = self._format_zone_card_list(discard_ids, db=db, unplayed_limit_note=True)
            exhaust_txt = self._format_zone_card_list(exhaust_ids, db=db, show_limit_tag=False)

            reshuffle_note = f" | 本场已触发弃牌→牌堆重洗 {self.battle_reshuffle_count} 次" if self.battle_reshuffle_count > 0 else ""
            if len(draw_ids) == 0:
                draw_hint = "空（⚠️ 下次抽牌将立即把【弃牌】全部洗回【牌堆】）"
            elif len(draw_ids) <= 3:
                draw_hint = f"{draw_txt}（即将见底：只有这 {len(draw_ids)} 张出完后【弃牌】才会洗回【牌堆】）"
            else:
                draw_hint = f"{draw_txt}（{len(draw_ids)}张出完前【弃牌】不会被抽到）"

            lines.append(f"  - 【本场四区牌库实时分布】{reshuffle_note.lstrip(' |')}")
            lines.append(f"    1) 🖐️ 【手牌】({len(hand_ids)}张): {hand_txt}")
            if self.pending_top_deck:
                gen_names = []
                for gid in self.pending_top_deck:
                    grow = db.get_card_by_id(gid) if db is not None else None
                    gen_names.append(f"《{(grow.get('name') if grow else None) or f'#{gid}'}》(#{gid})")
                lines.append("    * ⚡ 【牌库顶待抽生成卡】(下次最先上手): " + "、".join(gen_names))
            lines.append(f"    2) 🎴 【牌堆/待抽】({len(draw_ids)}张): {draw_hint}")
            lines.append(
                f"    3) ♻️ 【弃牌】({len(discard_ids)}张，未打出的手牌 & 已打出可重复牌；等牌堆出完才洗回牌堆): {discard_txt}"
            )
            lines.append(
                f"    4) 🚫 【除外】({len(exhaust_ids)}张，已打出限1牌，本场不可再使用): {exhaust_txt if exhaust_ids else '空'}"
            )

            non_hand_pool = draw_ids + discard_ids
            lines.append(
                "  - 📦 牌库中除手牌外的其余循环/待抽牌: "
                + self._format_zone_card_list(non_hand_pool, db=db)
            )
            if exhaust_ids:
                lines.append("  - 🚫 本局已消耗进除外区(限1不可再用): " + exhaust_txt)
            if self.battle_zone_events:
                lines.append("  - 🔄 【本步牌库四区变动记录】: " + " | ".join(self.battle_zone_events[-4:]))

        return "\n".join(lines)

    def reset_llm_session(self) -> None:
        """Reset incremental LLM knowledge registry for a new Produce session."""
        self.session_bootstrapped = False
        self.llm_synced_card_ids.clear()
        self.llm_synced_drink_ids.clear()
        self.llm_synced_p_items.clear()
        self.llm_last_deck_snapshot.clear()
        self.llm_last_week = None
        self.llm_last_scene = None
        self.llm_last_in_battle = False
        self.llm_pending_events.clear()

    def record_session_event(self, note: str) -> None:
        """Queue an event receipt (e.g. executed card, auto-skip, deck pickup) for next LLM delta."""
        clean = (note or "").strip()
        if not clean:
            return
        if not self.llm_pending_events or self.llm_pending_events[-1] != clean:
            self.llm_pending_events.append(clean)
        if len(self.llm_pending_events) > 20:
            self.llm_pending_events = self.llm_pending_events[-20:]

    def compute_deck_diff(self) -> Tuple[List[int], List[int]]:
        """Return (added_card_ids, removed_card_ids) since last committed LLM sync."""
        cur = Counter(self.produce_deck)
        prev = Counter(self.llm_last_deck_snapshot)
        added: List[int] = []
        removed: List[int] = []
        for cid, cnt in (cur - prev).items():
            added.extend([cid] * cnt)
        for cid, cnt in (prev - cur).items():
            removed.extend([cid] * cnt)
        return added, removed

    def commit_llm_sync(
        self,
        *,
        card_ids: Optional[Set[int]] = None,
        drink_ids: Optional[Set[int]] = None,
        p_item_keys: Optional[Set[str]] = None,
        week: Optional[int] = None,
        scene_type: Optional[str] = None,
        in_battle: Optional[bool] = None,
    ) -> None:
        """Mark the current state & newly described cards/items as committed to LLM session memory."""
        self.session_bootstrapped = True
        if card_ids:
            self.llm_synced_card_ids.update(int(c) for c in card_ids)
        if drink_ids:
            self.llm_synced_drink_ids.update(int(d) for d in drink_ids)
        if p_item_keys:
            self.llm_synced_p_items.update(p_item_keys)
        self.llm_last_deck_snapshot = list(self.produce_deck)
        if week is not None:
            self.llm_last_week = int(week)
        if scene_type is not None:
            self.llm_last_scene = scene_type
        if in_battle is not None:
            self.llm_last_in_battle = bool(in_battle)
        self.llm_pending_events.clear()

    def export_checkpoint(self) -> Dict[str, Any]:
        """Serialize persistent produce & LLM incremental session state for crash/interrupt recovery."""
        return {
            "plan": self.plan,
            "p_idol_id": self.p_idol_id,
            "scenario_id": self.scenario_id,
            "produce_deck": list(self.produce_deck),
            "equipped_p_items": [dict(it) for it in self.equipped_p_items],
            "card_memory_hooks": {str(k): dict(v) for k, v in self.card_memory_hooks.items()},
            "card_customizations": {str(k): dict(v) for k, v in self.card_customizations.items()},
            "memory_slots": [dict(s) for s in self.memory_slots],
            "memory_passives": [dict(p) for p in self.memory_passives],
            "recommended_effect": self.recommended_effect,
            "initial_loadout_captured": self.initial_loadout_captured,
            "produce_library_log": list(self.produce_library_log),
            "turn": self.turn,
            "turns_remaining": self.turns_remaining,
            "card_uses_remaining": self.card_uses_remaining,
            "stamina": self.stamina,
            "max_stamina": self.max_stamina,
            "genki": self.genki,
            "score": self.score,
            "p_points": self.p_points,
            "idol_stats": dict(self.idol_stats),
            "buffs": dict(self.buffs),
            "in_battle": self.in_battle,
            "used_limit_cards": sorted(self.used_limit_cards),
            "battle_played_counts": {str(k): int(v) for k, v in self.battle_played_counts.items()},
            "battle_exhausted_counts": {str(k): int(v) for k, v in self.battle_exhausted_counts.items()},
            "battle_generated_cards": list(self.battle_generated_cards),
            "pending_top_deck": list(self.pending_top_deck),
            "signature_genki_trigger_active": self.signature_genki_trigger_active,
            "signature_genki_counter": self.signature_genki_counter,
            "good_impression_effect_buff": self.good_impression_effect_buff,
            "signature_gi_recast_counter": self.signature_gi_recast_counter,
            "signature_gi_recasts_remaining": self.signature_gi_recasts_remaining,
            "signature_gi_recast_used_turn": self.signature_gi_recast_used_turn,
            "cards_played_history": list(self.cards_played_history),
            "drinks_used_history": list(self.drinks_used_history),
            "shield_history": list(self.shield_history),
            "battle_hand": list(self.battle_hand),
            "battle_draw_pile": list(self.battle_draw_pile),
            "battle_discard_pile": list(self.battle_discard_pile),
            "battle_exhaust_pile": list(self.battle_exhaust_pile),
            "battle_reshuffle_count": self.battle_reshuffle_count,
            "battle_zone_events": list(self.battle_zone_events),
            "battle_tracked_turn": self._battle_tracked_turn,
            "battle_tracked_turns_remaining": self._battle_tracked_turns_remaining,
            "session_bootstrapped": self.session_bootstrapped,
            "llm_synced_card_ids": sorted(self.llm_synced_card_ids),
            "llm_synced_drink_ids": sorted(self.llm_synced_drink_ids),
            "llm_synced_p_items": sorted(self.llm_synced_p_items),
            "llm_last_deck_snapshot": list(self.llm_last_deck_snapshot),
            "llm_last_week": self.llm_last_week,
            "llm_last_scene": self.llm_last_scene,
            "llm_last_in_battle": self.llm_last_in_battle,
            "llm_pending_events": list(self.llm_pending_events),
        }

    def load_checkpoint(self, data: Dict[str, Any], db: Optional[Any] = None) -> None:
        """Restore produce & battle tracker state from an exported checkpoint dict."""
        if not isinstance(data, dict):
            return
        if data.get("plan"):
            self.plan = str(data["plan"]).lower()
        if "p_idol_id" in data:
            self.p_idol_id = int(data["p_idol_id"]) if data["p_idol_id"] is not None else None
        if data.get("scenario_id"):
            self.scenario_id = str(data["scenario_id"]).lower()
        self.produce_deck = [int(x) for x in (data.get("produce_deck") or [])]
        self.equipped_p_items = [dict(it) for it in (data.get("equipped_p_items") or [])]
        self.card_memory_hooks = {
            int(k): dict(v) for k, v in (data.get("card_memory_hooks") or {}).items()
        }
        self.card_customizations = {
            int(k): dict(v)
            for k, v in (data.get("card_customizations") or {}).items()
            if isinstance(v, dict)
        }
        self.memory_slots = [dict(s) for s in (data.get("memory_slots") or []) if isinstance(s, dict)]
        self.memory_passives = [dict(p) for p in (data.get("memory_passives") or []) if isinstance(p, dict)]
        self.recommended_effect = (
            str(data["recommended_effect"]) if data.get("recommended_effect") is not None else None
        )
        self.initial_loadout_captured = bool(data.get("initial_loadout_captured", False))
        self.produce_library_log = [str(x) for x in (data.get("produce_library_log") or [])][-20:]
        self.sanitize_produce_deck(db=db)
        self.turn = int(data.get("turn", 1))
        self.turns_remaining = int(data.get("turns_remaining", 10))
        self.card_uses_remaining = int(data.get("card_uses_remaining", 1))
        self.stamina = int(data.get("stamina", 30))
        self.max_stamina = int(data.get("max_stamina", 30))
        self.genki = int(data.get("genki", 0))
        self.score = int(data.get("score", 0))
        self.p_points = int(data.get("p_points", 0))
        if isinstance(data.get("idol_stats"), dict):
            self.idol_stats = {str(k): int(v or 0) for k, v in data["idol_stats"].items()}
        self.buffs = {str(k): int(v) for k, v in (data.get("buffs") or {}).items()}
        self.in_battle = bool(data.get("in_battle", False))
        self.used_limit_cards = {int(x) for x in (data.get("used_limit_cards") or [])}
        self.battle_played_counts = {int(k): int(v) for k, v in (data.get("battle_played_counts") or {}).items()}
        self.battle_exhausted_counts = {int(k): int(v) for k, v in (data.get("battle_exhausted_counts") or {}).items()}
        self.battle_generated_cards = [int(x) for x in (data.get("battle_generated_cards") or [])]
        self.pending_top_deck = [int(x) for x in (data.get("pending_top_deck") or [])]
        self.signature_genki_trigger_active = bool(data.get("signature_genki_trigger_active", False))
        self.signature_genki_counter = int(data.get("signature_genki_counter", 0))
        self.good_impression_effect_buff = float(data.get("good_impression_effect_buff", 0.0))
        self.signature_gi_recast_counter = int(data.get("signature_gi_recast_counter", 0))
        self.signature_gi_recasts_remaining = int(data.get("signature_gi_recasts_remaining", 2))
        self.signature_gi_recast_used_turn = bool(data.get("signature_gi_recast_used_turn", False))
        self.cards_played_history = list(data.get("cards_played_history") or [])
        self.drinks_used_history = [int(x) for x in (data.get("drinks_used_history") or [])]
        self.shield_history = [str(x) for x in (data.get("shield_history") or [])][-8:]
        self.battle_hand = [int(x) for x in (data.get("battle_hand") or [])]
        self.battle_draw_pile = [int(x) for x in (data.get("battle_draw_pile") or [])]
        self.battle_discard_pile = [int(x) for x in (data.get("battle_discard_pile") or [])]
        self.battle_exhaust_pile = [int(x) for x in (data.get("battle_exhaust_pile") or [])]
        self.battle_reshuffle_count = int(data.get("battle_reshuffle_count", 0))
        self.battle_zone_events = [str(x) for x in (data.get("battle_zone_events") or [])][-12:]
        self._battle_tracked_turn = (
            int(data["battle_tracked_turn"]) if data.get("battle_tracked_turn") is not None else None
        )
        self._battle_tracked_turns_remaining = (
            int(data["battle_tracked_turns_remaining"])
            if data.get("battle_tracked_turns_remaining") is not None
            else None
        )
        if self.in_battle and self.produce_deck and self._zones_empty():
            self._ensure_battle_zones_initialized(db=db)
        self.session_bootstrapped = bool(data.get("session_bootstrapped", False))
        self.llm_synced_card_ids = {int(x) for x in (data.get("llm_synced_card_ids") or [])}
        self.llm_synced_drink_ids = {int(x) for x in (data.get("llm_synced_drink_ids") or [])}
        self.llm_synced_p_items = {str(x) for x in (data.get("llm_synced_p_items") or [])}
        self.llm_last_deck_snapshot = [int(x) for x in (data.get("llm_last_deck_snapshot") or [])]
        self.llm_last_week = int(data["llm_last_week"]) if data.get("llm_last_week") is not None else None
        self.llm_last_scene = data.get("llm_last_scene")
        self.llm_last_in_battle = bool(data.get("llm_last_in_battle", False))
        self.llm_pending_events = [str(x) for x in (data.get("llm_pending_events") or [])][-20:]

    def reset(self, keep_deck: bool = True) -> None:
        """Resets battle tracker state for a fresh stage; preserves produce_deck by default."""
        if not keep_deck:
            self.produce_deck.clear()
            self.equipped_p_items.clear()
            self.card_memory_hooks.clear()
            self.card_customizations.clear()
            self.memory_slots.clear()
            self.memory_passives.clear()
            self.recommended_effect = None
            self.initial_loadout_captured = False
            self.produce_library_log.clear()
            self.p_points = 0
            self.reset_llm_session()
        self.used_limit_cards.clear()
        self.battle_played_counts.clear()
        self.battle_exhausted_counts.clear()
        self.battle_generated_cards.clear()
        self.pending_top_deck.clear()
        self.cards_played_history.clear()
        self.drinks_used_history.clear()
        self.shield_history.clear()
        self.battle_hand.clear()
        self.battle_draw_pile.clear()
        self.battle_discard_pile.clear()
        self.battle_exhaust_pile.clear()
        self.battle_reshuffle_count = 0
        self.battle_zone_events.clear()
        self._battle_tracked_turn = None
        self._battle_tracked_turns_remaining = None
        self.signature_genki_trigger_active = False
        self.signature_genki_counter = 0
        self.good_impression_effect_buff = 0.0
        self.signature_gi_recast_counter = 0
        self.signature_gi_recasts_remaining = 2
        self.signature_gi_recast_used_turn = False
        self.in_battle = False
        self.turn = 1
        self.turns_remaining = 10
        self.card_uses_remaining = 1
        self.stamina = 30
        self.max_stamina = 30
        self.genki = 0
        self.score = 0
        self.buffs.clear()
