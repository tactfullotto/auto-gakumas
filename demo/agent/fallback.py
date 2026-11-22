"""
Deterministic Heuristic Rule Engine and Zero-Token Single Action Auto-Skip.
demo/agent/fallback.py
"""

from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence
from demo.schemas import DecisionResult, LegalAction, ScreenObservation


def check_single_legal_action(legal_actions: Sequence[LegalAction]) -> Optional[DecisionResult]:
    """
    Zero-token skip optimization:
    If there is exactly one legal action available, skip LLM inference completely.
    Returns DecisionResult with source='rule_skip', prompt_tokens=0, completion_tokens=0, latency_ms=0.0.
    """
    if len(legal_actions) == 1:
        act = legal_actions[0]
        return DecisionResult(
            choice_id=act.choice_id,
            reason="Single legal action auto-skip",
            source="rule_skip",
            latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
        )
    return None


class HeuristicDecisionEngine:
    """
    Deterministic rule engine that provides robust offline decisions when LLM is unavailable:
    - Missing API key
    - Network timeout / connection error
    - Exhausted retries (e.g. invalid JSON or out-of-bounds choice_id)

    Guarantees:
    - Zero token consumption (prompt_tokens=0, completion_tokens=0).
    - Sub-millisecond latency.
    - Always selects a valid choice_id from legal_actions.
    - Complete robustness against None attributes on ScreenObservation and LegalAction.
    """

    def decide(
        self,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation] = None,
    ) -> DecisionResult:
        """
        Evaluates legal actions using domain-specific heuristics across all game scenes.
        Safely handles obs being None or having None attributes.
        """
        start_time = perf_counter()

        if not legal_actions:
            return DecisionResult(
                choice_id=0,
                reason="No legal actions available",
                source="fallback",
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
            )

        # Zero-token single action skip check
        single_skip = check_single_legal_action(legal_actions)
        if single_skip is not None:
            return single_skip

        # Determine scene type safely
        scene_type = (getattr(obs, "scene_type", None) if obs else None) or self._infer_scene_type(legal_actions)

        if scene_type == "battle_turn":
            result = self._decide_battle(legal_actions, obs)
        elif scene_type == "produce_schedule":
            result = self._decide_produce_schedule(legal_actions, obs)
        elif scene_type == "card_reward":
            result = self._decide_card_reward(legal_actions, obs)
        elif scene_type == "popup":
            result = self._decide_popup(legal_actions, obs)
        elif scene_type == "shop":
            result = self._decide_shop(legal_actions, obs)
        else:
            first = legal_actions[0]
            result = DecisionResult(
                choice_id=first.choice_id,
                reason=f"Heuristic fallback default selection: {first.description}",
                source="fallback",
            )

        elapsed_ms = round((perf_counter() - start_time) * 1000.0, 3)
        result.latency_ms = elapsed_ms
        return result

    def _infer_scene_type(self, legal_actions: List[LegalAction]) -> str:
        for a in legal_actions:
            if a.action_type in ("play_card", "use_drink", "end_turn"):
                return "battle_turn"
            desc = (a.description or "").lower()
            if any(k in desc for k in ("sp", "训练", "lesson", "休息", "rest", "咨询")):
                return "produce_schedule"
            if any(k in desc for k in ("确认", "确定", "ok", "confirm", "close", "关闭")):
                return "popup"
        return "battle_turn"

    def _decide_battle(
        self,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation],
    ) -> DecisionResult:
        """
        Battle heuristics:
        1. Priority order: playable drink with buff > playable card with highest value/score > end turn.
        2. Archetype alignment:
           - Sense: boost 好调 / 集中 in early/mid turns, burst in late turns.
           - Logic: boost 好印象 / 干劲 (やる気) in early/mid turns, burst in late turns.
           - Anomaly: stance switches (强气) and charge in early/mid, full power burst in fullPower.
        3. Avoid suicidal HP cost (stamina <= actual cost).
        Safe against any None attributes in obs or legal_actions.
        """
        stamina = (getattr(obs, "stamina", None) if obs else None) or 30
        turns_remaining = (getattr(obs, "turns_remaining", None) if obs else None) or 10
        plan = ((getattr(obs, "plan", None) if obs else None) or "sense").lower()
        buffs = (getattr(obs, "buffs", None) if obs else None) or {}
        week = (getattr(obs, "week", None) if obs else None) or 1
        week_activity = (getattr(obs, "week_activity", None) if obs else None) or ""
        is_exam_stage = (week in (6, 13)) or ("試験" in week_activity)
        good_impression = int(buffs.get("goodImpressionTurns", 0) or buffs.get("good_impression", 0) or 0)

        drink_actions = [a for a in legal_actions if a.action_type == "use_drink"]
        card_actions = [a for a in legal_actions if a.action_type == "play_card"]
        end_turn_actions = [a for a in legal_actions if a.action_type == "end_turn"]

        # 1. Drink Priority: Playable drink with buff (avoid zero-value waste like goodImpression% at 0)
        best_drink = None
        best_drink_score = -1
        buff_keywords = (
            "好调", "好印象", "集中", "干劲", "やる気", "全力", "强气",
            "充能", "出牌", "额外", "元气", "恢复", "体力"
        )

        for d_act in drink_actions:
            score = 2000
            desc = d_act.description or ""
            payload = d_act.payload or {}
            dsl_actions = str(payload.get("dsl_actions") or "")
            summary = f"{dsl_actions} {payload.get('dsl_summary') or ''} {desc}"
            if ("入れ替える" in summary or "换牌" in summary or "exchangeHand" in summary) and stamina >= 15 and card_actions:
                continue
            if ("goodImpressionTurns" in dsl_actions and "score+=" in dsl_actions or "好印象の" in summary) and good_impression <= 0 and card_actions:
                continue
            if ("perfectCondition" in summary or "絶好調" in summary or "绝好调" in summary) and is_exam_stage and turns_remaining >= 5 and card_actions:
                continue

            # Bonus for buff keywords
            for kw in buff_keywords:
                if kw in summary:
                    score += 100

            # Plan alignment bonus
            if plan == "sense" and any(k in summary for k in ("好调", "集中", "绝好调")):
                score += 500
            elif plan == "logic" and any(k in summary for k in ("好印象", "干劲", "元气")):
                score += 500
            elif plan == "anomaly" and any(k in summary for k in ("强气", "全力", "充能")):
                score += 500

            if score > best_drink_score:
                best_drink_score = score
                best_drink = d_act

        if best_drink is not None and best_drink_score > 0:
            drink_payload = best_drink.payload or {}
            drink_name = drink_payload.get("name", best_drink.description)
            return DecisionResult(
                choice_id=best_drink.choice_id,
                reason=f"Heuristic: Drink buff prioritized ({drink_name})",
                source="fallback",
            )

        # 2. Playable Cards: Evaluate value and avoid suicide
        scored_cards = []
        obs_cards = (getattr(obs, "cards", None) if obs else None) or []
        for c_act in card_actions:
            payload = c_act.payload or {}
            hp_loss = payload.get("hp_loss") or 0

            # Avoid suicidal HP cost: playing card cannot reduce stamina to <= 0
            if hp_loss >= stamina:
                # Suicidal action - strictly disqualify
                continue

            # Base score
            card_score = 1000
            card_desc = c_act.description or ""
            card_name = str(payload.get("name") or "")

            # Look up card in obs.cards for dsl_summary if available
            summary = ""
            for c in obs_cards:
                if getattr(c, "card_id", None) == payload.get("card_id"):
                    summary = getattr(c, "dsl_summary", "") or ""
                    break
            if not summary:
                summary = str(payload.get("dsl_summary") or "")
            full_text = f"{card_name} {card_desc} {summary}"

            # Archetype-specific weighting
            if plan == "sense":
                if turns_remaining > 2:
                    # Early/Mid turn: build buffs
                    if "绝好调" in full_text:
                        card_score += 400
                    if "好调" in full_text:
                        card_score += 350
                    if "集中" in full_text:
                        card_score += 300
                    if "得分" in full_text or "score" in full_text:
                        card_score += 150
                else:
                    # Late turn: burst score
                    if "得分" in full_text or "score" in full_text:
                        card_score += 450
                    if "好调" in full_text or "集中" in full_text:
                        card_score += 150
            elif plan == "logic":
                if turns_remaining > 2:
                    if "好印象" in full_text:
                        card_score += 350
                    if "干劲" in full_text or "やる気" in full_text:
                        card_score += 300
                    if "元气" in full_text:
                        card_score += 250
                    if "得分" in full_text or "score" in full_text:
                        card_score += 150
                else:
                    if "得分" in full_text or "score" in full_text:
                        card_score += 450
                    if "好印象" in full_text:
                        card_score += 250
            elif plan == "anomaly":
                stance = str(buffs.get("stance", "") or "")
                is_full_power = (
                    stance == "fullPower"
                    or (buffs.get("isFullPower") or 0) > 0
                    or (buffs.get("fullPowerCharge") or 0) >= 10
                )
                if not is_full_power:
                    if "进入强气" in full_text or "强气" in full_text:
                        card_score += 350
                    if "全力充能" in full_text or "充能" in full_text:
                        card_score += 300
                    if "温存" in full_text:
                        card_score += 200
                else:
                    if "全力" in full_text or "得分" in full_text:
                        card_score += 450
                    if "强气" in full_text:
                        card_score += 150

            # Free extra-use / signature starter cards never consume turn play slots -> highest priority
            dsl_actions = str(payload.get("dsl_actions") or "")
            if "cardUsesRemaining+=" in dsl_actions or "使用次数" in card_desc:
                card_score += 800

            # Long-turn exam pacing: stack Mental (M) buff cards in early/mid turns, burst Active (A) later
            card_type = str(payload.get("card_type") or "")
            if is_exam_stage:
                if turns_remaining >= 5:
                    if card_type == "mental":
                        card_score += 260
                    elif card_type == "active" and "cardUsesRemaining+=" not in dsl_actions:
                        card_score -= 180
                else:
                    if card_type == "active":
                        card_score += 280

            # Upgraded card bonus
            if payload.get("upgraded"):
                card_score += 50

            # Limit count 1 card bonus (powerful signature move)
            if payload.get("limit_count") == 1:
                card_score += 100

            # HP cost penalty (favor efficient cards)
            card_score -= hp_loss * 5

            scored_cards.append((card_score, c_act))

        if scored_cards:
            scored_cards.sort(key=lambda x: x[0], reverse=True)
            best_score, best_card = scored_cards[0]
            name = (best_card.payload or {}).get("name", "Card")
            return DecisionResult(
                choice_id=best_card.choice_id,
                reason=f"Heuristic: Play card '{name}' matching plan '{plan}' (score={best_score})",
                source="fallback",
            )

        # 3. End Turn: When no safe or advantageous card exists
        if end_turn_actions:
            end_act = end_turn_actions[0]
            return DecisionResult(
                choice_id=end_act.choice_id,
                reason="Heuristic: End turn (no safe card playable)",
                source="fallback",
            )

        # Fallback to first available action
        first = legal_actions[0]
        return DecisionResult(
            choice_id=first.choice_id,
            reason=f"Heuristic fallback: {first.description}",
            source="fallback",
        )

    def _decide_produce_schedule(
        self,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation],
    ) -> DecisionResult:
        """
        Produce schedule heuristics:
        If stamina < 15, prioritize Rest/Outing.
        Else prefer SP Lesson > Regular Lesson > Consultation.
        Safely handles any None attributes.
        """
        stamina = (getattr(obs, "stamina", None) if obs else None) or 30
        rest_keywords = ("休息", "外出", "休む", "お对かけ", "rest", "outing", "睡眠", "恢复", "recover")

        # Low stamina guard: Rest or Outing
        if stamina < 15:
            for act in legal_actions:
                desc = (act.description or "").lower()
                payload = act.payload or {}
                title = str(payload.get("title") or "").lower()
                if any(kw in desc or kw in title for kw in rest_keywords):
                    return DecisionResult(
                        choice_id=act.choice_id,
                        reason=f"Heuristic: Low stamina ({stamina}<15), prioritize Rest/Outing",
                        source="fallback",
                    )

        # High stamina: SP Lesson > Regular Lesson > Consultation > Rest > Other
        def score_schedule_action(act: LegalAction) -> int:
            desc = (act.description or "").lower()
            payload = act.payload or {}
            title = str(payload.get("title") or "").lower()
            combined = f"{desc} {title}"

            if "sp" in combined and any(k in combined for k in ("训练", "lesson", "レッスン")):
                return 1500
            if any(k in combined for k in ("训练", "lesson", "レッスン")):
                return 1200
            if any(k in combined for k in ("咨询", "相談", "consultation", "advice")):
                return 900
            if any(k in combined for k in rest_keywords):
                return 600
            return 300

        best_act = max(legal_actions, key=score_schedule_action)
        return DecisionResult(
            choice_id=best_act.choice_id,
            reason=f"Heuristic: Schedule selection with stamina {stamina} ({best_act.description})",
            source="fallback",
        )

    def _decide_card_reward(
        self,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation],
    ) -> DecisionResult:
        """
        Card reward heuristics:
        Pick card matching plan; prefer upgraded or high-tier cards.
        Safely handles any None attributes.
        """
        plan = ((getattr(obs, "plan", None) if obs else None) or "sense").lower()

        def score_card_reward(act: LegalAction) -> int:
            payload = act.payload or {}
            desc = (act.description or "").lower()
            opt_name = str(payload.get("name") or "")
            raw_title = str(payload.get("title") or opt_name)
            title = raw_title.lower()
            combined = f"{desc} {title}"

            # Confirmed "受け取らない" skip or active claim button takes top priority over slot previews
            if opt_name == "決定" or "已勾选『受け取らない』" in raw_title:
                return 2000
            if opt_name == "受け取る":
                return 1600
            if opt_name == "受け取らない":
                return 1300
            if opt_name.startswith("破棄候補_") or "预览" in opt_name or "キャンセル" in opt_name:
                return 200

            score = 100
            card_plan = str(payload.get("plan") or "").lower()
            if card_plan == plan or card_plan in ("all", "free", ""):
                score += 100
            else:
                score -= 100

            # Upgraded preference
            if payload.get("upgraded") or "+" in combined:
                score += 50

            # Rarity / Tier preference
            rarity = str(payload.get("rarity") or "").upper()
            if rarity == "SSR":
                score += 40
            elif rarity == "SR":
                score += 25
            elif rarity == "R":
                score += 15
            else:
                score += 5

            # Plan keyword synergy
            if plan == "sense" and any(k in combined for k in ("好调", "集中", "绝好调")):
                score += 20
            elif plan == "logic" and any(k in combined for k in ("好印象", "干劲", "元气")):
                score += 20
            elif plan == "anomaly" and any(k in combined for k in ("强气", "全力", "充能")):
                score += 20

            return score

        best_act = max(legal_actions, key=score_card_reward)
        return DecisionResult(
            choice_id=best_act.choice_id,
            reason=f"Heuristic: Card reward chosen for plan '{plan}' ({best_act.description})",
            source="fallback",
        )

    def _decide_popup(
        self,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation],
    ) -> DecisionResult:
        """
        Popup heuristics:
        Pick confirm / OK / close button.
        """
        confirm_keywords = ("确认", "确定", "ok", "confirm", "はい", "決定", "次へ", "知道了", "同意", "捨てる", "破棄する", "破棄", "使う", "使用する", "交換", "受け取る")
        close_keywords = ("关闭", "close", "閉じる", "dismiss")
        cancel_keywords = ("取消", "cancel", "いいえ", "キャンセル")

        def score_popup_action(act: LegalAction) -> int:
            desc = (act.description or "").lower()
            payload = act.payload or {}
            title = str(payload.get("title") or "").lower()
            combined = f"{desc} {title}"

            if any(k in combined for k in confirm_keywords):
                return 1000
            if any(k in combined for k in close_keywords):
                return 800
            if any(k in combined for k in cancel_keywords):
                return 200
            return 500

        best_act = max(legal_actions, key=score_popup_action)
        return DecisionResult(
            choice_id=best_act.choice_id,
            reason=f"Heuristic: Popup button selected ({best_act.description})",
            source="fallback",
        )

    def _decide_shop(
        self,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation],
    ) -> DecisionResult:
        """
        Shop heuristics:
        Exit immediately when P-points cannot afford any non-exit shop item; otherwise prefer Exchange > Upgrade > Drink/Card > Exit.
        """
        p_pts = int(getattr(obs, "p_points", 0) or 0) if obs is not None else 0
        has_exchange = any((a.payload or {}).get("name") == "交換する" or "交換する" in (a.description or "") for a in legal_actions)
        priced = [a for a in legal_actions if (a.payload or {}).get("price") is not None]
        if priced and not has_exchange:
            min_price = min(int((a.payload or {}).get("price") or 9999) for a in priced)
            if p_pts < min_price:
                exit_act = next(
                    (
                        a for a in legal_actions
                        if any(k in (a.description or "").lower() or k in str((a.payload or {}).get("name") or "").lower() for k in ("終了", "スキップ", "退出", "离开", "exit", "skip"))
                    ),
                    legal_actions[0],
                )
                return DecisionResult(
                    choice_id=exit_act.choice_id,
                    reason=f"Heuristic: P-points ({p_pts}P) insufficient (< min {min_price}P), exit shop",
                    source="fallback",
                )

        def score_shop_action(act: LegalAction) -> int:
            desc = (act.description or "").lower()
            name = str((act.payload or {}).get("name") or "").lower()
            if "交換する" in desc or "交換する" in name:
                return 1200
            if any(k in desc for k in ("升级", "强化", "強化", "upgrade")):
                return 1000
            if any(k in desc for k in ("饮料", "drink", "pドリンク", "卡牌", "card", "スキルカード")):
                return 800
            if any(k in desc for k in ("終了", "スキップ", "离开", "退出", "确认", "exit", "skip")):
                return 500
            return 300

        best_act = max(legal_actions, key=score_shop_action)
        return DecisionResult(
            choice_id=best_act.choice_id,
            reason=f"Heuristic: Shop action selected ({best_act.description})",
            source="fallback",
        )


# Compatibility alias
HeuristicRuleEngine = HeuristicDecisionEngine
