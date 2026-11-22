"""
Prompt and Legal Action Builder for Gakumas Decision Engine.

Supports two modes:
- Full Context Mode (compact=False, default): No 800-token cap. Introduces the
  produce scenario & current week flow, archetype playstyle (Sense/Logic/Anomaly),
  passive P-Item bonuses, stage/criteria bonuses, initial & current stats, and
  then enters the card-playing / schedule phase with complete, untruncated card
  and drink effects for LLM decision-making.
- Compact Mode (compact=True): Legacy <800-token condensed prompt format.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from demo.core.db import GakumasDB
from demo.core.idol_guides import (
    format_idol_session_guide_block,
    format_idol_turn_nudge,
    get_card_guide_annotation,
    get_memory_card_custom_meta,
    get_p_idol_guide,
    resolve_p_idol_id,
)
from demo.core.rules import (
    evaluate_card_cost_and_usability,
    format_card_conditions_zh,
    inspect_card_cost_and_conditions,
    summarize_dsl_actions,
)
from demo.core.scenarios import (
    compute_week_stats_and_stage_bonus,
    get_scenario_spec,
    get_week_node,
)
from demo.core.tracker import GameStateTracker
from demo.schemas import (
    BoundingBox,
    LegalAction,
    RecognizedCard,
    RecognizedDrink,
    ScreenObservation,
)


BUFF_NAME_MAP: Dict[str, str] = {
    "goodConditionTurns": "好调(得分倍率+50%)",
    "good_condition": "好调(得分倍率+50%)",
    "perfectConditionTurns": "绝好调(每层好调额外+10%倍率)",
    "perfect_condition": "绝好调(每层好调额外+10%倍率)",
    "concentration": "集中(每点附加固定得分)",
    "motivation": "干劲(每点附加额外元气护盾)",
    "goodImpressionTurns": "好印象(回合末按层数跳分)",
    "good_impression": "好印象(回合末按层数跳分)",
    "fullPowerCharge": "全力充能(满10点下回合进全力)",
    "full_effort": "全力充能(满10点下回合进全力)",
    "strong_spirit": "强气姿态(高倍率增伤)",
    "preserve": "温存姿态(减耗蓄力)",
    "nullifyCostCards": "免除体力消耗",
    "halfCostTurns": "体力消耗减半",
    "doubleCostTurns": "体力消耗加倍",
    "cardUsesRemaining": "额外出牌次数",
}


class PromptBuilder:
    """
    Builds structured LLM decision prompts and numbered legal action lists.
    By default (`compact=False`), generates the full-context prompt without
    an 800-token cap, covering scenario week flow, archetype playstyle,
    P-Item passives, stage bonuses, initial/current stats, and full card effects.
    """

    def __init__(self, db: Optional[GakumasDB] = None, compact: bool = False):
        self.db = db
        self.compact = compact
        self._card_cache: Dict[int, Optional[Dict[str, Any]]] = {}
        self._drink_cache: Dict[int, Optional[Dict[str, Any]]] = {}

    def _get_card_dict(self, card_id: int) -> Optional[Dict[str, Any]]:
        if not self.db:
            return None
        if card_id not in self._card_cache:
            self._card_cache[card_id] = self.db.get_card_by_id(card_id)
        return self._card_cache[card_id]

    def _get_drink_dict(self, drink_id: int) -> Optional[Dict[str, Any]]:
        if not self.db:
            return None
        if drink_id not in self._drink_cache:
            self._drink_cache[drink_id] = self.db.get_drink_by_id(drink_id)
        return self._drink_cache[drink_id]

    @staticmethod
    def _resolve_card_description(c_row: Optional[Dict[str, Any]], fallback_summary: str = "") -> str:
        if c_row and (c_row.get("description") or "").strip():
            return str(c_row["description"]).strip()
        if not c_row:
            return fallback_summary or "无描述"
        parts = []
        c_str = (c_row.get("dsl_cost") or "").strip()
        if c_str.startswith("stamina-="):
            parts.append(f"体力消費{c_str.split('-=', 1)[1]} (红血直扣)")
        elif c_str.startswith("cost-="):
            parts.append(f"消費体力{c_str.split('-=', 1)[1]} (绿血优先扣盾)")
        cond_zh = format_card_conditions_zh(c_row.get("dsl_conditions") or "", dsl_cost=c_str, dsl_actions="")
        if cond_zh and cond_zh != "无特殊前置条件":
            parts.append(cond_zh)
        act_zh = summarize_dsl_actions(c_row.get("dsl_actions") or "")
        if act_zh:
            parts.append(act_zh)
        if c_row.get("limit_count") == 1:
            parts.append("レッスン中1回")
        return " / ".join(parts) if parts else (fallback_summary or "无描述")

    @staticmethod
    def _format_card_cost_mechanism(
        dsl_cost: str,
        effective_cost: Optional[int] = None,
        shield_cost: Optional[int] = None,
        hp_loss: Optional[int] = None,
        cost_type: Optional[str] = None,
    ) -> str:
        ctype = cost_type or ("red" if (dsl_cost or "").startswith("stamina-=") else ("buff" if (dsl_cost or "") and not (dsl_cost or "").startswith("cost-=") else ("free" if (dsl_cost or "") in ("", "cost-=0") else "green")))
        if ctype == "red":
            val = hp_loss if hp_loss is not None else (dsl_cost.split("-=", 1)[1] if "-=" in (dsl_cost or "") else "?")
            return f"🔴红血直扣{val}(无视护盾与减耗直接扣体力)"
        if ctype == "buff":
            return f"🔷消耗Buff({dsl_cost})"
        if ctype == "free":
            return "🟢0消耗(无实际扣血)"
        eff_v = effective_cost if effective_cost is not None else (dsl_cost.split("-=", 1)[1] if "-=" in (dsl_cost or "") else "?")
        if shield_cost is not None and hp_loss is not None:
            if hp_loss == 0:
                return f"🟢绿血{eff_v}(先扣护盾{shield_cost}→无实际扣血)"
            return f"🟢绿血{eff_v}(护盾抵{shield_cost}+实际扣血{hp_loss})"
        return f"🟢绿血消耗{eff_v}(先扣护盾/受减半影响)"

    def generate_legal_actions_and_update_obs(
        self,
        obs: ScreenObservation,
        tracker: Optional[GameStateTracker] = None,
        compact: Optional[bool] = None,
    ) -> List[LegalAction]:
        """
        Pre-computes usability, shield/green-HP/red-HP breakdown, and trigger conditions for all cards/drinks,
        updates observation in-place, and returns the list of legal candidate actions.
        Always fetches metadata from DB (using cache) to ensure idempotent results.
        """
        use_compact = self.compact if compact is None else compact
        legal_actions: List[LegalAction] = []
        used_limit_cards = tracker.used_limit_cards if tracker else set()
        card_uses = obs.card_uses_remaining

        if obs.scene_type == "battle_turn":
            resolved_battle_pid = resolve_p_idol_id(
                plan=obs.plan,
                p_idol_id=getattr(tracker, "p_idol_id", None) if tracker is not None else None,
                deck_ids=getattr(tracker, "produce_deck", None) if tracker is not None else None,
                hand_cards=obs.cards,
            )
            battle_played_counts = getattr(tracker, "battle_played_counts", None) if tracker is not None else None

            # 1. Evaluate cards
            for card in obs.cards:
                card_dict = self._get_card_dict(card.card_id)
                if card_dict:
                    if card_dict.get("type"):
                        card.type = str(card_dict["type"])
                    if card_dict.get("rarity"):
                        card.rarity = str(card_dict["rarity"])

                card_data = {
                    "id": card.card_id,
                    "base_id": card_dict.get("base_id", card.card_id) if card_dict else card.card_id,
                    "limit_count": card_dict.get("limit_count") if card_dict else None,
                    "dsl_cost": card_dict.get("dsl_cost", "") if card_dict else "",
                    "dsl_conditions": card_dict.get("dsl_conditions", "") if card_dict else "",
                    "dsl_actions": card_dict.get("dsl_actions", "") if card_dict else "",
                    "dsl_effects": card_dict.get("dsl_effects", "") if card_dict else "",
                    "is_signature": card_dict.get("is_signature", 0) if card_dict else 0,
                    "force_initial_hand": card_dict.get("force_initial_hand", 0) if card_dict else 0,
                }

                removed_cnt = len(getattr(tracker, "battle_exhaust_pile", []) or []) if tracker is not None else 0
                inspected = inspect_card_cost_and_conditions(
                    card=card_data,
                    stamina=obs.stamina,
                    genki=obs.genki,
                    buffs=obs.buffs,
                    card_uses_remaining=card_uses,
                    used_limit_cards=used_limit_cards,
                    max_stamina=obs.max_stamina,
                    turn=obs.turn,
                    turns_remaining=obs.turns_remaining,
                    score=obs.score,
                    removed_count=removed_cnt,
                )

                playable = bool(inspected["playable"])
                reason = str(inspected["reason"])
                card.hp_loss = int(inspected["hp_loss"])
                card.cost_type = str(inspected["cost_type"])
                card.raw_cost = int(inspected["raw_cost"])
                card.effective_cost = int(inspected["effective_cost"])
                card.shield_cost = int(inspected["shield_cost"])
                card.condition_desc = str(inspected["condition_desc"])
                if card.visual_playable is False and playable:
                    playable = False
                    reason = f"卡面灰置未解锁({card.condition_desc or '条件或体力未满足'})"
                elif card.visual_playable is True and (not playable) and card_uses > 0:
                    c_base = card_data.get("base_id", card.card_id)
                    limit_only_fail = (
                        card_data.get("limit_count") == 1
                        and (card.card_id in used_limit_cards or c_base in used_limit_cards)
                    )
                    buff_only_fail = (
                        card.cost_type == "buff"
                        or any(k in reason for k in ("motivation", "goodImpressionTurns", "goodConditionTurns", "concentration", "干劲", "好印象", "好调", "集中", "不足"))
                    )
                    stamina_ocr_zero = (obs.stamina == 0 and card.cost_type in ("green", "red", "buff", "free"))
                    if (
                        (not limit_only_fail and buff_only_fail and obs.stamina > card.hp_loss)
                        or (limit_only_fail and (obs.stamina > card.hp_loss or stamina_ocr_zero))
                        or stamina_ocr_zero
                    ):
                        playable = True
                        reason = "实机画面亮起可打出(Buff/状态已就绪)"
                card.playable = playable
                card.reason = reason

                if not card.dsl_summary and card_dict:
                    raw_desc = self._resolve_card_description(card_dict)
                    dsl_sum = summarize_dsl_actions(card_dict.get("dsl_actions", ""))
                    if use_compact:
                        card.dsl_summary = dsl_sum
                    else:
                        card.dsl_summary = f"{raw_desc} [{dsl_sum}]".strip() if raw_desc and dsl_sum and dsl_sum not in raw_desc else (raw_desc or dsl_sum)
                elif use_compact and card.dsl_summary and len(card.dsl_summary) > 60:
                    card.dsl_summary = card.dsl_summary[:57] + "..."

                if card.playable:
                    choice_id = len(legal_actions)
                    raw_dsl_actions = card_data.get("dsl_actions") or ""
                    has_dsl_free_extra = "cardUsesRemaining+=" in raw_dsl_actions
                    custom_meta = get_memory_card_custom_meta(
                        obs.scenario_id,
                        resolved_battle_pid,
                        card.card_id,
                        battle_played_counts=battle_played_counts,
                    )
                    has_mem_free_extra = bool(custom_meta.get("grant_extra_use_now"))
                    is_sig = bool(card_data.get("is_signature") == 1 or card_data.get("force_initial_hand") == 1)
                    dsl_act_out = raw_dsl_actions
                    if has_mem_free_extra and "cardusesremaining+=" not in dsl_act_out.lower():
                        dsl_act_out = f"{dsl_act_out}; cardUsesRemaining+=1".strip("; ")
                    desc = f"打出卡牌: {card.name}"
                    if use_compact:
                        if card.hp_loss > 0:
                            desc += f" (需扣血{card.hp_loss})"
                        else:
                            desc += " (无实际扣血)"
                    else:
                        if is_sig and has_dsl_free_extra:
                            desc += " [⭐固有卡+1使用次数(不耗行动点)]"
                        elif has_dsl_free_extra:
                            desc += " [⚡免费+1使用次数(不耗行动点)]"
                        elif has_mem_free_extra:
                            desc += " [⚡メモリー免费再动+1使用次数(不耗行动点)]"
                        if custom_meta.get("exiles_trouble"):
                            desc += " [🧹免费除外1张眠気]"
                        if card.cost_type == "red":
                            desc += f" (🔴红血直扣{card.hp_loss}·无视护盾)"
                        elif card.cost_type == "green" and card.effective_cost > 0:
                            if card.hp_loss == 0:
                                desc += f" (🟢绿血{card.effective_cost}: 盾抵{card.shield_cost}+无实际扣血)"
                            else:
                                desc += f" (🟢绿血{card.effective_cost}: 盾抵{card.shield_cost}+需扣血{card.hp_loss})"
                        elif card.cost_type == "buff":
                            desc += " (🔷消耗Buff·无实际扣血)"
                        else:
                            desc += " (🟢0消耗·无实际扣血)"

                    legal_actions.append(
                        LegalAction(
                            choice_id=choice_id,
                            action_type="play_card",
                            description=desc,
                            bbox=card.bbox,
                            payload={
                                "card_id": card.card_id,
                                "base_id": card_data.get("base_id"),
                                "limit_count": card_data.get("limit_count"),
                                "card_type": (card_dict.get("type") if card_dict else None) or card.type or "active",
                                "dsl_actions": dsl_act_out,
                                "name": card.name,
                                "upgraded": card.upgraded,
                                "hp_loss": card.hp_loss,
                            },
                        )
                    )

            # 2. Evaluate drinks
            for drink in obs.drinks:
                drink_dict = self._get_drink_dict(drink.drink_id)
                if drink_dict and not drink.dsl_summary:
                    raw_desc = drink_dict.get("description") or ""
                    dsl_sum = summarize_dsl_actions(drink_dict.get("dsl_actions", ""))
                    drink.dsl_summary = dsl_sum if use_compact else (raw_desc or dsl_sum)

                if use_compact and drink.dsl_summary and len(drink.dsl_summary) > 40:
                    drink.dsl_summary = drink.dsl_summary[:37] + "..."

                if drink.playable:
                    choice_id = len(legal_actions)
                    desc = f"使用饮料: {drink.name}"
                    if drink.dsl_summary:
                        short_drink = drink.dsl_summary[:37] + "..." if (use_compact and len(drink.dsl_summary) > 40) else drink.dsl_summary
                        desc += f" ({short_drink})"
                    legal_actions.append(
                        LegalAction(
                            choice_id=choice_id,
                            action_type="use_drink",
                            description=desc,
                            bbox=drink.bbox,
                            payload={
                                "drink_id": drink.drink_id,
                                "name": drink.name,
                                "dsl_actions": (drink_dict.get("dsl_actions") if drink_dict else "") or "",
                                "dsl_summary": drink.dsl_summary or "",
                            },
                        )
                    )

            # 3. Always provide End Turn / SKIP in battle_turn
            choice_id = len(legal_actions)
            end_turn_bbox = BoundingBox(xmin=850, ymin=1700, xmax=1030, ymax=1850)
            end_turn_desc = "结束回合 (End Turn)"
            for opt in obs.options:
                if opt.get("action_type") == "end_turn" and opt.get("bbox"):
                    raw_b = opt["bbox"]
                    end_turn_bbox = raw_b if isinstance(raw_b, BoundingBox) else BoundingBox(**raw_b)
                    end_turn_desc = opt.get("description") or opt.get("title") or end_turn_desc
                    break
            has_playable_card = any(c.playable for c in obs.cards)
            if (not use_compact) and has_playable_card and obs.stamina >= 10:
                end_turn_desc += " (⚠️ 当前体力充沛且有可出牌，非万不得已勿点SKIP浪费回合)"
            legal_actions.append(
                LegalAction(
                    choice_id=choice_id,
                    action_type="end_turn",
                    description=end_turn_desc,
                    bbox=end_turn_bbox,
                    payload={},
                )
            )

        elif obs.scene_type in ("home", "produce_setup", "produce_schedule", "produce_event", "card_reward", "shop", "popup", "skip"):
            option_idol_guide = None
            if (not use_compact) and obs.scene_type in ("card_reward", "shop"):
                resolved_pid = resolve_p_idol_id(
                    plan=obs.plan,
                    p_idol_id=getattr(tracker, "p_idol_id", None) if tracker is not None else None,
                    deck_ids=getattr(tracker, "produce_deck", None) if tracker is not None else None,
                    hand_cards=obs.cards,
                )
                option_idol_guide = get_p_idol_guide(resolved_pid)

            for idx, opt in enumerate(obs.options):
                choice_id = len(legal_actions)
                title = opt.get("title") or opt.get("name") or f"选项{idx + 1}"
                desc = opt.get("description") or opt.get("effect") or ""
                if not desc and opt.get("card_id"):
                    c_row = self._get_card_dict(int(opt["card_id"]))
                    if c_row:
                        desc = c_row.get("description") or summarize_dsl_actions(c_row.get("dsl_actions", ""))
                elif not desc and opt.get("drink_id"):
                    d_row = self._get_drink_dict(int(opt["drink_id"]))
                    if d_row:
                        desc = d_row.get("description") or summarize_dsl_actions(d_row.get("dsl_actions", ""))
                if opt.get("price") and f"{opt['price']}P" not in str(desc):
                    desc = f"{opt['price']}P | {desc}".strip(" |")

                guide_note = ""
                if option_idol_guide is not None:
                    guide_note = get_card_guide_annotation(
                        option_idol_guide,
                        card_id=opt.get("card_id"),
                        card_name=str(title),
                    )
                text = f"选择: {title}{guide_note}"
                if desc:
                    text += f" ({desc})"

                opt_bbox = opt.get("bbox")
                if not isinstance(opt_bbox, BoundingBox):
                    if isinstance(opt_bbox, dict):
                        opt_bbox = BoundingBox(**opt_bbox)
                    elif isinstance(opt_bbox, (tuple, list)) and len(opt_bbox) == 4:
                        opt_bbox = BoundingBox(
                            xmin=opt_bbox[0],
                            ymin=opt_bbox[1],
                            xmax=opt_bbox[2],
                            ymax=opt_bbox[3],
                        )
                    else:
                        opt_bbox = BoundingBox(xmin=540, ymin=960, xmax=540, ymax=960)

                payload = dict(opt)
                payload.setdefault("scene_type", obs.scene_type)
                legal_actions.append(
                    LegalAction(
                        choice_id=choice_id,
                        action_type="pick_option",
                        description=text,
                        bbox=opt_bbox,
                        payload=payload,
                    )
                )

        return legal_actions

    def build_prompt(
        self,
        obs: ScreenObservation,
        tracker: Optional[GameStateTracker] = None,
        compact: Optional[bool] = None,
    ) -> Tuple[str, List[LegalAction]]:
        """
        Builds the prompt string and legal action list.
        When `compact=False` (default), generates the full rich prompt without token cap.
        """
        use_compact = self.compact if compact is None else compact
        legal_actions = self.generate_legal_actions_and_update_obs(obs, tracker, compact=use_compact)

        if use_compact:
            if obs.scene_type == "battle_turn":
                return self._build_compact_battle_prompt(obs, legal_actions, tracker), legal_actions
            return self._build_compact_option_prompt(obs, legal_actions), legal_actions

        if obs.scene_type == "battle_turn":
            return self._build_detailed_battle_prompt(obs, legal_actions, tracker), legal_actions
        return self._build_detailed_option_prompt(obs, legal_actions, tracker), legal_actions

    def _format_buff_str(self, buffs: Dict[str, int]) -> str:
        items = []
        for k, v in buffs.items():
            if v > 0:
                display_name = BUFF_NAME_MAP.get(k, k)
                unit = "回合" if ("Turns" in k or k in ("good_condition", "good_impression")) else "层/点"
                items.append(f"{display_name}: {v}{unit}")
        return "；".join(items) if items else "当前无激活Buff"

    def _format_used_limit_cards(self, tracker: Optional[GameStateTracker]) -> str:
        used_ids = sorted(tracker.used_limit_cards) if tracker else []
        used_names = []
        for uid in used_ids:
            c_row = self._get_card_dict(uid)
            used_names.append(f"{c_row['name']}(#{uid})" if c_row and c_row.get("name") else f"#{uid}")
        return "、".join(used_names) if used_names else "无（所有限1卡均尚未使用）"

    def _format_score_progress_line(
        self,
        obs: ScreenObservation,
        week_node: Any,
        is_exam_stage: bool,
    ) -> str:
        act_up = (obs.week_activity or "").upper()
        is_perfect_ring = ("PERFE" in act_up) or ("パーフェクト" in (obs.week_activity or ""))
        is_lesson_countdown = (not is_exam_stage) and (
            is_perfect_ring or ("CLEAR" in act_up) or (obs.target_score > 0 and obs.score == 0)
        )
        if is_lesson_countdown:
            stage_note = "（已过CLEAR、正冲击满额PERFECT！）" if is_perfect_ring else ""
            return (
                f"- 达标倒计时进度: 距离达成 CLEAR/PERFECT 还需 {obs.target_score} 分{stage_note} "
                f"(每次得分会从该数字往下倒扣，归 0 立即达标过关；目前累计实得参数: {obs.score})"
            )
        return (
            f"- 分数进度: 当前得分 {obs.score} / 目标得分 "
            f"{obs.target_score or week_node.target_score or '冲击最高分'}"
        )

    def _card_role_label(self, c_row: Optional[Dict[str, Any]], fallback_type: str = "active") -> str:
        ctype = (c_row.get("type") if c_row else None) or fallback_type or "active"
        limit_cnt = c_row.get("limit_count") if c_row else None
        if ctype == "mental":
            return " [M卡·叠Buff/引擎]"
        if ctype == "trouble":
            return " [T干扰卡]"
        return " [A卡·限1爆发]" if limit_cnt == 1 else " [A卡·主动得分]"

    def _drink_timing_tag(
        self,
        d_row: Optional[Dict[str, Any]],
        obs_drink: Optional[RecognizedDrink] = None,
        obs: Optional[ScreenObservation] = None,
        is_exam_stage: bool = False,
    ) -> str:
        dsl = (d_row.get("dsl_actions") if d_row else "") or ""
        desc = (d_row.get("description") if d_row else "") or (obs_drink.dsl_summary if obs_drink else "") or ""
        txt = f"{dsl} {desc}"
        did = (d_row.get("id") if d_row else None) or (obs_drink.drink_id if obs_drink else None)
        is_double_active = (
            did == 25
            or "doubleCardEffectCards" in dsl
            or "次に使用するアクティブスキルカード" in desc
        )
        if is_double_active:
            if obs is not None:
                playable_non_l_active: List[int] = []
                for c in obs.cards:
                    if not getattr(c, "playable", False):
                        continue
                    c_row = self._get_card_dict(c.card_id)
                    ctype = (c_row.get("type") if c_row else None) or getattr(c, "type", "active") or "active"
                    crar = (c_row.get("rarity") if c_row else None) or getattr(c, "rarity", "N") or "N"
                    if ctype == "active" and crar != "L" and int(c.card_id) != 781:
                        playable_non_l_active.append(int(c.card_id))
                if not playable_non_l_active:
                    return " [⛔本回合无可用非Legend主动A卡: 仅1T有效，严禁空喝作废!]"
                if any(cid in (777, 778) for cid in playable_non_l_active):
                    return " [🔥王炸配合就绪: 必须【先喝本饮料】再打《びしっとキメ顔》叠+400%强化与双倍追击!]"
                if is_exam_stage and obs.turns_remaining <= 3:
                    return f" [🔥终盘必喝双发: 仅剩{obs.turns_remaining}回合，先喝本饮料再打高伤A卡!]"
                return " [1T双发核爆: 优先留配《びしっとキメ顔》或终盘高伤A卡同回合先喝]"
            return " [1T双发核爆: 头号配《びしっとキメ顔》/次选终盘高伤A卡]"
        if obs is not None and is_exam_stage and obs.turns_remaining <= 4:
            if ("exchangeHand" in dsl or "入れ替え" in desc or "换牌" in desc) and any(c.playable for c in obs.cards) and obs.turns_remaining > 2:
                return " [换牌急救: 若手牌不差留到最后2回合]"
            return f" [🔥终盘必喝: 仅剩{obs.turns_remaining}回合不耗出牌点，立即喝光绝不留过终场!]"
        if "exchangeHand" in dsl or "入れ替え" in desc or "换牌" in desc:
            return " [换牌急救: 卡手再喝]"
        if "upgradeHand" in dsl or "強化" in desc:
            if obs is not None and sum(1 for c in obs.cards if not c.upgraded) >= 2 and any(c.playable for c in obs.cards):
                return " [🔥手牌≥2张未强化: 喝完立刻强化后再出牌收益更大!]"
            return " [手牌全强化: 等多核心卡同手]"
        if "perfectConditionTurns" in dsl or "絶好調" in desc or "绝好调" in desc:
            return " [1T绝好调: 配高分A卡同回合喝]"
        if "goodImpressionTurns" in dsl and "score+=" in dsl:
            return " [好印象转化爆发: 忌0层白喝]"
        if any(k in dsl for k in ("goodConditionTurns", "goodImpressionTurns", "motivation", "concentration", "setScoreBuff", "afterStartOfTurn", "halfCostTurns")):
            return " [持续Buff/引擎: 考试前中盘早喝，普通课囤]"
        if "genki+=" in dsl or "cardUsesRemaining" in dsl or "score+=" in dsl or "fixedStamina+=" in dsl or txt.strip():
            return " [补盾/加次/补分: 配合爆发或考试末盘清光]"
        return ""

    def _format_battle_pacing_banner(
        self,
        obs: ScreenObservation,
        is_exam_stage: bool,
        resolved_pid: Optional[int] = None,
    ) -> str:
        if is_exam_stage and resolved_pid == 163:
            if obs.turns_remaining >= 5:
                return (
                    f"- 🎯 阶段节奏: 【高速除外压缩期】(剩 {obs.turns_remaining} 回合) — "
                    "先打光+1使用数再动卡(含首次やる気は満点)与《ありのままのわたし》，"
                    "最后1点行动点优先打【限1】卡送入除外压缩牌库(《眠気》留给メモリー卡免费除外，绝不手打也不无故SKIP)；"
                    "《びしっとキメ顔》留中后盘配《特製ハツボシエキス》双发，《キセキの魔法》留末盘收割"
                )
            return (
                f"- ⚠️ 【终盘循环爆发期】阶段考试仅剩 {obs.turns_remaining} 回合！"
                "若有《特製ハツボシエキス》配《びしっとキメ顔》/高伤A卡务必【先喝再打】，"
                "围绕《オトメゴコロ+》→《みんな大好き+》→《認めてくれるひと+》(除外≥11达280%)高速循环并倾泻《キセキの魔法+》与其余可用饮料收割，绝不点SKIP！"
            )
        if is_exam_stage:
            if obs.turns_remaining >= 5:
                return (
                    f"- 🎯 阶段节奏: 【考试前中盘·M卡叠Buff蓄力期】(剩 {obs.turns_remaining} 回合) — "
                    "优先打 Mental(M) 卡叠高Buff与元气，可开持续引擎型饮料；暂缓普通基础A卡与限1转化爆发A卡"
                )
            return (
                f"- ⚠️ 【末盘抢分爆发期】阶段考试仅剩 {obs.turns_remaining} 回合！转入【后期A卡集中收割期】：\n"
                "  务必倾泻高伤/转化 A 卡（含《気分転換》《私を超えて(翔)》等限1大招）并将持有可用饮料全部用光兑现分数，绝勿带过终场或点SKIP！"
            )
        if obs.turns_remaining <= 3:
            if obs.target_score > 0:
                return (
                    f"- 🚨 【课程末盘清分死令】仅剩 {obs.turns_remaining} 回合且距离 CLEAR/PERFECT 还差 {obs.target_score} 分！"
                    "注意：剩余出牌=1时打无+1使用数的卡会立刻结束回合！必须【先打】『+1使用次数』免费A卡(如《私を超えて(翔)》)，"
                    "再打『🔥0扣血免费A卡』或高分A卡全力打满 CLEAR/PERFECT 0！严禁在有可打A卡时打0分纯M卡！若仅剩 1–2 回合手牌分不够，立即喝加分/属性饮料补足！"
                )
            return (
                f"- ⚠️ 【末盘抢分爆发期】课程仅剩 {obs.turns_remaining} 回合！抓紧打满 CLEAR/PERFECT 达标（普通课若手牌稳过仍留稀缺饮料给阶段考试）。"
            )
        if obs.turns_remaining == 4 and obs.target_score >= 15:
            return (
                f"- ⏰ 【课程中后盘转火期】仅剩 {obs.turns_remaining} 回合且仍差 {obs.target_score} 分："
                "若有已解锁的『+1使用次数』免费卡(如《私を超えて(翔)》)须先打出(否则打普通1费卡直接结束回合)，"
                "并可开始衔接高分/0扣血A卡稳步清分，避免最后2回合分差过大打不满！"
            )
        return (
            "- 💡 课程节奏: 【日常速通期】先适当铺 M 卡/固有启动，随后用 A 卡早打穿 CLEARまで 把剩余回合全部回血；手牌能稳过则保留稀缺 P 饮料给 W6/W13 阶段考试"
        )

    def _card_priority_tag(
        self,
        c_row: Optional[Dict[str, Any]],
        obs_card: Optional[RecognizedCard] = None,
        is_exam_stage: bool = False,
        scenario_id: Optional[str] = None,
        p_idol_id: Optional[int] = None,
        battle_played_counts: Optional[Dict[int, int]] = None,
    ) -> str:
        full_dsl = (c_row.get("dsl_actions") or "") if c_row else ""
        card_id = (c_row.get("id") if c_row else None) or (obs_card.card_id if obs_card else None)
        is_sig = bool(c_row and (c_row.get("is_signature") == 1 or c_row.get("force_initial_hand") == 1))
        is_free = "cardUsesRemaining+=" in full_dsl
        parts: List[str] = []
        if is_sig and is_free:
            parts.append(" 【⭐固有启动+免费+1使用数：不耗行动点，上手必须立刻打出！】")
        elif is_free:
            parts.append(" 【⚡免费+1使用数：不消耗本回合出牌次数】")
        custom_meta = get_memory_card_custom_meta(
            scenario_id,
            p_idol_id,
            card_id,
            battle_played_counts=battle_played_counts,
        )
        if custom_meta.get("tag"):
            parts.append(str(custom_meta["tag"]))
        ctype = ((c_row.get("type") if c_row else None) or (obs_card.type if obs_card else "active") or "active")
        limit_cnt = c_row.get("limit_count") if c_row else None
        if (not is_exam_stage) and obs_card is not None and obs_card.playable and obs_card.hp_loss == 0 and ctype == "active":
            parts.append(" 【🔥0扣血免费A卡】")
        elif is_exam_stage and limit_cnt == 1 and ctype == "active" and not is_free:
            if card_id in (777, 778):
                parts.append(" [🔥核心增幅A卡：留配《特製ハツボシエキス》开启4T双倍追击]")
            elif card_id not in (757, 758, 304, 305, 781):
                parts.append(" [⏳限1纯倍率终结A卡：前中盘勿早交，留末盘高Buff收割]")
        return "".join(parts)

    def _card_reuse_label(self, c_row: Optional[Dict[str, Any]]) -> str:
        limit_cnt = c_row.get("limit_count") if c_row else None
        return "[限1→打出进除外/未打进弃牌]" if limit_cnt == 1 else "[可重复→打出/未打均进弃牌]"

    def _format_session_hand_lines(
        self,
        cards: List[RecognizedCard],
        is_exam_stage: bool = False,
        scenario_id: Optional[str] = None,
        p_idol_id: Optional[int] = None,
        battle_played_counts: Optional[Dict[int, int]] = None,
    ) -> str:
        lines: List[str] = []
        for idx, c in enumerate(cards, 1):
            c_row = self._get_card_dict(c.card_id)
            sel_mark = " [★当前SELECT选中]" if getattr(c, "selected", False) else ""
            raw_cost = (c_row.get("dsl_cost") if c_row else "") or ""
            cost_mech = self._format_card_cost_mechanism(
                raw_cost,
                effective_cost=c.effective_cost,
                shield_cost=c.shield_cost,
                hp_loss=c.hp_loss,
                cost_type=c.cost_type,
            )
            cond_txt = c.condition_desc or format_card_conditions_zh(
                (c_row.get("dsl_conditions") if c_row else "") or "",
                dsl_cost=raw_cost,
                dsl_actions=(c_row.get("dsl_actions") if c_row else "") or "",
                dsl_effects=(c_row.get("dsl_effects") if c_row else "") or "",
            )
            status_txt = f"[✅可打出 | {cost_mech}]" if c.playable else f"[❌不可打 | {cost_mech} | 原因:{c.reason}]"
            cond_seg = f" | 条件/触发:{cond_txt}" if (cond_txt and cond_txt != "无特殊前置条件") else ""
            lines.append(
                f"  手牌{idx}: 《{c.name}》(#{c.card_id}){' [已强化+]' if c.upgraded else ''}{sel_mark}"
                f"{self._card_role_label(c_row, c.type)}"
                f"{self._card_reuse_label(c_row)}"
                f"{self._card_priority_tag(c_row, obs_card=c, is_exam_stage=is_exam_stage, scenario_id=scenario_id, p_idol_id=p_idol_id, battle_played_counts=battle_played_counts)}"
                f"{cond_seg} -> {status_txt}"
            )
        return "\n".join(lines) or "  - 当前手牌为空"

    def _format_session_drink_lines(
        self,
        drinks: List[RecognizedDrink],
        obs: ScreenObservation,
        is_exam_stage: bool,
    ) -> str:
        return "\n".join(
            f"  饮料{idx}: 《{d.name}》(#{d.drink_id})"
            f"{self._drink_timing_tag(self._get_drink_dict(d.drink_id), d, obs=obs, is_exam_stage=is_exam_stage)}"
            f" -> {'[本回合可用]' if d.playable else '[不可用]'}"
            for idx, d in enumerate(drinks, 1)
        ) or "  - 无饮料"

    def _format_card_archive_entry(
        self,
        card_id: int,
        obs_card: Optional[RecognizedCard] = None,
        note: str = "",
        idol_guide: Optional[Any] = None,
        scenario_id: Optional[str] = None,
        p_idol_id: Optional[int] = None,
    ) -> str:
        c_row = self._get_card_dict(card_id)
        name = (c_row.get("name") if c_row else None) or (obs_card.name if obs_card else f"#{card_id}")
        rarity = (c_row.get("rarity") if c_row else None) or (obs_card.rarity if obs_card else "N")
        ctype = (c_row.get("type") if c_row else None) or (obs_card.type if obs_card else "active")
        ctype_zh = {"active": "主动得分卡(Active)", "mental": "精神辅助卡(Mental)", "trouble": "干扰卡(Trouble)"}.get(ctype, ctype)
        limit_cnt = c_row.get("limit_count") if c_row else None
        limit_tag = (
            "不可重复使用(限1次：打出进【除外】不可再用，未打出进【弃牌】)"
            if limit_cnt == 1
            else "可重复使用(打出或未打出均进【弃牌】，【牌堆】出完后洗回)"
        )
        raw_cost = (c_row.get("dsl_cost") if c_row else "") or ""
        raw_cond = (c_row.get("dsl_conditions") if c_row else "") or ""
        full_dsl = (c_row.get("dsl_actions") if c_row else "") or ""
        full_eff = (c_row.get("dsl_effects") if c_row else "") or ""
        full_desc = self._resolve_card_description(c_row, obs_card.dsl_summary if obs_card else "")
        cost_mech = self._format_card_cost_mechanism(raw_cost)
        cond_zh = format_card_conditions_zh(raw_cond, dsl_cost=raw_cost, dsl_actions=full_dsl, dsl_effects=full_eff)
        dsl_zh = summarize_dsl_actions(full_dsl) if full_dsl else (obs_card.dsl_summary if obs_card else "")
        prio = self._card_priority_tag(
            c_row,
            scenario_id=scenario_id,
            p_idol_id=p_idol_id,
        )
        guide_tag = get_card_guide_annotation(idol_guide, card_id=card_id, card_name=str(name))
        mem_meta = get_memory_card_custom_meta(scenario_id, p_idol_id, card_id)
        mem_note = f" | {mem_meta['archive_note']}" if mem_meta.get("archive_note") else ""
        suffix = f" {note}" if note else ""
        return (
            f"  * 《{name}》(#{card_id}){prio}{guide_tag}{suffix} | {rarity} | {ctype_zh} | {limit_tag}\n"
            f"    消耗DSL: {raw_cost or '无消耗'} ({cost_mech}) | 触发/使用条件: {cond_zh} | 原文: {full_desc} | 解析: {dsl_zh}{mem_note}"
        )

    def _format_drink_archive_entry(
        self,
        drink_id: int,
        obs_drink: Optional[RecognizedDrink] = None,
        obs: Optional[ScreenObservation] = None,
        is_exam_stage: bool = False,
    ) -> str:
        d_row = self._get_drink_dict(drink_id)
        name = (d_row.get("name") if d_row else None) or (obs_drink.name if obs_drink else f"#{drink_id}")
        full_desc = (d_row.get("description") if d_row else "") or (obs_drink.dsl_summary if obs_drink else "") or "无描述"
        full_dsl = (d_row.get("dsl_actions") if d_row else "") or ""
        dsl_zh = summarize_dsl_actions(full_dsl) if full_dsl else (obs_drink.dsl_summary if obs_drink else "")
        d_tag = self._drink_timing_tag(d_row, obs_drink=obs_drink, obs=obs, is_exam_stage=is_exam_stage)
        return f"  * 《{name}》(#{drink_id}){d_tag} -> 原文: {full_desc} | 解析: {dsl_zh}"

    @staticmethod
    def _p_item_key(item: Dict[str, str]) -> str:
        return f"{item.get('name', 'P道具')}|{item.get('trigger', '')}|{item.get('effect', '')}"

    def _format_nonbattle_drink_cap_note(self, obs: ScreenObservation) -> str:
        """Render held P-drinks and full-slot cap decision hints for reward/shop prompts."""
        if obs.scene_type not in ("card_reward", "shop") or not obs.drinks:
            return ""
        max_slots = 4 if (obs.scenario_id in ("hif_honsen", "hif") or len(obs.drinks) > 3) else 3
        parts: List[str] = []
        for d in obs.drinks:
            d_row = self._get_drink_dict(d.drink_id)
            eff = d.dsl_summary or (d_row.get("description") if d_row else "") or ""
            eff_clean = eff.replace("\n", " ").strip()
            parts.append(f"《{d.name}》(#{d.drink_id}:{eff_clean})" if eff_clean else f"《{d.name}》(#{d.drink_id})")
        line = f"\n- 当前持有P饮料 ({len(obs.drinks)}/{max_slots}): " + "、".join(parts)
        has_cap_opts = len(obs.drinks) >= 3 and any(
            o.get("name") == "受け取らない" or str(o.get("name", "")).startswith("破棄候補_")
            for o in (obs.options or [])
        )
        if has_cap_opts:
            line += (
                f"\n- ⚠️ 【P饮料已满{len(obs.drinks)}/{max_slots}上限 · 请自主权衡】: "
                "① 若新饮料更强，点击上面饮料栏对应旧饮料(如低收益旧饮料)打开详情点『捨てる/破棄』删除腾位后再领取；"
                "② 若现有旧饮料更优，勾选『受け取らない』保留旧饮料直接跳过！"
            )
        return line

    def commit_session_sync(self, tracker: GameStateTracker, sync_meta: Dict[str, Any]) -> None:
        """Commit archived cards/drinks/P-items and snapshot after LLM receives the message."""
        tracker.commit_llm_sync(
            card_ids=sync_meta.get("card_ids"),
            drink_ids=sync_meta.get("drink_ids"),
            p_item_keys=sync_meta.get("p_item_keys"),
            week=sync_meta.get("week"),
            scene_type=sync_meta.get("scene_type"),
            in_battle=sync_meta.get("in_battle"),
        )

    def build_session_prompt(
        self,
        obs: ScreenObservation,
        tracker: Optional[GameStateTracker] = None,
        commit: bool = False,
    ) -> Tuple[str, List[LegalAction], Dict[str, Any]]:
        """
        Build either the one-time Produce initialization briefing (when not yet bootstrapped)
        or a concise incremental delta update referencing previously archived cards & P-items.
        Returns (prompt_text, legal_actions, sync_meta).
        """
        active_tracker = tracker if tracker is not None else GameStateTracker(plan=obs.plan)
        legal_actions = self.generate_legal_actions_and_update_obs(obs, active_tracker, compact=False)

        # Gather cards/drinks visible or in deck right now (before resolving P-Idol guide)
        card_map: Dict[int, Optional[RecognizedCard]] = {}
        ordered_cids: List[int] = []
        for cid in active_tracker.produce_deck + active_tracker.pending_top_deck:
            if cid and cid not in card_map:
                card_map[cid] = None
                ordered_cids.append(cid)
        # Pre-register derived signature token cards (#757/#758) when #755/#756 is in deck,
        # and signature cards (#919..#922) for ガラクタロード秦谷美鈴 (#163)
        if 755 in card_map and 757 not in card_map:
            card_map[757] = None
            ordered_cids.append(757)
        if 756 in card_map and 758 not in card_map:
            card_map[758] = None
            ordered_cids.append(758)
        if active_tracker.p_idol_id == 163 or any(cid in card_map for cid in (919, 920, 921, 922)):
            for mcid in (919, 920, 921, 922):
                if mcid not in card_map:
                    card_map[mcid] = None
                    ordered_cids.append(mcid)
        for c in obs.cards:
            if c.card_id not in card_map:
                ordered_cids.append(c.card_id)
            card_map[c.card_id] = c
        for opt in obs.options:
            ocid = opt.get("card_id")
            if ocid:
                ocid_i = int(ocid)
                if ocid_i not in card_map:
                    card_map[ocid_i] = None
                    ordered_cids.append(ocid_i)

        resolved_pid = resolve_p_idol_id(
            plan=obs.plan,
            p_idol_id=active_tracker.p_idol_id,
            deck_ids=ordered_cids,
            hand_cards=obs.cards,
        )
        idol_guide = get_p_idol_guide(resolved_pid)

        ctx = compute_week_stats_and_stage_bonus(
            scenario_id=obs.scenario_id,
            week=obs.week,
            plan=obs.plan,
            override_stats=obs.idol_stats if obs.idol_stats else None,
            p_idol_id=resolved_pid,
        )
        spec = get_scenario_spec(obs.scenario_id)
        week_node = ctx["week_node"]
        guide = ctx["playstyle_guide"]
        p_items = obs.p_items if obs.p_items else ctx["p_items"]
        stage_bonus = obs.stage_bonuses if obs.stage_bonuses else ctx["stage_bonuses"]
        stats = ctx["idol_stats"]

        vo_pct = stage_bonus.get("vocal_bonus_pct", 150.0)
        da_pct = stage_bonus.get("dance_bonus_pct", 140.0)
        vi_pct = stage_bonus.get("visual_bonus_pct", 130.0)
        stage_rule = stage_bonus.get("special_rule", "")

        drink_map: Dict[int, Optional[RecognizedDrink]] = {}
        ordered_dids: List[int] = []
        for d in obs.drinks:
            if d.drink_id not in drink_map:
                ordered_dids.append(d.drink_id)
            drink_map[d.drink_id] = d
        for opt in obs.options:
            odid = opt.get("drink_id")
            if odid:
                odid_i = int(odid)
                if odid_i not in drink_map:
                    drink_map[odid_i] = None
                    ordered_dids.append(odid_i)

        all_p_keys = {self._p_item_key(it) for it in p_items}
        is_bootstrap = not active_tracker.session_bootstrapped

        sync_meta: Dict[str, Any] = {
            "is_bootstrap": is_bootstrap,
            "card_ids": set(ordered_cids),
            "drink_ids": set(ordered_dids),
            "p_item_keys": all_p_keys,
            "week": obs.week,
            "scene_type": obs.scene_type,
            "in_battle": bool(obs.scene_type == "battle_turn" or active_tracker.in_battle),
        }

        actions_str = "\n".join(f"  [{act.choice_id}] {act.description}" for act in legal_actions) or "  [0] 确认"
        is_exam_stage = (
            any(k in (obs.week_activity or "") for k in ("試験", "ラウンド", "本戦", "FINAL"))
            or (obs.week in (spec.midterm_week, spec.final_week))
        )
        score_progress_line = self._format_score_progress_line(obs, week_node, is_exam_stage)
        buff_str = self._format_buff_str(obs.buffs)
        used_str = self._format_used_limit_cards(active_tracker)
        if obs.scene_type == "battle_turn" and active_tracker.produce_deck:
            active_tracker.sync_observed_hand(
                obs.cards,
                db=self.db,
                turn=obs.turn,
                turns_remaining=obs.turns_remaining,
            )
        deck_summary_str = (
            active_tracker.get_deck_prompt_summary(self.db, obs.cards if obs.scene_type == "battle_turn" else None)
            if active_tracker.produce_deck
            else "  - 尚未初始化卡库记录"
        )

        if is_bootstrap:
            roadmap_lines = []
            for w in range(1, spec.total_weeks + 1):
                wn = get_week_node(obs.scenario_id, w)
                marker = "👉 [当前]" if w == obs.week else ("✓ [已过]" if w < obs.week else "  [后续]")
                roadmap_lines.append(f"  {marker} 第{w:02d}周: {wn.title} -> 推荐: {wn.recommended_action}")
            roadmap_str = "\n".join(roadmap_lines)
            mechanics_lines = "\n".join(f"  - {m}" for m in guide["core_mechanics"])
            p_items_archive = "\n".join(
                f"  {idx}. 【{it.get('name', 'P道具')}】({it.get('trigger', '常驻')}): {it.get('effect', '')}"
                for idx, it in enumerate(p_items, 1)
            ) or "  - 无额外P道具"
            cards_archive = "\n".join(
                self._format_card_archive_entry(
                    cid,
                    obs_card=card_map.get(cid),
                    idol_guide=idol_guide,
                    scenario_id=obs.scenario_id,
                    p_idol_id=resolved_pid,
                )
                for cid in ordered_cids
            ) or "  - 暂无卡牌档案"
            drinks_archive = "\n".join(
                self._format_drink_archive_entry(
                    did,
                    obs_drink=drink_map.get(did),
                    obs=obs if obs.scene_type == "battle_turn" else None,
                    is_exam_stage=is_exam_stage,
                )
                for did in ordered_dids
            ) or "  - 当前未持有P饮料"
            idol_guide_block = (
                f"\n\n{format_idol_session_guide_block(idol_guide)}"
                if idol_guide is not None
                else ""
            )

            if obs.scene_type == "battle_turn":
                pacing_banner = self._format_battle_pacing_banner(obs, is_exam_stage, resolved_pid=resolved_pid)
                idol_nudge = format_idol_turn_nudge(
                    idol_guide,
                    obs,
                    exhaust_count=len(active_tracker.battle_exhaust_pile),
                )
                pacing_with_nudge = f"{pacing_banner}\n{idol_nudge}" if idol_nudge else pacing_banner
                hand_brief = self._format_session_hand_lines(
                    obs.cards,
                    is_exam_stage=is_exam_stage,
                    scenario_id=obs.scenario_id,
                    p_idol_id=resolved_pid,
                    battle_played_counts=active_tracker.battle_played_counts,
                )
                drinks_brief = self._format_session_drink_lines(obs.drinks, obs, is_exam_stage)
                shield_summary = active_tracker.get_shield_prompt_summary()
                live_state_block = (
                    f"- 场景: 战斗第 {obs.turn} 回合 (剩余 {obs.turns_remaining} 回合) | 剩余出牌次数: {obs.card_uses_remaining}\n"
                    f"- 体力: {obs.stamina} / {obs.max_stamina} | 盾(元气): {obs.genki} | 后台状态: {shield_summary}\n"
                    f"{score_progress_line}\n"
                    f"{pacing_with_nudge}\n"
                    f"- 场上Buff: {buff_str}\n"
                    f"- 本战已用限1: {used_str}\n"
                    f"【本场四区牌库实时追踪（手牌/牌堆/弃牌/除外｜牌堆出完后才将弃牌洗回牌堆，除外不可用）】\n{deck_summary_str}\n"
                    f"【当前手牌】\n{hand_brief}\n"
                    f"【当前持有饮料】\n{drinks_brief}"
                )
            else:
                shop_note = (
                    f"\n- 🛒 商店P点铁律: 当前持有 {obs.p_points}P；若P点数不足购买心仪饮料/换牌/强化，必须直接选择「終了 / スキップ」离开，严禁反复切换预览！"
                    if obs.scene_type == "shop"
                    else ""
                )
                interval_note = (
                    "\n- ⚡ 【H.I.F 商店/インターバル铁律】: インターバル无删卡功能！切忌乱买普通『スキルカード交換』新卡塞胖牌库破坏终盘循环！请按顺序花光P点：① 买强力P饮料补满空槽 → ② 走『セレクトチェンジ / チェンジ』1换1换强卡(或6日目削除废卡) → ③ 余钱点『スキルカード強化 / カスタマイズ』强化自定义主力卡！买完再选『終了 / 次へ』。"
                    if (spec.scenario_id == "hif_honsen" and (obs.week in (6, 8) or "インターバル" in (obs.week_activity or "")))
                    else ""
                )
                drink_cap_note = self._format_nonbattle_drink_cap_note(obs)
                live_state_block = (
                    f"- 场景: {obs.scene_type} | 第 {obs.week} 阶段 — {obs.week_activity or week_node.title}\n"
                    f"- 体力: {obs.stamina} / {obs.max_stamina} | 持有P点数: {obs.p_points}P | 三维: Vo {stats.get('vocal', 150)} / Da {stats.get('dance', 135)} / Vi {stats.get('visual', 120)}{shop_note}{interval_note}{drink_cap_note}\n"
                    f"- 整局卡库构成:\n{deck_summary_str}"
                )

            pre_events_block = ""
            if active_tracker.llm_pending_events:
                pre_lines = "\n".join(f"  - {ev}" for ev in active_tracker.llm_pending_events)
                pre_events_block = f"\n【建档前系统已自动执行操作】\n{pre_lines}"

            prompt_text = f"""【学园偶像大师 (Gakumas) 全培育持久化单会话 · 唯一开局建档】
(提示：以下剧本总览、流派与角色P卡专属指导、抽牌四区规则、初始卡库与道具档案仅在本次培育开局发送这一次，请牢记！后续整局只同步增量状态、四区牌库变化与新获卡牌/道具。)

==================== 一、剧本流程与当前周目 ====================
- 所属剧本: {spec.name} (共 {spec.total_weeks} 周，属性上限 {spec.stat_cap})
- 当前位置: 第 {obs.week} / {spec.total_weeks} 周 — {obs.week_activity or week_node.title}
- 本周指引: {week_node.operation_guide}
- 当前舞台倍率: Vocal {vo_pct}% | Dance {da_pct}% | Visual {vi_pct}% ({stage_rule})
【剧本全周目路线表】
{roadmap_str}

==================== 二、职业流派核心玩法 ({guide['name']}) ====================
{mechanics_lines}
- 出牌节奏: {guide['turn_playstyle']}
【打牌抽卡与四区牌库循环规则（手牌 / 牌堆 / 弃牌 / 除外）及盾/绿血/红血机制】
- 🛡️盾(元气)与🟢绿血 vs 🔴红血机制:
  * **🛡️护盾(元気)**: 护盾优先抵扣🟢绿血消耗(`cost-=`)，受【干劲】加成（每次通过卡牌获得护盾时额外+干劲层数），且可被《私を超えて(翔)》《気分転換》按比例转化为高额得分。
  * **🟢绿血(`cost-=X`, 消費体力)**: 先扣除🛡️护盾、不够再扣真实体力；受【减耗/羽毛 `halfCostTurns`】减半与免耗效果影响。
  * **🔴红血(`stamina-=X`, 体力消費)**: **无视护盾、不受减耗影响，直接扣除真实体力！** 即使盾再高，红血<=真实体力也不足打出（会灰置不可选）。
  * **🎯卡牌触发条件**: 凡带前置条件的卡（如《私を超えて(翔)》需当前🛡️盾≥15；部分卡需好调/好印象/干劲/姿态达标）只有条件与体力同时满足才解锁；启动卡内的长期触发机制（如《私を超えて》后续每打2张元气卡往牌堆顶生(翔)）需提前规划手牌顺序。
- 卡牌能否重复使用: 未标记「限1」(`limit_count!=1`)的卡可重复使用；标记「限1」(`limit_count=1`)的卡每场仅能使用1次。
- 进入【弃牌】: 打出可重复使用的牌立即进【弃牌】；回合结束时【手牌】中所有未打出的牌(含未打出限1卡)全部进【弃牌】。
- 抽牌与转移顺序: 每次抽牌严格优先从【牌堆】抽；必须等【牌堆】中的牌全部出完后，才会把【弃牌】洗切转移回【牌堆】；【牌堆】未空前【弃牌】不会被抽到。
- 【除外】区不可再用: 打出的限1卡进入【除外】，本场永久移除不能再使用也不回牌堆。
- 后台每场课程均已建立对应四区牌库与盾血台账并向你同步变化，请据此算牌规划。{idol_guide_block}

==================== 三、初始被动道具档案 (P-Items，已建档) ====================
{p_items_archive}

==================== 四、初始卡库与衍生卡 / 饮料完整档案 (已建档，后续不重发) ====================
【初始卡库及衍生卡详细效果】
{cards_archive}
【初始P饮料详细效果】
{drinks_archive}

==================== 五、当前实时面板与手牌状态 ====================
{live_state_block}{pre_events_block}

==================== 七、合法动作列表与决策输出 ====================
{actions_str}

【开局建档完毕 · 请做出本局第1步决策】
严格仅输出标准 JSON: {{"choice_id": <动作编号int>, "reason": "<简短理由>"}}"""
        else:
            # Incremental delta message: ONLY events, new cards/drinks/items, current stats & legal actions
            diff_added, diff_removed = active_tracker.compute_deck_diff()
            delta_notes: List[str] = []
            for ev in active_tracker.llm_pending_events:
                delta_notes.append(f"  - {ev}")

            if active_tracker.llm_last_week != obs.week:
                delta_notes.append(
                    f"  - 【周目切换】进入第 {obs.week}/{spec.total_weeks} 周: {obs.week_activity or week_node.title} "
                    f"(三维 Vo {stats.get('vocal', 150)} / Da {stats.get('dance', 135)} / Vi {stats.get('visual', 120)}) | 指引: {week_node.operation_guide}"
                )
            if obs.scene_type == "battle_turn" and not active_tracker.llm_last_in_battle:
                delta_notes.append(
                    f"  - 【进入新战斗】{obs.week_activity or week_node.title} | 舞台倍率 Vo {vo_pct}% / Da {da_pct}% / Vi {vi_pct}% | 已为本场建立四区牌库(卡库置入【牌堆】，弃牌/除外已重置)"
                )
            if diff_added or diff_removed:
                add_names = [
                    f"《{(self._get_card_dict(cid) or {}).get('name') or f'#{cid}'}》(#{cid})"
                    for cid in diff_added
                ]
                rem_names = [
                    f"《{(self._get_card_dict(cid) or {}).get('name') or f'#{cid}'}》(#{cid})"
                    for cid in diff_removed
                ]
                parts = []
                if add_names:
                    parts.append("新增 " + "、".join(add_names))
                if rem_names:
                    parts.append("移除/被替换 " + "、".join(rem_names))
                delta_notes.append(f"  - 【卡库变动】{'；'.join(parts)} (当前总计 {len(active_tracker.produce_deck)} 张)")

            new_card_ids = [cid for cid in ordered_cids if cid not in active_tracker.llm_synced_card_ids]
            new_drink_ids = [did for did in ordered_dids if did not in active_tracker.llm_synced_drink_ids]
            new_p_items = [it for it in p_items if self._p_item_key(it) not in active_tracker.llm_synced_p_items]

            delta_section = "\n".join(delta_notes) if delta_notes else "  - 连续决策中（无跨周或自动跳过事件）"
            new_archives: List[str] = []
            if new_card_ids:
                new_archives.append("【新卡牌档案收录（请记住，后续直接简称）】\n" + "\n".join(
                    self._format_card_archive_entry(
                        cid,
                        obs_card=card_map.get(cid),
                        idol_guide=idol_guide,
                        scenario_id=obs.scenario_id,
                        p_idol_id=resolved_pid,
                    )
                    for cid in new_card_ids
                ))
            if new_drink_ids:
                new_archives.append("【新P饮料档案收录】\n" + "\n".join(
                    self._format_drink_archive_entry(
                        did,
                        obs_drink=drink_map.get(did),
                        obs=obs if obs.scene_type == "battle_turn" else None,
                        is_exam_stage=is_exam_stage,
                    )
                    for did in new_drink_ids
                ))
            if new_p_items:
                new_archives.append("【新P道具档案收录】\n" + "\n".join(
                    f"  * 【{it.get('name', 'P道具')}】({it.get('trigger', '常驻')}): {it.get('effect', '')}"
                    for it in new_p_items
                ))
            archive_block = ("\n" + "\n".join(new_archives)) if new_archives else ""

            if obs.scene_type == "battle_turn":
                pacing_banner = self._format_battle_pacing_banner(obs, is_exam_stage, resolved_pid=resolved_pid)
                idol_nudge = format_idol_turn_nudge(
                    idol_guide,
                    obs,
                    exhaust_count=len(active_tracker.battle_exhaust_pile),
                )
                pacing_with_nudge = f"{pacing_banner}\n{idol_nudge}" if idol_nudge else pacing_banner
                hand_concise = self._format_session_hand_lines(
                    obs.cards,
                    is_exam_stage=is_exam_stage,
                    scenario_id=obs.scenario_id,
                    p_idol_id=resolved_pid,
                    battle_played_counts=active_tracker.battle_played_counts,
                )
                drinks_concise = self._format_session_drink_lines(obs.drinks, obs, is_exam_stage)
                shield_summary = active_tracker.get_shield_prompt_summary()
                state_block = (
                    f"- 进度: 第{obs.week}周 {obs.week_activity or week_node.title} | 舞台倍率 Vo {vo_pct}% / Da {da_pct}% / Vi {vi_pct}% | 第 {obs.turn} 回合(剩余 {obs.turns_remaining} 回合) | 剩余出牌: {obs.card_uses_remaining}\n"
                    f"- 面板与盾血台账: {shield_summary} | Buff: {buff_str}\n"
                    f"{score_progress_line}\n"
                    f"{pacing_with_nudge}\n"
                    f"- 本战已用限1: {used_str}\n"
                    f"【本场四区牌库实时变化（手牌/牌堆/弃牌/除外｜牌堆出完才将弃牌洗回牌堆，除外不可用）】\n{deck_summary_str}\n"
                    f"【当前手牌（🟢绿血扣盾→🔴红血穿盾｜含实时触发条件与解锁状态）】\n{hand_concise}\n"
                    f"【持有饮料（完整效果见前序建档）】\n{drinks_concise}"
                )
            else:
                shop_note = (
                    f"\n- 🛒 商店P点决策: 当前持有 {obs.p_points}P；P点数不足购买任何商品或已无可买项时，必须直接选「終了 / スキップ」离开商店！"
                    if obs.scene_type == "shop"
                    else ""
                )
                interval_note = (
                    "\n- ⚡ 【H.I.F 商店/インターバル铁律】: インターバル无删卡功能！切忌乱买普通『スキルカード交換』新卡塞胖牌库破坏终盘循环！请按顺序花光P点：① 买强力P饮料补满空槽 → ② 走『セレクトチェンジ / チェンジ』1换1换强卡(或6日目削除废卡) → ③ 余钱点『スキルカード強化 / カスタマイズ』强化自定义主力卡！买完再选『終了 / 次へ』。"
                    if (spec.scenario_id == "hif_honsen" and (obs.week in (6, 8) or "インターバル" in (obs.week_activity or "")))
                    else ""
                )
                drink_cap_note = self._format_nonbattle_drink_cap_note(obs)
                state_block = (
                    f"- 场景: 第 {obs.week} 阶段 {obs.scene_type} ({obs.week_activity or week_node.title})\n"
                    f"- 面板: 体力 {obs.stamina}/{obs.max_stamina} | 持有P点数: {obs.p_points}P | 三维 Vo {stats.get('vocal', 150)} / Da {stats.get('dance', 135)} / Vi {stats.get('visual', 120)}{shop_note}{interval_note}{drink_cap_note}\n"
                    f"{deck_summary_str}"
                )

            prompt_text = f"""【第 {obs.week} 周 · 增量状态同步 ({obs.scene_type})】
【本步变动与执行回执】
{delta_section}{archive_block}

【当前即时状态】
{state_block}

【合法动作列表】
{actions_str}

请直接调用已记忆的卡库/道具档案结合上述状态快速决策，严格输出 JSON: {{"choice_id": <int>, "reason": "<str>"}}"""

        if commit:
            self.commit_session_sync(active_tracker, sync_meta)

        return prompt_text, legal_actions, sync_meta

    def _build_detailed_battle_prompt(
        self,
        obs: ScreenObservation,
        legal_actions: List[LegalAction],
        tracker: Optional[GameStateTracker] = None,
    ) -> str:
        resolved_pid = resolve_p_idol_id(
            plan=obs.plan,
            p_idol_id=getattr(tracker, "p_idol_id", None) if tracker is not None else None,
            deck_ids=getattr(tracker, "produce_deck", None) if tracker is not None else None,
            hand_cards=obs.cards,
        )
        idol_guide = get_p_idol_guide(resolved_pid)
        ctx = compute_week_stats_and_stage_bonus(
            scenario_id=obs.scenario_id,
            week=obs.week,
            plan=obs.plan,
            override_stats=obs.idol_stats if obs.idol_stats else None,
            p_idol_id=resolved_pid,
        )
        spec = get_scenario_spec(obs.scenario_id)
        week_node = ctx["week_node"]
        guide = ctx["playstyle_guide"]
        p_items = obs.p_items if obs.p_items else ctx["p_items"]
        stage_bonus = obs.stage_bonuses if obs.stage_bonuses else ctx["stage_bonuses"]
        stats = ctx["idol_stats"]

        # 1. Playstyle mechanics
        mechanics_lines = "\n".join(f"  - {m}" for m in guide["core_mechanics"])
        idol_guide_block = (
            f"\n\n{format_idol_session_guide_block(idol_guide)}"
            if idol_guide is not None
            else ""
        )

        # 2. Passive P-Items
        p_item_lines = []
        for idx, item in enumerate(p_items, 1):
            p_item_lines.append(
                f"  {idx}. 【{item.get('name', 'P道具')}】 触发条件: {item.get('trigger', '常驻')} -> 效果: {item.get('effect', '')}"
            )
        p_items_str = "\n".join(p_item_lines) if p_item_lines else "  - 无额外被动P道具"

        # 3. Stage bonuses
        vo_pct = stage_bonus.get("vocal_bonus_pct", 150.0)
        da_pct = stage_bonus.get("dance_bonus_pct", 140.0)
        vi_pct = stage_bonus.get("visual_bonus_pct", 130.0)
        stage_rule = stage_bonus.get("special_rule", "")

        # 4. Active Buffs & Used Limit Cards
        buff_str = self._format_buff_str(obs.buffs)
        used_str = self._format_used_limit_cards(tracker)

        is_exam_stage = (
            any(k in (obs.week_activity or "") for k in ("試験", "ラウンド", "本戦", "FINAL"))
            or (obs.week in (spec.midterm_week, spec.final_week))
        )

        # 5. Drawn Hand Cards (Full Effects)
        card_lines = []
        for idx, card in enumerate(obs.cards, 1):
            c_row = self._get_card_dict(card.card_id)
            rarity = (c_row.get("rarity") if c_row else None) or card.rarity or "N"
            ctype = (c_row.get("type") if c_row else None) or card.type or "active"
            ctype_zh = {"active": "主动得分卡(Active)", "mental": "精神辅助卡(Mental)", "trouble": "干扰卡(Trouble)"}.get(ctype, ctype)
            limit_cnt = c_row.get("limit_count") if c_row else None
            limit_tag = (
                "【不可重复使用(限1：打出进【除外】不可再用/未打进【弃牌】)】"
                if limit_cnt == 1
                else "【可重复使用(打出或未打均进【弃牌】，【牌堆】出完后洗回)】"
            )
            upg_tag = " [已强化+]" if card.upgraded else ""
            raw_cost = (c_row.get("dsl_cost") if c_row else "") or ""
            raw_cond = (c_row.get("dsl_conditions") if c_row else "") or ""
            full_desc = self._resolve_card_description(c_row, card.dsl_summary)
            full_dsl = (c_row.get("dsl_actions") if c_row else "") or ""
            full_eff = (c_row.get("dsl_effects") if c_row else "") or ""
            cost_mech = self._format_card_cost_mechanism(
                raw_cost,
                effective_cost=card.effective_cost,
                shield_cost=card.shield_cost,
                hp_loss=card.hp_loss,
                cost_type=card.cost_type,
            )
            cond_zh = card.condition_desc or format_card_conditions_zh(raw_cond, dsl_cost=raw_cost, dsl_actions=full_dsl, dsl_effects=full_eff)
            dsl_zh = summarize_dsl_actions(full_dsl) if full_dsl else card.dsl_summary
            priority_tag = self._card_priority_tag(
                c_row,
                obs_card=card,
                is_exam_stage=is_exam_stage,
                scenario_id=obs.scenario_id,
                p_idol_id=resolved_pid,
                battle_played_counts=getattr(tracker, "battle_played_counts", None) if tracker is not None else None,
            )

            status_str = (
                f"[✅可打出 | {cost_mech}]"
                if card.playable
                else f"[❌不可打出 | {cost_mech} | 原因: {card.reason}]"
            )
            card_lines.append(
                f"  手牌{idx}: 《{card.name}》(#{card.card_id}){upg_tag}{priority_tag} | 稀有度:{rarity} | 类型:{ctype_zh} | {limit_tag}\n"
                f"    - 消耗与状态: {status_str} (消耗机制: {cost_mech})\n"
                f"    - 触发与前置条件: {cond_zh}\n"
                f"    - 卡牌效果原文: {full_desc}\n"
                f"    - 效果数值解析: {dsl_zh}"
            )
        cards_section = "\n".join(card_lines) if card_lines else "  - 当前手牌为空"

        stamina_rule_tip = (
            "本战为阶段考试，剩血不带入下周！长回合考试分两段打：【前中盘(剩>=5回合)优先M卡叠Buff与引擎 + 尽早喝持续引擎饮料；后期(剩<=4回合)倾泻爆发A卡与全部饮料】，切勿因开局分低盲目甩掉限1爆发A卡或打低收益基础A卡"
            if is_exam_stage
            else "《初》课程机制：提前打满目标分通关后，【剩余未用回合会自动兑换为体力回复】！适当起引擎后用A卡速通；稀缺P饮料默认囤给阶段考试（除非满格或急需救场），体力>=10且有有效牌严禁点SKIP"
        )
        pacing_banner = self._format_battle_pacing_banner(obs, is_exam_stage, resolved_pid=resolved_pid)
        idol_nudge = format_idol_turn_nudge(
            idol_guide,
            obs,
            exhaust_count=len(tracker.battle_exhaust_pile) if tracker is not None else 0,
        )
        pacing_with_nudge = f"{pacing_banner}\n{idol_nudge}" if idol_nudge else pacing_banner
        score_progress_line = self._format_score_progress_line(obs, week_node, is_exam_stage)
        if tracker is not None and tracker.produce_deck:
            tracker.sync_observed_hand(
                obs.cards,
                db=self.db,
                turn=obs.turn,
                turns_remaining=obs.turns_remaining,
            )
        deck_summary_str = (
            tracker.get_deck_prompt_summary(self.db, obs.cards)
            if (tracker is not None and tracker.produce_deck)
            else "  - 未启用整局卡库记录（仅基于当前手牌判断）"
        )

        # 6. Available P-Drinks (Full Effects)
        drink_lines = []
        for idx, drink in enumerate(obs.drinks, 1):
            d_row = self._get_drink_dict(drink.drink_id)
            full_desc = (d_row.get("description") if d_row else "") or drink.dsl_summary or "无描述"
            full_dsl = (d_row.get("dsl_actions") if d_row else "") or ""
            dsl_zh = summarize_dsl_actions(full_dsl) if full_dsl else drink.dsl_summary
            status_str = "[本回合可使用]" if drink.playable else "[不可使用]"
            d_tag = self._drink_timing_tag(d_row, obs_drink=drink, obs=obs, is_exam_stage=is_exam_stage)
            drink_lines.append(
                f"  饮料{idx}: 《{drink.name}》(#{drink.drink_id}){d_tag} {status_str}\n"
                f"    - 饮料效果原文: {full_desc}\n"
                f"    - 效果数值解析: {dsl_zh}"
            )
        drinks_section = "\n".join(drink_lines) if drink_lines else "  - 当前无可用P饮料"

        # 7. Legal Actions
        actions_str = "\n".join(f"  [{act.choice_id}] {act.description}" for act in legal_actions)

        return f"""【学园偶像大师 (Gakumas) 智能打牌完整上下文输入】

==================== 一、剧本流程与当前周目 ====================
- 所属剧本: {spec.name} (共 {spec.total_weeks} 周，属性上限 {spec.stat_cap})
- 当前周目: 第 {obs.week} / {spec.total_weeks} 周 — {obs.week_activity or week_node.title}
- 本周阶段指引: {week_node.operation_guide}

==================== 二、职业流派核心玩法 ({guide['name']}) ====================
{mechanics_lines}
- 阶段出牌节奏: {guide['turn_playstyle']}{idol_guide_block}

==================== 三、被动道具加成 (P-Items) ====================
{p_items_str}

==================== 四、场景与舞台审查加成 ====================
- 当前舞台/课程: {stage_bonus.get('stage_name', week_node.title)}
- 三维得分倍率加成: Vocal {vo_pct}% | Dance {da_pct}% | Visual {vi_pct}%
- 场景特殊规则: {stage_rule}

==================== 五、初始与当前面板数值 ====================
- 偶像当前三维属性: Vocal {stats.get('vocal', 180)} / Dance {stats.get('dance', 160)} / Visual {stats.get('visual', 150)}
- 回合进度: 第 {obs.turn} 回合 (剩余 {obs.turns_remaining} 回合) | 本回合剩余出牌次数: {obs.card_uses_remaining}
{pacing_with_nudge}
- 生存与盾血机制: 当前体力 {obs.stamina} / 最大体力 {obs.max_stamina} | 当前🛡️护盾(元気): {obs.genki}（🟢绿血`cost-=`先扣盾再扣血、受减半；🔴红血`stamina-=`无视护盾直接扣真实体力！）
{f"- 后台盾血台账: {tracker.get_shield_prompt_summary()}" if tracker is not None else ""}
- 💡 体力与SKIP规则: {stamina_rule_tip}
{score_progress_line}
- 场上已生效 Buff: {buff_str}
- 本局已消耗的「限1」卡牌: {used_str}

【抽牌四区规则与完整卡库/本场牌堆实时追踪（手牌/牌堆/弃牌/除外）】
- 🔄 抽牌流转规则: 摸牌仅从【牌堆】抽；可重复卡打出或【手牌】未打出(含未打出限1卡)均进【弃牌】；**只有【牌堆】中的牌全部出完后，才将【弃牌】转移回【牌堆】**；限1卡打出后进【除外】，本场不可再用。
{deck_summary_str}

==================== 六、打牌阶段：当前抽到的手牌与可用饮料效果 ====================
【当前抽到的手牌明细】
{cards_section}

【当前持有的 P 饮料明细】
{drinks_section}

==================== 七、合法动作列表与决策输出 ====================
{actions_str}

【决策要求】
请综合以上【职业玩法】、【被动道具加成】、【场景加成】、【初始与当前面板数值】以及【当前抽到的手牌与饮料完整效果】，选择本步收益最高的合法动作编号。
请严格仅输出以下 JSON 格式（不要输出 Markdown 代码块以外的多余文字）：
{{"choice_id": <动作编号int>, "reason": "<结合流派机制、加成与手牌效果的决策理由>"}}"""

    def _build_detailed_option_prompt(
        self,
        obs: ScreenObservation,
        legal_actions: List[LegalAction],
        tracker: Optional[GameStateTracker] = None,
    ) -> str:
        resolved_pid = resolve_p_idol_id(
            plan=obs.plan,
            p_idol_id=getattr(tracker, "p_idol_id", None) if tracker is not None else None,
            deck_ids=getattr(tracker, "produce_deck", None) if tracker is not None else None,
            hand_cards=obs.cards,
        )
        idol_guide = get_p_idol_guide(resolved_pid)
        ctx = compute_week_stats_and_stage_bonus(
            scenario_id=obs.scenario_id,
            week=obs.week,
            plan=obs.plan,
            override_stats=obs.idol_stats if obs.idol_stats else None,
            p_idol_id=resolved_pid,
        )
        spec = get_scenario_spec(obs.scenario_id)
        week_node = ctx["week_node"]
        guide = ctx["playstyle_guide"]
        p_items = obs.p_items if obs.p_items else ctx["p_items"]
        stats = ctx["idol_stats"]
        idol_guide_block = (
            f"\n\n{format_idol_session_guide_block(idol_guide)}"
            if idol_guide is not None
            else ""
        )

        # Full 13-week roadmap overview so LLM knows what week we are in and what comes next
        roadmap_lines = []
        for w in range(1, spec.total_weeks + 1):
            wn = get_week_node(obs.scenario_id, w)
            marker = "👉 [当前周]" if w == obs.week else ("✓ [已完成]" if w < obs.week else "  [后续周]")
            roadmap_lines.append(f"  {marker} 第{w:02d}周: {wn.title} -> 推荐: {wn.recommended_action}")
        roadmap_str = "\n".join(roadmap_lines)

        p_items_str = "\n".join(
            f"  - 【{it.get('name', 'P道具')}】({it.get('trigger', '')}): {it.get('effect', '')}"
            for it in p_items
        )
        actions_str = "\n".join(f"  [{act.choice_id}] {act.description}" for act in legal_actions) or "  [0] 确认"

        stage_bonus = obs.stage_bonuses if obs.stage_bonuses else ctx["stage_bonuses"]
        vo_pct = stage_bonus.get("vocal_bonus_pct", 150.0)
        da_pct = stage_bonus.get("dance_bonus_pct", 140.0)
        vi_pct = stage_bonus.get("visual_bonus_pct", 130.0)
        stage_rule = stage_bonus.get("special_rule", "")

        deck_summary_opt = (
            tracker.get_deck_prompt_summary(self.db)
            if (tracker is not None and tracker.produce_deck)
            else "  - 未启用整局卡库记录"
        )

        drink_cap_opt = self._format_nonbattle_drink_cap_note(obs)
        interval_opt = (
            "- ⚡ 【H.I.F 商店/インターバル铁律】: インターバル无删卡功能！严禁乱买普通『スキルカード交換』新卡塞胖牌库破坏终盘循环；优先买强力P饮料补满空槽，走『セレクトチェンジ / チェンジ』换牌(或6日目削除废卡)，余钱全部用于『スキルカード強化 / カスタマイズ』强化自定义主力卡！买完再选『終了 / 次へ』。\n"
            if (spec.scenario_id == "hif_honsen" and (obs.week in (6, 8) or "インターバル" in (obs.week_activity or "")))
            else ""
        )
        return f"""【学园偶像大师 (Gakumas) 剧本周目日程/选卡完整上下文输入】

==================== 一、剧本流程与当前周目 ====================
- 剧本名称: {spec.name} (共 {spec.total_weeks} 周，属性上限 {spec.stat_cap})
- 识别到的当前周目: 第 {obs.week} / {spec.total_weeks} 周 — {obs.week_activity or week_node.title}
- 本周操作流程指引: {week_node.operation_guide}

【剧本全周目流程表】
{roadmap_str}

==================== 二、职业流派核心玩法 ({guide['name']}) ====================
{chr(10).join('  - ' + m for m in guide['core_mechanics'])}
- 阶段出牌节奏: {guide['turn_playstyle']}{idol_guide_block}

==================== 三、被动道具加成 (P-Items) ====================
{p_items_str}

==================== 四、场景与舞台审查加成 ====================
- 当前周目场景: {stage_bonus.get('stage_name', week_node.title)}
- 三维得分倍率加成: Vocal {vo_pct}% | Dance {da_pct}% | Visual {vi_pct}%
- 场景特殊规则: {stage_rule}

==================== 五、初始与当前面板数值 ====================
- 偶像当前三维属性: Vocal {stats.get('vocal', 180)} | Dance {stats.get('dance', 160)} | Visual {stats.get('visual', 150)}
- 当前体力: {obs.stamina} / {obs.max_stamina} | 当前元气: {obs.genki} | 持有P点数: {obs.p_points}P{drink_cap_opt}
{interval_opt}{("- 🛒 商店P点提示: P点数不足购买心仪卡/饮料/强化时，直接选择「終了 / スキップ」离开商店，勿反复切换预览。" + chr(10)) if obs.scene_type == "shop" else ""}- 当前整局卡库构成:
{deck_summary_opt}

==================== 六、当前周目候选选项与效果 ====================
{actions_str}

==================== 七、合法动作列表与决策输出 ====================
{actions_str}

【决策要求】
请严格根据当前识别到的周目（第 {obs.week} 周：{week_node.title}）、体力状态与【{guide['name']}】构筑需求，选择最佳动作编号。
严格输出 JSON 格式：
{{"choice_id": <动作编号int>, "reason": "<结合当前第{obs.week}周流程与流派构筑的决策理由>"}}"""

    def _build_compact_battle_prompt(
        self,
        obs: ScreenObservation,
        legal_actions: List[LegalAction],
        tracker: Optional[GameStateTracker] = None,
    ) -> str:
        buff_items = []
        for k, v in obs.buffs.items():
            if v > 0:
                name_map = {
                    "goodConditionTurns": "好调",
                    "perfectConditionTurns": "绝好调",
                    "concentration": "集中",
                    "motivation": "干劲",
                    "goodImpressionTurns": "好印象",
                    "fullPowerCharge": "全力充能",
                    "nullifyCostCards": "免除消耗",
                    "halfCostTurns": "消耗减半",
                    "doubleCostTurns": "消耗加倍",
                    "cardUsesRemaining": "额外出牌",
                }
                display_name = name_map.get(k, k)
                unit = "回合" if "Turns" in k else ""
                buff_items.append(f"{display_name}:{v}{unit}")
        buff_str = ", ".join(buff_items) if buff_items else "无"

        used_ids = list(tracker.used_limit_cards) if tracker else []
        used_str = str(used_ids) if used_ids else "无"

        cards_str_lines = []
        for card in obs.cards:
            status_tag = "[可用]" if card.playable else f"[不可用:{card.reason}]"
            cost_info = f"实际扣血{card.hp_loss}"
            summary = card.dsl_summary
            if summary and len(summary) > 60:
                summary = summary[:57] + "..."
            eff_info = f"效果: {summary}" if summary else ""
            cards_str_lines.append(f"- {card.name} {status_tag} ({cost_info} | {eff_info})")
        cards_str = "\n".join(cards_str_lines) if cards_str_lines else "- 无手牌"

        actions_str = "\n".join(f"[{act.choice_id}] {act.description}" for act in legal_actions)
        plan_display = {
            "sense": "sense (感性)",
            "logic": "logic (理性)",
            "anomaly": "anomaly (异常)",
        }.get(obs.plan.lower(), obs.plan)

        return f"""【学园偶像大师 对战决策输入】
职业: {plan_display} | 回合: 第{obs.turn}回合 (剩余{obs.turns_remaining}回合)
得分: {obs.score} (目标: {obs.target_score or '最高'}) | 体力: {obs.stamina}/{obs.max_stamina} | 元气(护盾): {obs.genki}
Buff状态: {buff_str}
已消耗限1卡牌: {used_str}

【手牌明细】
{cards_str}

【可用动作列表】
{actions_str}

【决策要求】
请结合当前职业流派核心机制做出收益最大化的决策。
请严格输出以下 JSON 格式，禁止输出任何额外文字或分析：
{{"choice_id": <动作编号>, "reason": "<简短决策理由>"}}"""

    def _build_compact_option_prompt(
        self,
        obs: ScreenObservation,
        legal_actions: List[LegalAction],
    ) -> str:
        actions_str = "\n".join(f"[{act.choice_id}] {act.description}" for act in legal_actions) or "[0] 确认"
        scene_name = {
            "produce_schedule": "培育日程选择",
            "card_reward": "卡牌三选一奖励",
            "shop": "商店购买",
            "popup": "提示弹窗确认",
        }.get(obs.scene_type, obs.scene_type)

        return f"""【学园偶像大师 场景选择输入】
当前场景: {scene_name}
体力: {obs.stamina}/{obs.max_stamina}

【可用选项列表】
{actions_str}

【决策要求】
从以上选项中选出最佳项。严格输出以下 JSON 格式：
{{"choice_id": <动作编号>, "reason": "<简短决策理由>"}}"""
