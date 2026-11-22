"""
Game rules and cost/playability evaluation engine for Gakumas.
"""

import math
import re
from typing import Any, Dict, Optional, Set, Tuple


def summarize_dsl_actions(dsl_actions: str) -> str:
    """
    Translates raw DSL action strings into concise, human-and-LLM-readable summaries.
    """
    if not dsl_actions:
        return ""

    actions = [a.strip() for a in dsl_actions.split(";") if a.strip()]
    summaries = []

    for action in actions:
        # Score
        m = re.match(r"^score\+=(\d+)$", action)
        if m:
            summaries.append(f"得分+{m.group(1)}")
            continue
        m = re.match(r"^score\+=([a-zA-Z]+)\*([\d\.]+)", action)
        if m:
            var_sc, mult_sc = m.group(1), m.group(2)
            if var_sc == "genki":
                summaries.append(f"按护盾(元気)x{mult_sc}得分")
            elif var_sc == "goodImpressionTurns":
                pct_sc = int(round(float(mult_sc) * 100))
                summaries.append(f"按好印象×{pct_sc}%得分")
            elif var_sc == "motivation":
                pct_sc = int(round(float(mult_sc) * 100))
                summaries.append(f"按干劲×{pct_sc}%得分")
            else:
                summaries.append(f"得分+{var_sc}x{mult_sc}")
            continue
        m_var_sc = re.match(r"^score\+=([a-zA-Z]+)$", action)
        if m_var_sc:
            var_sc = m_var_sc.group(1)
            if var_sc == "goodImpressionTurns":
                summaries.append("按好印象×100%得分")
                continue
            elif var_sc == "genki":
                summaries.append("按护盾(元気)×100%得分")
                continue
        m_gimb = re.search(r"setGoodImpressionTurnsEffectBuff\(([\d\.]+)(?:,(\d+))?\)", action)
        if m_gimb:
            pct_b = int(round(float(m_gimb.group(1)) * 100))
            dur_b = m_gimb.group(2)
            summaries.append(f"好印象强化+{pct_b}%({dur_b}T)" if dur_b else f"好印象强化+{pct_b}%")
            continue
        m_rem = re.search(r"if:countCards\[removed\]>=(\d+)\s*\{\s*([^}]+)\}", action)
        if m_rem:
            thresh = m_rem.group(1)
            inner = summarize_dsl_actions(m_rem.group(2).strip())
            summaries.append(f"除外≥{thresh}→{inner or m_rem.group(2).strip()}")
            continue

        # Genki & Stamina
        m = re.match(r"^genki/=2\b", action)
        if m:
            summaries.append("护盾减半")
            continue
        m = re.match(r"^genki\+=(\d+)", action)
        if m:
            summaries.append(f"元气+{m.group(1)}")
            continue
        m = re.match(r"^fixedStamina\+=(\d+)", action)
        if m:
            summaries.append(f"体力+{m.group(1)}")
            continue
        m = re.match(r"^fixedStamina\-=(\d+)", action)
        if m:
            summaries.append(f"红血直扣{m.group(1)}")
            continue

        # Buffs
        m = re.match(r"^concentration\+=(\d+)", action)
        if m:
            summaries.append(f"集中+{m.group(1)}")
            continue
        m = re.match(r"^motivation\+=(\d+)", action)
        if m:
            summaries.append(f"干劲+{m.group(1)}")
            continue
        m = re.match(r"^goodConditionTurns\+=(\d+)", action)
        if m:
            summaries.append(f"好调+{m.group(1)}回合")
            continue
        m = re.match(r"^goodImpressionTurns\+=(\d+)", action)
        if m:
            summaries.append(f"好印象+{m.group(1)}回合")
            continue
        m = re.match(r"^perfectConditionTurns\+=(\d+)", action)
        if m:
            summaries.append(f"绝好调+{m.group(1)}回合")
            continue
        m = re.match(r"^fullPowerCharge\+=(\d+)", action)
        if m:
            summaries.append(f"全力充能+{m.group(1)}")
            continue
        m = re.match(r"^cardUsesRemaining\+=(\d+)", action)
        if m:
            summaries.append(f"出牌次数+{m.group(1)}")
            continue

        # Stances & utilities
        if "setStance(strength" in action:
            summaries.append("进入强气姿态")
        elif "setStance(preservation" in action:
            summaries.append("进入温存姿态")
        elif "setStance(fullPower" in action:
            summaries.append("进入全力姿态")
        elif "upgradeHand" in action:
            summaries.append("手牌全部升级")
        elif "exchangeHand" in action:
            summaries.append("重置手牌")
        elif "drawCard" in action:
            summaries.append("抽卡")
        elif "halfCostTurns" in action:
            m_hc = re.search(r"halfCostTurns\+=(\d+)", action)
            summaries.append(f"绿血消耗减半+{m_hc.group(1)}回合" if m_hc else "消耗体力减半")
        elif "doubleCostTurns" in action:
            summaries.append("消耗体力加倍")
        else:
            # Fallback to truncated raw action
            clean = re.sub(r"[\{\}\[\]]", "", action).strip()
            if len(clean) > 25:
                clean = clean[:22] + "..."
            summaries.append(clean)

    # Deduplicate summaries while preserving order
    deduped = []
    seen = set()
    for s in summaries:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    result = ", ".join(deduped)
    if len(result) > 60:
        result = result[:57] + "..."
    return result


def evaluate_single_condition(condition_expr: str, context: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Evaluates a single condition expression (e.g. 'if:goodConditionTurns>=1', 'if:isFullPower').
    Returns (satisfied: bool, reason: str).
    """
    clean_expr = condition_expr.strip()
    if clean_expr.startswith("@usable"):
        clean_expr = clean_expr[len("@usable"):].strip()
    if clean_expr.startswith("if:"):
        clean_expr = clean_expr[len("if:"):].strip()

    if not clean_expr:
        return True, ""

    # Stance boolean checks
    if clean_expr == "isFullPower":
        is_ok = bool(context.get("isFullPower") or context.get("stance") == "fullPower")
        return is_ok, "" if is_ok else "需处于全力姿态"

    if clean_expr == "isPreservation":
        stance = str(context.get("stance", "none"))
        is_ok = bool(context.get("isPreservation") or stance in ("preservation", "preservation2"))
        return is_ok, "" if is_ok else "需处于温存姿态"

    if clean_expr == "isStrength":
        stance = str(context.get("stance", "none"))
        is_ok = bool(context.get("isStrength") or stance in ("strength", "strength2"))
        return is_ok, "" if is_ok else "需处于强气姿态"

    # Comparison checks: LHS OP RHS
    match = re.match(r"^([a-zA-Z_]+)(>=|<=|==|!=|>|<)(.+)$", clean_expr)
    if not match:
        # Unknown expression, treat as satisfied or log
        return True, ""

    lhs, op, rhs = match.group(1), match.group(2), match.group(3).strip()

    # Determine LHS value
    lhs_val = context.get(lhs, 0)

    # Determine RHS value
    if rhs == "none":
        rhs_val = "none"
        lhs_val = str(lhs_val)
    elif rhs == "strength2":
        rhs_val = "strength2"
        lhs_val = str(lhs_val)
    elif "maxStamina" in rhs:
        # e.g. maxStamina*0.5
        max_stam = float(context.get("maxStamina", 30))
        factor = float(rhs.replace("maxStamina*", "").replace("maxStamina", "1"))
        rhs_val = max_stam * factor
        lhs_val = float(lhs_val)
    else:
        try:
            rhs_val = float(rhs)
            lhs_val = float(lhs_val)
        except ValueError:
            rhs_val = rhs
            lhs_val = str(lhs_val)

    # Perform comparison
    satisfied = False
    if op == ">=":
        satisfied = lhs_val >= rhs_val
    elif op == "<=":
        satisfied = lhs_val <= rhs_val
    elif op == "==":
        satisfied = lhs_val == rhs_val
    elif op == "!=":
        satisfied = lhs_val != rhs_val
    elif op == ">":
        satisfied = lhs_val > rhs_val
    elif op == "<":
        satisfied = lhs_val < rhs_val

    if not satisfied:
        return False, f"条件未满足: {lhs}{op}{rhs} (当前值: {lhs_val})"

    return True, ""


_COST_BUFF_ZH: Dict[str, str] = {
    "concentration": "集中",
    "motivation": "干劲(やる気)",
    "goodConditionTurns": "好调",
    "goodImpressionTurns": "好印象",
    "perfectConditionTurns": "绝好调",
    "fullPowerCharge": "全力充能",
}


def _describe_single_condition_zh(cond_expr: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Translate a single DSL usability condition into clear Chinese with optional live satisfaction status."""
    clean = cond_expr.strip()
    if clean.startswith("@usable"):
        clean = clean[len("@usable") :].strip()
    if clean.startswith("if:"):
        clean = clean[len("if:") :].strip()
    if not clean:
        return ""

    desc = ""
    if clean == "isFullPower":
        desc = "需处于全力姿态"
    elif clean == "isPreservation":
        desc = "需处于温存姿态"
    elif clean == "isStrength":
        desc = "需处于强气姿态"
    elif clean == "stance==strength2":
        desc = "需处于强气Lv2姿态"
    elif clean == "stance!=none":
        desc = "需处于任意姿态(非无姿态)"
    else:
        m = re.match(r"^([a-zA-Z_]+)(>=|<=|==|!=|>|<)(.+)$", clean)
        if m:
            lhs, op, rhs = m.group(1), m.group(2), m.group(3).strip()
            op_zh = {"==": "=", "!=": "≠", ">=": "≥", "<=": "≤"}.get(op, op)
            if lhs == "genki":
                desc = "需当前护盾(元気)=0" if (op == "==" and rhs == "0") else f"需当前护盾(元気){op_zh}{rhs}"
            elif lhs == "motivation":
                desc = f"需干劲(やる気){op_zh}{rhs}"
            elif lhs == "goodImpressionTurns":
                desc = f"需好印象{op_zh}{rhs}层"
            elif lhs == "goodConditionTurns":
                desc = f"需好调{op_zh}{rhs}回合"
            elif lhs == "perfectConditionTurns":
                desc = f"需绝好调{op_zh}{rhs}回合"
            elif lhs == "concentration":
                desc = f"需集中{op_zh}{rhs}"
            elif lhs == "stamina" and "maxStamina" in rhs:
                pct = int(round(float(rhs.replace("maxStamina*", "").replace("maxStamina", "1")) * 100))
                desc = f"需当前体力≥最大体力的{pct}%"
            elif lhs == "score":
                desc = f"需当前得分{op_zh}{rhs}"
            elif lhs == "turnsElapsed":
                desc = f"需已过回合≥{rhs}(第{int(float(rhs)) + 1}回合起)"
            elif lhs == "turnsRemaining":
                desc = f"限最后1回合(剩余回合={rhs})" if rhs == "1" else f"需剩余回合{op_zh}{rhs}"
            elif lhs == "fullPowerCharge":
                desc = f"需全力充能{op_zh}{rhs}"
            elif lhs == "cumulativeFullPowerCharge":
                desc = f"需累计全力充能{op_zh}{rhs}"
            elif lhs == "fullPowerTimes":
                desc = f"需本场进过全力姿态≥{rhs}次"
            elif lhs == "preservationTimes":
                desc = f"需本场进入温存≥{rhs}次"
            elif lhs == "stanceChangedTimes":
                desc = f"需本场切换姿态≥{rhs}次"
            else:
                desc = f"需满足 {lhs}{op_zh}{rhs}"
        else:
            desc = f"需满足条件 {clean}"

    if context is not None:
        ok, _ = evaluate_single_condition(clean, context)
        if ok:
            return f"{desc}[✅已满足]"
        cur_hint = ""
        m = re.match(r"^([a-zA-Z_]+)(>=|<=|==|!=|>|<)(.+)$", clean)
        if m:
            lhs = m.group(1)
            cur_val = context.get(lhs, 0)
            cur_hint = f",当前={cur_val}"
        return f"{desc}[❌未满足{cur_hint}]"
    return desc


def format_card_conditions_zh(
    dsl_conditions: str,
    dsl_cost: str = "",
    dsl_actions: str = "",
    context: Optional[Dict[str, Any]] = None,
    dsl_effects: str = "",
) -> str:
    """Summarize card usability prerequisites, buff costs, and embedded effect triggers in clear Chinese."""
    parts: list[str] = []

    cost_s = (dsl_cost or "").strip()
    m_cost = re.match(r"^([a-zA-Z]+)-=(\d+)$", cost_s)
    if m_cost and m_cost.group(1) not in ("cost", "stamina"):
        var, amt = m_cost.group(1), int(m_cost.group(2))
        var_zh = _COST_BUFF_ZH.get(var, var)
        if context is not None:
            cur_v = int(context.get(var, 0) or 0)
            mark = "✅已够" if cur_v >= amt else f"❌不足(当前{cur_v})"
            parts.append(f"消耗{var_zh}{amt}[{mark}]")
        else:
            parts.append(f"消耗{var_zh}{amt}")

    cond_s = (dsl_conditions or "").strip()
    if cond_s:
        for sub_c in [c.strip() for c in cond_s.split(";") if c.strip()]:
            zh = _describe_single_condition_zh(sub_c, context=context)
            if zh:
                parts.append(zh)

    act_s = dsl_actions or ""
    eff_s = dsl_effects or ""
    if "useRandomCardFree[baseId(919)]" in eff_s or ("baseId(919)" in eff_s and "removed" in eff_s):
        parts.append("触发机制(再演): 本卡打出进【除外】后，后续每次直接增加好印象每满3次自动免费再使用(每T限1·共2次)")
    elif "addCardToTopOfDeck(757)" in act_s or ("cardHasEffect(genki)" in act_s and "at:cardUsed" in act_s):
        parts.append("触发机制: 本场后续每打出2张带元气(盾)效果的卡(第1、3、5…张)，自动在【牌堆】顶生成《私を超えて(翔)》")
    elif "at:goodImpressionTurnsIncreased" in act_s:
        parts.append("触发机制: 后续直接效果增加好印象后自动触发追加效果")
    elif "at:cardUsed[mental]" in act_s:
        m_b = re.search(r"at:cardUsed\[mental\]\s*\{\s*([a-zA-Z]+)\+=(\d+)", act_s)
        if m_b:
            bzh = _COST_BUFF_ZH.get(m_b.group(1), m_b.group(1))
            parts.append(f"触发机制: 后续每打出M(精神)卡追加{bzh}+{m_b.group(2)}")
        else:
            parts.append("触发机制: 后续打出M(精神)卡时自动触发额外增益")
    elif "at:cardUsed[active]" in act_s:
        parts.append("触发机制: 后续打出A(主动)卡时自动触发额外增益")
    elif "at:endOfTurn" in act_s:
        m_e = re.search(r"at:endOfTurn\s*\{\s*if:([^{]+)\{", act_s)
        if m_e:
            parts.append(f"回合末条件触发({_describe_single_condition_zh(m_e.group(1), context=None)})")
        else:
            parts.append("触发机制: 回合结束时自动触发持续增益")

    rem_matches = list(re.finditer(r"if:countCards\[removed\]>=(\d+)", act_s))
    if rem_matches:
        cur_rem = int(context.get("removedCards", 0) or 0) if context is not None else None
        for m_r in rem_matches:
            th = int(m_r.group(1))
            if cur_rem is not None:
                mark_r = f"✅已满足(当前{cur_rem}张)" if cur_rem >= th else f"❌未达(当前{cur_rem}/{th}张)"
                parts.append(f"附加条件: 除外区≥{th}张[{mark_r}]")
            else:
                parts.append(f"附加条件: 除外区≥{th}张触发强力追加")
    elif "if:" in act_s:
        for m_if in re.finditer(r"if:([a-zA-Z0-9_>=<!.*]+)", act_s):
            parts.append(f"附加效果条件({_describe_single_condition_zh(m_if.group(1), context=None)})")
            break

    return "；".join(parts) if parts else "无特殊前置条件"


def inspect_card_cost_and_conditions(
    card: Dict[str, Any],
    stamina: int,
    genki: int,
    buffs: Dict[str, int],
    card_uses_remaining: int,
    used_limit_cards: Set[int],
    max_stamina: int = 30,
    turn: int = 1,
    turns_remaining: int = 10,
    score: int = 0,
    stance: str = "none",
    removed_count: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate card usability and inspect full cost breakdown (green shielded vs red bypass-shield vs buff),
    net shield consumption, net HP loss, and Chinese trigger/condition explanation.
    """
    card_id = card.get("id", 0)
    base_id = card.get("base_id", card_id)
    dsl_cost = (card.get("dsl_cost") or "").strip()
    dsl_cond = (card.get("dsl_conditions") or "").strip()
    dsl_actions = (card.get("dsl_actions") or "").strip()
    dsl_effects = (card.get("dsl_effects") or "").strip()

    context: Dict[str, Any] = {
        "stamina": stamina,
        "maxStamina": max_stamina,
        "genki": genki,
        "score": score,
        "turn": turn,
        "turnsElapsed": max(0, turn - 1),
        "turnsRemaining": turns_remaining,
        "removedCards": removed_count,
        "stance": stance or buffs.get("stance", "none"),
        "isFullPower": stance == "fullPower" or bool(buffs.get("isFullPower")),
        "isPreservation": stance in ("preservation", "preservation2") or bool(buffs.get("isPreservation")),
        "isStrength": stance in ("strength", "strength2") or bool(buffs.get("isStrength")),
    }
    context.update(buffs)

    cost_type = "free"
    raw_cost = 0
    effective_cost = 0
    shield_cost = 0
    actual_hp_cost = 0
    buff_cost_fail = ""

    if dsl_cost:
        cost_match = re.match(r"^([a-zA-Z]+)-=(\d+)$", dsl_cost)
        if cost_match:
            var = cost_match.group(1)
            raw_cost = int(cost_match.group(2))
            if var == "cost":
                if buffs.get("nullifyCostCards", 0) > 0:
                    effective_cost = 0
                elif buffs.get("halfCostTurns", 0) > 0:
                    effective_cost = math.ceil(raw_cost / 2)
                elif buffs.get("doubleCostTurns", 0) > 0:
                    effective_cost = raw_cost * 2
                else:
                    effective_cost = raw_cost
                cost_type = "free" if raw_cost == 0 else "green"
                shield_cost = min(max(0, genki), effective_cost)
                actual_hp_cost = max(0, effective_cost - max(0, genki))
            elif var == "stamina":
                cost_type = "red"
                effective_cost = raw_cost
                shield_cost = 0
                actual_hp_cost = raw_cost
            else:
                cost_type = "buff"
                effective_cost = raw_cost
                curr_buff_val = buffs.get(var, 0)
                if curr_buff_val < raw_cost:
                    buff_cost_fail = f"{var}不足(需{raw_cost},当前{curr_buff_val})"

    # Also account for fixedStamina-= in dsl_actions if paired with a green cost
    m_fixed = re.search(r"(?<![a-zA-Z])fixedStamina-=(\d+)", dsl_actions)
    if m_fixed and cost_type != "red":
        fixed_loss = int(m_fixed.group(1))
        if fixed_loss > 0:
            actual_hp_cost += fixed_loss
            if cost_type == "free":
                cost_type = "red"
                raw_cost = fixed_loss
                effective_cost = fixed_loss

    condition_desc = format_card_conditions_zh(
        dsl_cond,
        dsl_cost=dsl_cost,
        dsl_actions=dsl_actions,
        context=context,
        dsl_effects=dsl_effects,
    )

    playable = True
    reason = "可用"
    if card.get("limit_count") == 1 and (card_id in used_limit_cards or base_id in used_limit_cards):
        playable = False
        reason = "本场对战已使用过(限1次)"
    elif card_uses_remaining <= 0:
        playable = False
        reason = "本回合出牌次数已耗尽"
    elif buff_cost_fail:
        playable = False
        reason = buff_cost_fail
    elif stamina <= actual_hp_cost:
        playable = False
        if cost_type == "red":
            reason = f"体力不足(🔴红血直扣{actual_hp_cost}无视护盾,当前体力{stamina})"
        else:
            reason = f"体力不足(需扣血{actual_hp_cost},当前体力{stamina})"
    elif dsl_cond:
        for sub_c in [c.strip() for c in dsl_cond.split(";") if c.strip()]:
            is_satisfied, fail_reason = evaluate_single_condition(sub_c, context)
            if not is_satisfied:
                playable = False
                reason = f"{fail_reason} ({_describe_single_condition_zh(sub_c, context=context)})"
                break

    return {
        "playable": playable,
        "hp_loss": actual_hp_cost,
        "reason": reason,
        "cost_type": cost_type,
        "raw_cost": raw_cost,
        "effective_cost": effective_cost,
        "shield_cost": shield_cost,
        "condition_desc": condition_desc,
    }


def evaluate_card_cost_and_usability(
    card: Dict[str, Any],
    stamina: int,
    genki: int,
    buffs: Dict[str, int],
    card_uses_remaining: int,
    used_limit_cards: Set[int],
    max_stamina: int = 30,
    turn: int = 1,
    turns_remaining: int = 10,
    score: int = 0,
    stance: str = "none",
    removed_count: int = 0,
) -> Tuple[bool, int, str]:
    """
    Evaluates whether a card can be played in the current state and computes
    net HP loss after genki absorption.

    Returns:
        (playable: bool, actual_hp_cost: int, reason: str)
    """
    res = inspect_card_cost_and_conditions(
        card=card,
        stamina=stamina,
        genki=genki,
        buffs=buffs,
        card_uses_remaining=card_uses_remaining,
        used_limit_cards=used_limit_cards,
        max_stamina=max_stamina,
        turn=turn,
        turns_remaining=turns_remaining,
        score=score,
        stance=stance,
        removed_count=removed_count,
    )
    return bool(res["playable"]), int(res["hp_loss"]), str(res["reason"])
