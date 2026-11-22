"""
Comprehensive Test Suite for demo/agent/fallback.py and Heuristic Decision Engine.
Tests zero-token single action skip, battle heuristics (Sense, Logic, Anomaly, suicidal avoidance, drink buffs),
produce schedule heuristics (stamina < 15 threshold, SP lesson vs regular lesson vs consultation),
card reward heuristics, and popup/shop dismissal.
"""

from typing import List, Optional
import pytest

from demo.schemas import BoundingBox, DecisionResult, LegalAction, ScreenObservation
from demo.agent.fallback import (
    HeuristicDecisionEngine,
    HeuristicRuleEngine,
    check_single_legal_action,
)


def dummy_bbox() -> BoundingBox:
    return BoundingBox(xmin=100, ymin=200, xmax=300, ymax=400)


def make_card_action(
    choice_id: int,
    card_id: int,
    name: str,
    hp_loss: int = 0,
    upgraded: bool = False,
    limit_count: Optional[int] = None,
    dsl_summary: str = "",
) -> LegalAction:
    desc = f"打出卡牌: {name}" + (f" (需扣血{hp_loss})" if hp_loss > 0 else " (无实际扣血)")
    return LegalAction(
        choice_id=choice_id,
        action_type="play_card",
        description=desc,
        bbox=dummy_bbox(),
        payload={
            "card_id": card_id,
            "name": name,
            "hp_loss": hp_loss,
            "upgraded": upgraded,
            "limit_count": limit_count,
            "dsl_summary": dsl_summary,
        },
    )


def make_drink_action(choice_id: int, drink_id: int, name: str, dsl_summary: str = "") -> LegalAction:
    desc = f"使用饮料: {name}" + (f" ({dsl_summary})" if dsl_summary else "")
    return LegalAction(
        choice_id=choice_id,
        action_type="use_drink",
        description=desc,
        bbox=dummy_bbox(),
        payload={"drink_id": drink_id, "name": name, "dsl_summary": dsl_summary},
    )


def make_end_turn_action(choice_id: int) -> LegalAction:
    return LegalAction(
        choice_id=choice_id,
        action_type="end_turn",
        description="结束回合 (End Turn)",
        bbox=dummy_bbox(),
        payload={},
    )


def make_option_action(choice_id: int, title: str, description: str = "", **kwargs) -> LegalAction:
    payload = {"title": title, "description": description}
    payload.update(kwargs)
    text = f"选择: {title}" + (f" ({description})" if description else "")
    return LegalAction(
        choice_id=choice_id,
        action_type="pick_option",
        description=text,
        bbox=dummy_bbox(),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# 1. Zero-Token Single Legal Action Skip Tests
# ---------------------------------------------------------------------------

def test_single_legal_action_skip_battle_end_turn():
    """Verify single legal action auto-skips with 0 tokens and source='rule_skip'."""
    actions = [make_end_turn_action(choice_id=0)]
    res = check_single_legal_action(actions)
    assert res is not None
    assert res.choice_id == 0
    assert res.source == "rule_skip"
    assert res.reason == "Single legal action auto-skip"
    assert res.prompt_tokens == 0
    assert res.completion_tokens == 0
    assert res.latency_ms == 0.0


def test_single_legal_action_skip_popup_confirm():
    """Verify single confirm button in popup auto-skips."""
    actions = [make_option_action(choice_id=3, title="确认 (OK)")]
    res = check_single_legal_action(actions)
    assert res is not None
    assert res.choice_id == 3
    assert res.source == "rule_skip"
    assert res.prompt_tokens == 0


def test_multiple_legal_actions_does_not_skip():
    """Verify multiple actions return None from single-skip checker."""
    actions = [make_card_action(0, 1, "Card A"), make_end_turn_action(1)]
    assert check_single_legal_action(actions) is None


def test_empty_legal_actions():
    """Verify empty action list does not crash."""
    assert check_single_legal_action([]) is None
    engine = HeuristicDecisionEngine()
    res = engine.decide([])
    assert res.source == "fallback"
    assert res.choice_id == 0


def test_heuristic_engine_delegates_to_single_skip():
    """Verify HeuristicDecisionEngine.decide returns rule_skip when exactly 1 action exists."""
    engine = HeuristicDecisionEngine()
    actions = [make_end_turn_action(choice_id=0)]
    res = engine.decide(actions)
    assert res.source == "rule_skip"
    assert res.prompt_tokens == 0
    assert res.choice_id == 0


# ---------------------------------------------------------------------------
# 2. Battle Turn Heuristics Tests
# ---------------------------------------------------------------------------

def test_battle_drink_buff_priority_over_cards():
    """Playable drink with buff is prioritized over cards and end turn."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="battle_turn", plan="sense", stamina=25, turns_remaining=5)
    actions = [
        make_card_action(0, 101, "存在感", hp_loss=2, dsl_summary="集中+=2"),
        make_drink_action(1, 1, "乌龙茶", dsl_summary="集中+2"),
        make_end_turn_action(2),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1
    assert res.source == "fallback"
    assert "Drink buff prioritized" in res.reason


def test_battle_sense_archetype_prioritizes_good_condition_and_concentration():
    """Sense idol prioritizes Good Condition (好调) / Concentration (集中) in early/mid turns."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="battle_turn", plan="sense", stamina=25, turns_remaining=5)
    actions = [
        make_card_action(0, 999, "常规卡牌", hp_loss=1, dsl_summary="得分+5"),
        make_card_action(1, 102, "好调累积卡", hp_loss=1, dsl_summary="好调+=2回合"),
        make_end_turn_action(2),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1
    assert res.source == "fallback"
    assert "好调累积卡" in res.reason


def test_battle_sense_late_turn_burst():
    """Sense idol prioritizes score burst cards in late turns (turns_remaining <= 2)."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="battle_turn", plan="sense", stamina=25, turns_remaining=2)
    actions = [
        make_card_action(0, 102, "好调累积卡", hp_loss=1, dsl_summary="好调+=2回合"),
        make_card_action(1, 999, "强力爆发卡", hp_loss=1, dsl_summary="得分+50"),
        make_end_turn_action(2),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1
    assert res.source == "fallback"
    assert "强力爆发卡" in res.reason


def test_battle_avoids_suicidal_card_cost():
    """Cards whose hp_loss >= current stamina must be avoided."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="battle_turn", plan="sense", stamina=5, genki=0, turns_remaining=5)
    actions = [
        make_card_action(0, 201, "高伤自残卡", hp_loss=6, dsl_summary="得分+100"),
        make_card_action(1, 202, "安全低耗卡", hp_loss=1, dsl_summary="得分+10"),
        make_end_turn_action(2),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1  # Safe card chosen, choice 0 avoided
    assert res.source == "fallback"


def test_battle_suicidal_all_cards_chooses_end_turn():
    """If all available cards have hp_loss >= stamina, engine must choose End Turn."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="battle_turn", plan="sense", stamina=3, genki=0, turns_remaining=5)
    actions = [
        make_card_action(0, 201, "自残大招", hp_loss=5, dsl_summary="得分+200"),
        make_end_turn_action(1),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1
    assert res.source == "fallback"
    assert "End turn" in res.reason


def test_battle_logic_archetype_prioritizes_good_impression_and_motivation():
    """Logic idol prioritizes Good Impression (好印象) and Motivation (干劲/やる気)."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="battle_turn", plan="logic", stamina=25, turns_remaining=6)
    actions = [
        make_card_action(0, 101, "感性卡牌", hp_loss=1, dsl_summary="好调+=2回合"),
        make_card_action(1, 301, "好印象积累", hp_loss=1, dsl_summary="好印象+=3回合"),
        make_end_turn_action(2),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1
    assert res.source == "fallback"
    assert "好印象积累" in res.reason


def test_battle_anomaly_archetype_prioritizes_strength_and_charge():
    """Anomaly idol prioritizes Strength stance (强气) and Charge (全力充能)."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="battle_turn", plan="anomaly", stamina=25, turns_remaining=6)
    actions = [
        make_card_action(0, 101, "普通攻击", hp_loss=1, dsl_summary="得分+5"),
        make_card_action(1, 401, "强气激昂", hp_loss=1, dsl_summary="进入强气姿态, 全力充能+=2"),
        make_end_turn_action(2),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1
    assert res.source == "fallback"
    assert "强气激昂" in res.reason


def test_battle_anomaly_full_power_burst():
    """Anomaly idol in Full Power stance triggers Full Power burst cards."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(
        scene_type="battle_turn",
        plan="anomaly",
        stamina=25,
        turns_remaining=3,
        buffs={"fullPowerCharge": 10, "isFullPower": 1},
    )
    actions = [
        make_card_action(0, 401, "强气过渡", hp_loss=1, dsl_summary="进入强气姿态"),
        make_card_action(1, 402, "全力绝杀", hp_loss=1, dsl_summary="全力状态得分+120"),
        make_end_turn_action(2),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1
    assert res.source == "fallback"
    assert "全力绝杀" in res.reason


# ---------------------------------------------------------------------------
# 3. Produce Schedule Heuristics Tests
# ---------------------------------------------------------------------------

def test_produce_schedule_low_stamina_prioritizes_rest_or_outing():
    """If stamina < 15, prioritize Rest/Outing."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="produce_schedule", stamina=12, max_stamina=30)
    actions = [
        make_option_action(0, "Vo SP训练 (SP Lesson)"),
        make_option_action(1, "休息 (Rest)"),
        make_option_action(2, "咨询 (Consultation)"),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1
    assert "Rest/Outing" in res.reason


def test_produce_schedule_high_stamina_prefers_sp_lesson():
    """If stamina >= 15, prefer SP Lesson > Regular Lesson > Consultation."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="produce_schedule", stamina=25, max_stamina=30)
    actions = [
        make_option_action(0, "基础训练 (Regular Lesson)"),
        make_option_action(1, "咨询 (Consultation)"),
        make_option_action(2, "Vo SP训练 (SP Lesson)"),
        make_option_action(3, "休息 (Rest)"),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 2  # SP Lesson


def test_produce_schedule_high_stamina_regular_lesson_when_no_sp():
    """If stamina >= 15 and no SP lesson, prefer Regular Lesson."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="produce_schedule", stamina=20, max_stamina=30)
    actions = [
        make_option_action(0, "基础训练 (Regular Lesson)"),
        make_option_action(1, "咨询 (Consultation)"),
        make_option_action(2, "休息 (Rest)"),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 0


def test_produce_schedule_high_stamina_consultation_when_no_lessons():
    """If stamina >= 15 and no lessons available, prefer Consultation over Rest."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="produce_schedule", stamina=20, max_stamina=30)
    actions = [
        make_option_action(0, "休息 (Rest)"),
        make_option_action(1, "咨询 (Consultation)"),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1


# ---------------------------------------------------------------------------
# 4. Card Reward, Popup, and Shop Tests
# ---------------------------------------------------------------------------

def test_card_reward_prioritizes_matching_plan_and_upgraded():
    """Card reward picks card matching plan; prefers upgraded (+) or higher tier."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="card_reward", plan="sense")
    actions = [
        make_option_action(0, "理性卡牌+", plan="logic", upgraded=True, rarity="SSR"),
        make_option_action(1, "感性基础卡", plan="sense", upgraded=False, rarity="R"),
        make_option_action(2, "感性强化卡+", plan="sense", upgraded=True, rarity="SR"),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 2  # Sense plan + upgraded


def test_popup_dismissal_prefers_confirm_or_ok():
    """Popup dismisses by selecting confirm/OK button."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="popup")
    actions = [
        make_option_action(0, "取消 (Cancel)"),
        make_option_action(1, "确认 (OK)"),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1


def test_popup_dismissal_prefers_close_over_cancel():
    """Popup prefers Close over Cancel when Confirm is absent."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="popup")
    actions = [
        make_option_action(0, "キャンセル (Cancel)"),
        make_option_action(1, "閉じる (Close)"),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1


def test_shop_prioritizes_upgrade():
    """Shop heuristics prioritize card upgrades."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="shop")
    actions = [
        make_option_action(0, "退出商店 (Exit)"),
        make_option_action(1, "卡牌强化 (Upgrade Card)"),
        make_option_action(2, "购买饮料 (Buy Drink)"),
    ]
    res = engine.decide(actions, obs)
    assert res.choice_id == 1


def test_heuristic_rule_engine_alias():
    """Verify HeuristicRuleEngine alias points to HeuristicDecisionEngine."""
    assert HeuristicRuleEngine is HeuristicDecisionEngine


# ---------------------------------------------------------------------------
# 5. Robustness against None Attributes Regression Tests
# ---------------------------------------------------------------------------

def test_fallback_robustness_none_attributes_battle():
    """Verify engine handles obs with None attributes (buffs, plan, stamina, turns_remaining, cards) without crashing."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="battle_turn")
    obs.buffs = None
    obs.plan = None
    obs.stamina = None
    obs.turns_remaining = None
    obs.cards = None

    actions = [
        make_card_action(0, 101, "测试卡A", hp_loss=1, dsl_summary="集中+=2"),
        make_card_action(1, 102, "测试卡B", hp_loss=2, dsl_summary="好调+=1"),
        make_drink_action(2, 1, "测试饮料", dsl_summary="好调+1"),
        make_end_turn_action(3),
    ]

    res = engine.decide(actions, obs)
    assert res is not None
    assert res.choice_id in {0, 1, 2, 3}
    assert res.source == "fallback"


def test_fallback_robustness_none_attributes_anomaly_battle():
    """Verify engine handles anomaly plan with None buffs safely."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="battle_turn", plan="anomaly")
    obs.buffs = None
    obs.stamina = None

    actions = [
        make_card_action(0, 401, "强气激昂", hp_loss=1, dsl_summary="进入强气姿态"),
        make_card_action(1, 402, "全力绝杀", hp_loss=2, dsl_summary="全力状态得分+100"),
        make_end_turn_action(2),
    ]

    res = engine.decide(actions, obs)
    assert res is not None
    assert res.choice_id in {0, 1, 2}
    assert res.source == "fallback"


def test_fallback_robustness_none_attributes_produce_schedule():
    """Verify engine handles produce schedule with None stamina without crashing."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="produce_schedule")
    obs.stamina = None

    actions = [
        make_option_action(0, "Vo SP训练 (SP Lesson)"),
        make_option_action(1, "休息 (Rest)"),
    ]

    res = engine.decide(actions, obs)
    assert res is not None
    assert res.choice_id in {0, 1}
    assert res.source == "fallback"


def test_fallback_robustness_none_attributes_card_reward():
    """Verify engine handles card reward with None plan without crashing."""
    engine = HeuristicDecisionEngine()
    obs = ScreenObservation(scene_type="card_reward")
    obs.plan = None

    actions = [
        make_option_action(0, "卡牌A", plan="sense", upgraded=True),
        make_option_action(1, "卡牌B", plan="logic", upgraded=False),
    ]

    res = engine.decide(actions, obs)
    assert res is not None
    assert res.choice_id in {0, 1}
    assert res.source == "fallback"


def test_fallback_robustness_none_obs_battle():
    """Verify engine handles obs=None in battle scene."""
    engine = HeuristicDecisionEngine()
    actions = [
        make_card_action(0, 1, "普通卡", hp_loss=1),
        make_end_turn_action(1),
    ]
    res = engine.decide(actions, obs=None)
    assert res is not None
    assert res.choice_id in {0, 1}

