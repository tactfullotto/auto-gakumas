"""
Unit tests for demo/agent/prompts.py.
Verifies system prompts, archetype strategies, strict JSON schema output requirements,
dynamic prompt resolution, and token budget compliance (<800 tokens total input).
"""

from demo.core.db import GakumasDB
from demo.core.prompt import PromptBuilder
from demo.core.tracker import GameStateTracker
from demo.schemas import BoundingBox, RecognizedCard, RecognizedDrink, ScreenObservation
from demo.agent.prompts import (
    ANOMALY_SYSTEM_PROMPT,
    CARD_REWARD_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DETAILED_SYSTEM_PROMPT,
    LOGIC_SYSTEM_PROMPT,
    PRODUCE_SCHEDULE_PROMPT,
    SENSE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    estimate_tokens,
    get_system_prompt,
)


def test_system_prompt_constants():
    """Verify SYSTEM_PROMPT contains all critical archetype keywords and constraints."""
    assert SYSTEM_PROMPT is not None
    required_keywords = [
        "choice_id",
        "reason",
        "好调",
        "集中",
        "好印象",
        "干劲",
        "强气",
        "全力",
        "SP训练",
        "limit_count=1",
    ]
    for kw in required_keywords:
        assert kw in SYSTEM_PROMPT, f"Missing expected keyword '{kw}' in SYSTEM_PROMPT"

    assert DEFAULT_SYSTEM_PROMPT is not None
    assert SENSE_SYSTEM_PROMPT is not None
    assert LOGIC_SYSTEM_PROMPT is not None
    assert ANOMALY_SYSTEM_PROMPT is not None
    assert PRODUCE_SCHEDULE_PROMPT is not None
    assert CARD_REWARD_PROMPT is not None
    assert DETAILED_SYSTEM_PROMPT is not None


def test_json_schema_constraints():
    """Verify JSON output instructions are strict and forbid extraneous markdown."""
    assert '{"choice_id": <int>, "reason": "<str>"}' in SYSTEM_PROMPT
    assert "严禁越界" in SYSTEM_PROMPT or "严禁" in SYSTEM_PROMPT


def test_plan_specific_prompt_generation():
    """Verify get_system_prompt dynamically isolates archetype rules in compact mode."""
    sense = get_system_prompt(plan="sense", scene_type="battle_turn", compact=True)
    assert "Sense" in sense or "感性" in sense
    assert "好调" in sense
    assert "集中" in sense
    assert "Logic" not in sense
    assert "Anomaly" not in sense

    logic = get_system_prompt(plan="logic", scene_type="battle_turn", compact=True)
    assert "Logic" in logic or "理性" in logic
    assert "好印象" in logic
    assert "干劲" in logic
    assert "Sense" not in logic
    assert "Anomaly" not in logic

    anomaly = get_system_prompt(plan="anomaly", scene_type="battle_turn", compact=True)
    assert "Anomaly" in anomaly or "异常" in anomaly
    assert "强气" in anomaly
    assert "全力" in anomaly
    assert "Sense" not in anomaly
    assert "Logic" not in anomaly


def test_scene_specific_prompt_generation():
    """Verify get_system_prompt isolates produce schedule and card reward scenes."""
    schedule = get_system_prompt(scene_type="produce_schedule", compact=True)
    assert "培育选课" in schedule or "SP训练" in schedule

    card_reward = get_system_prompt(scene_type="card_reward", compact=True)
    assert "加卡三选一" in card_reward or "卡牌" in card_reward


def test_token_budget_compliance_typical_battle():
    """
    Verify that combining SYSTEM_PROMPT with PromptBuilder's compact battle turn prompt
    strictly complies with the <800 token budget, while default mode builds the full uncapped prompt.
    """
    builder = PromptBuilder(compact=True)
    bbox = BoundingBox(xmin=100, ymin=200, xmax=300, ymax=400)
    cards = [
        RecognizedCard(card_id=1, name="普通攻击", hp_loss=1, bbox=bbox, dsl_summary="得分+5"),
        RecognizedCard(card_id=2, name="集中姿态", hp_loss=2, bbox=bbox, dsl_summary="集中+=2"),
        RecognizedCard(card_id=3, name="全力一击", hp_loss=3, bbox=bbox, dsl_summary="得分+20"),
    ]
    drinks = [
        RecognizedDrink(drink_id=1, name="乌龙茶", bbox=bbox, dsl_summary="集中+2"),
    ]
    obs = ScreenObservation(
        scene_type="battle_turn",
        plan="sense",
        turn=1,
        turns_remaining=8,
        stamina=28,
        max_stamina=30,
        genki=5,
        score=150,
        target_score=500,
        cards=cards,
        drinks=drinks,
        buffs={"goodConditionTurns": 2, "concentration": 3},
    )

    user_prompt, legal_actions = builder.build_prompt(obs)
    assert len(legal_actions) > 0

    # Test universal compact SYSTEM_PROMPT
    combined_universal = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    universal_tokens = estimate_tokens(combined_universal)
    approx_universal = len(combined_universal) // 2

    assert universal_tokens < 800, f"Universal prompt tokens {universal_tokens} exceeds 800 budget!"
    assert approx_universal < 800, f"Universal prompt approx {approx_universal} exceeds 800 budget!"

    # Test dynamic plan-specific system prompt
    plan_prompt = get_system_prompt(plan="sense", scene_type="battle_turn", compact=True)
    combined_plan = f"{plan_prompt}\n\n{user_prompt}"
    plan_tokens = estimate_tokens(combined_plan)
    approx_plan = len(combined_plan) // 2

    assert plan_tokens < 700, f"Plan-specific prompt tokens {plan_tokens} exceeds 700 budget!"
    assert approx_plan < 700, f"Plan-specific approx {approx_plan} exceeds 700 budget!"


def test_token_budget_compliance_produce_schedule():
    """Verify that schedule prompt + schedule observation fits comfortably under budget in compact mode."""
    builder = PromptBuilder(compact=True)
    obs = ScreenObservation(
        scene_type="produce_schedule",
        stamina=22,
        max_stamina=30,
        options=[
            {"title": "Vo SP训练 (SP Lesson)", "description": "主属性高收益训练"},
            {"title": "通常训练 (Regular Lesson)", "description": "常规训练"},
            {"title": "休息 (Rest)", "description": "回复体力"},
        ],
    )
    user_prompt, legal_actions = builder.build_prompt(obs)
    assert len(legal_actions) == 3

    sys_prompt = get_system_prompt(scene_type="produce_schedule", compact=True)
    combined = f"{sys_prompt}\n\n{user_prompt}"
    tokens = estimate_tokens(combined)

    assert tokens < 600, f"Schedule prompt tokens {tokens} exceeds 600 budget!"


def test_token_budget_compliance_extreme_battle_under_800_tokens_o200k():
    """
    Regression test: Verifies that combining SYSTEM_PROMPT with PromptBuilder's
    extreme battle turn (5 complex cards, 3 drinks, 10 buffs) strictly complies
    with < 800 tokens under o200k_base.
    """
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")

    with GakumasDB() as db:
        builder = PromptBuilder(db, compact=True)
        tracker = GameStateTracker("sense")
        bbox = BoundingBox(xmin=100, ymin=1400, xmax=250, ymax=1600)

        cards = [
            RecognizedCard(
                card_id=932,
                name="レジェンドスター+",
                bbox=bbox,
                dsl_summary="集中+=5; score+=concentration*2.5; concentration+=3; score+=50",
            ),
            RecognizedCard(card_id=831, name="笑顔を生み出す一番星", bbox=bbox),
            RecognizedCard(card_id=832, name="微熱ノスタルジー♡", bbox=bbox),
            RecognizedCard(
                card_id=930,
                name="レジェンドスター",
                bbox=bbox,
                dsl_summary="集中+=4; score+=concentration*2.0; concentration+=2; score+=30",
            ),
            RecognizedCard(card_id=342, name="こくりとひとくち", bbox=bbox),
        ]

        drinks = [
            RecognizedDrink(drink_id=1, name="初星水", bbox=bbox),
            RecognizedDrink(drink_id=2, name="烏龍茶", bbox=bbox),
            RecognizedDrink(drink_id=3, name="ルイボスティー", bbox=bbox),
        ]

        buffs = {
            "goodConditionTurns": 4,
            "perfectConditionTurns": 2,
            "concentration": 10,
            "motivation": 6,
            "goodImpressionTurns": 5,
            "fullPowerCharge": 3,
            "nullifyCostCards": 1,
            "halfCostTurns": 2,
            "doubleCostTurns": 0,
            "cardUsesRemaining": 2,
        }

        obs = ScreenObservation(
            scene_type="battle_turn",
            turn=5,
            turns_remaining=5,
            stamina=22,
            max_stamina=30,
            genki=12,
            score=1800,
            target_score=5000,
            plan="sense",
            buffs=buffs,
            cards=cards,
            drinks=drinks,
        )

        user_prompt, actions = builder.build_prompt(obs, tracker)

    # 1. Combined with universal compact SYSTEM_PROMPT
    combined_universal = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    universal_tokens = len(enc.encode(combined_universal))
    assert universal_tokens < 800, f"Extreme battle + SYSTEM_PROMPT tokens ({universal_tokens}) exceeds 800 budget!"

    # 2. Combined with dynamic plan-specific prompt
    plan_prompt = get_system_prompt("sense", compact=True)
    combined_plan = f"{plan_prompt}\n\n{user_prompt}"
    plan_tokens = len(enc.encode(combined_plan))
    assert plan_tokens < 700, f"Extreme battle + plan_prompt tokens ({plan_tokens}) exceeds 700 budget!"


def test_dynamic_plan_specific_prompts_token_budget():
    """Verify dynamic plan-specific prompts in compact mode stay <= 80 tokens under o200k_base."""
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")

    for plan in ("sense", "logic", "anomaly"):
        prompt = get_system_prompt(plan=plan, scene_type="battle_turn", compact=True)
        tokens = len(enc.encode(prompt))
        assert tokens <= 80, f"Dynamic prompt for {plan} has {tokens} tokens, exceeding 80 token target!"

    for scene in ("produce_schedule", "card_reward"):
        prompt = get_system_prompt(scene_type=scene, compact=True)
        tokens = len(enc.encode(prompt))
        assert tokens <= 80, f"Dynamic prompt for {scene} has {tokens} tokens, exceeding 80 token target!"

