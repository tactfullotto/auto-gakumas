"""
Unit tests for demo/schemas.py data models.
"""

import pytest
from demo.schemas import (
    BoundingBox,
    RecognizedCard,
    RecognizedDrink,
    ScreenObservation,
    LegalAction,
    DecisionResult,
)


def test_bounding_box_center_and_dimensions():
    bbox = BoundingBox(xmin=100, ymin=200, xmax=300, ymax=400)
    assert bbox.center() == (200, 300)
    assert bbox.width == 200
    assert bbox.height == 200
    assert bbox.to_tuple() == (100, 200, 300, 400)


def test_bounding_box_from_xywh():
    bbox = BoundingBox.from_xywh(50, 60, 100, 120)
    assert bbox.xmin == 50
    assert bbox.ymin == 60
    assert bbox.xmax == 150
    assert bbox.ymax == 180
    assert bbox.center() == (100, 120)


def test_recognized_card_validation():
    bbox = BoundingBox(xmin=10, ymin=20, xmax=30, ymax=40)
    card = RecognizedCard(
        card_id=5,
        name="挑戦",
        upgraded=False,
        playable=True,
        bbox=bbox,
        hp_loss=2,
        dsl_summary="得分+25",
    )
    assert card.card_id == 5
    assert card.name == "挑戦"
    assert not card.upgraded
    assert card.playable
    assert card.hp_loss == 2
    assert card.dsl_summary == "得分+25"


def test_recognized_drink_validation():
    bbox = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)
    drink = RecognizedDrink(
        drink_id=1,
        name="初星水",
        playable=True,
        bbox=bbox,
        dsl_summary="得分+10",
    )
    assert drink.drink_id == 1
    assert drink.name == "初星水"
    assert drink.playable
    assert drink.dsl_summary == "得分+10"


def test_screen_observation_round_trip():
    bbox = BoundingBox(xmin=100, ymin=500, xmax=200, ymax=700)
    obs = ScreenObservation(
        scene_type="battle_turn",
        turn=3,
        turns_remaining=7,
        stamina=22,
        max_stamina=30,
        genki=8,
        score=450,
        plan="sense",
        buffs={"goodConditionTurns": 3, "concentration": 2},
        cards=[
            RecognizedCard(
                card_id=5,
                name="挑戦",
                upgraded=False,
                playable=True,
                bbox=bbox,
            )
        ],
        drinks=[
            RecognizedDrink(
                drink_id=2,
                name="烏龍茶",
                playable=True,
                bbox=bbox,
            )
        ],
        options=[],
    )

    json_str = obs.model_dump_json()
    restored = ScreenObservation.model_validate_json(json_str)

    assert restored.scene_type == "battle_turn"
    assert restored.turn == 3
    assert restored.turns_remaining == 7
    assert restored.stamina == 22
    assert restored.genki == 8
    assert restored.buffs["goodConditionTurns"] == 3
    assert len(restored.cards) == 1
    assert restored.cards[0].name == "挑戦"
    assert len(restored.drinks) == 1
    assert restored.drinks[0].name == "烏龍茶"


def test_legal_action_creation():
    bbox = BoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)
    act = LegalAction(
        choice_id=0,
        action_type="play_card",
        description="打出卡牌: 挑戦",
        bbox=bbox,
        payload={"card_id": 5},
    )
    assert act.choice_id == 0
    assert act.action_type == "play_card"
    assert act.payload["card_id"] == 5


def test_decision_result_creation():
    res = DecisionResult(
        choice_id=1,
        reason="选择高收益卡牌",
        source="llm",
        latency_ms=120.5,
        prompt_tokens=220,
        completion_tokens=25,
    )
    assert res.choice_id == 1
    assert res.reason == "选择高收益卡牌"
    assert res.source == "llm"
    assert res.latency_ms == 120.5
    assert res.prompt_tokens == 220
    assert res.completion_tokens == 25
