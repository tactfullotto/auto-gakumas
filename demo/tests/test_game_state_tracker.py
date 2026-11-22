"""
Unit tests for demo/core/tracker.py GameStateTracker.
"""

from demo.core.tracker import GameStateTracker
from demo.schemas import ScreenObservation


def test_tracker_initialization():
    tracker = GameStateTracker(plan="sense")
    assert tracker.plan == "sense"
    assert tracker.turn == 1
    assert tracker.turns_remaining == 10
    assert tracker.card_uses_remaining == 1
    assert len(tracker.used_limit_cards) == 0


def test_tracker_record_card_play_normal():
    tracker = GameStateTracker(plan="logic")
    tracker.record_card_play(card_id=1, limit_count=None)
    assert tracker.card_uses_remaining == 0
    assert not tracker.is_card_exhausted(1)
    assert len(tracker.used_limit_cards) == 0


def test_tracker_record_limit_card_play():
    tracker = GameStateTracker(plan="sense")
    # Card 5 is limit_count=1, base_id=5
    tracker.record_card_play(card_id=5, limit_count=1, base_id=5)
    assert tracker.card_uses_remaining == 0
    assert tracker.is_card_exhausted(5)
    assert 5 in tracker.used_limit_cards

    # Advance turn
    tracker.start_new_turn(turns_remaining=9)
    assert tracker.turn == 2
    assert tracker.turns_remaining == 9
    assert tracker.card_uses_remaining == 1
    # Limit card should remain exhausted in subsequent turns
    assert tracker.is_card_exhausted(5)


def test_tracker_update_observation():
    tracker = GameStateTracker(plan="anomaly")
    obs = ScreenObservation(
        turn=4,
        turns_remaining=6,
        stamina=18,
        max_stamina=30,
        genki=12,
        score=500,
        card_uses_remaining=1,
        buffs={"fullPowerCharge": 4},
        plan="anomaly",
    )
    tracker.update_observation(obs)
    assert tracker.turn == 4
    assert tracker.turns_remaining == 6
    assert tracker.stamina == 18
    assert tracker.genki == 12
    assert tracker.score == 500
    assert tracker.buffs["fullPowerCharge"] == 4


def test_tracker_reset():
    tracker = GameStateTracker(plan="sense")
    tracker.record_card_play(card_id=5, limit_count=1)
    tracker.record_drink_use(1)
    assert len(tracker.used_limit_cards) == 1
    assert len(tracker.drinks_used_history) == 1

    tracker.reset()
    assert len(tracker.used_limit_cards) == 0
    assert len(tracker.drinks_used_history) == 0
    assert tracker.turn == 1
    assert tracker.card_uses_remaining == 1
