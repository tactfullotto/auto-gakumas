"""
Comprehensive Test Suite for Gakumas Vision Recognition and Mock Frame Generator (Milestone 2).
Covers:
- Frame generation validity (1080x1920, 3 channels, non-empty, multiple scenes)
- ROI coordinate extraction, normalization, and arbitrary resolution scaling
- Card recognition accuracy across Sense, Logic, and Anomaly plans
- Drink recognition accuracy on filled and empty slots
- Playable vs dimmed card detection
- Upgraded '+' card detection
- Offline end-to-end round-trip verification (>= 95% accuracy)
"""

import time
import numpy as np
import pytest

from demo.core.db import GakumasDB
from demo.schemas import BoundingBox, ScreenObservation
from demo.vision.roi import (
    BASE_WIDTH,
    BASE_HEIGHT,
    scale_bbox,
    to_normalized,
    from_normalized,
    crop_roi,
    get_hand_bboxes,
    get_drink_bboxes,
    get_status_bboxes,
    extract_hand_rois,
    extract_drink_rois,
    extract_status_rois,
    StatusOCRExtractor,
)
from demo.vision.recognizer import (
    CardRecognizer,
    DrinkRecognizer,
    recognize_screen,
    clear_recognizer_cache,
)
from demo.vision.mock_generator import MockFrameGenerator


@pytest.fixture(scope="module")
def shared_db():
    """Provides read-only database instance."""
    db = GakumasDB()
    yield db
    db.close()


def test_frame_generation_validity(shared_db):
    """Verify that MockFrameGenerator produces valid 1080x1920 frames for multiple scenes."""
    generator = MockFrameGenerator(db=shared_db, plan="sense")

    scenes = ["battle_turn", "produce_schedule", "card_reward", "shop", "popup"]
    for scene in scenes:
        frame, obs = generator.generate_frame(scene_type=scene)
        assert isinstance(frame, np.ndarray), f"Scene {scene} must return np.ndarray"
        assert frame.shape == (BASE_HEIGHT, BASE_WIDTH, 3), f"Scene {scene} shape mismatch: {frame.shape}"
        assert frame.dtype == np.uint8, f"Scene {scene} dtype mismatch: {frame.dtype}"
        assert frame.max() > 0, f"Scene {scene} frame cannot be all black"
        assert np.var(frame) > 10.0, f"Scene {scene} frame must have variance"
        assert isinstance(obs, ScreenObservation), f"Scene {scene} must return ScreenObservation"
        assert obs.scene_type == scene


def test_roi_coordinate_extraction_and_scaling():
    """Verify ROI coordinates, normalization [0.0, 1.0], and arbitrary resolution scaling."""
    # Test hand bboxes for different card counts (1 to 5)
    for count in range(1, 6):
        bboxes = get_hand_bboxes(count=count)
        assert len(bboxes) == count
        for bbox in bboxes:
            assert 0 <= bbox.xmin < bbox.xmax <= BASE_WIDTH
            assert 0 <= bbox.ymin < bbox.ymax <= BASE_HEIGHT
            assert bbox.width > 0
            assert bbox.height > 0

    # Test drink bboxes (up to 3)
    drink_boxes = get_drink_bboxes(count=3)
    assert len(drink_boxes) == 3
    for b in drink_boxes:
        assert 0 <= b.xmin < b.xmax <= BASE_WIDTH
        assert 0 <= b.ymin < b.ymax <= BASE_HEIGHT

    # Test status bboxes
    status_boxes = get_status_bboxes()
    for name in ["turn", "stamina", "genki", "score", "buffs", "end_turn"]:
        assert name in status_boxes
        box = status_boxes[name]
        assert 0 <= box.xmin < box.xmax <= BASE_WIDTH
        assert 0 <= box.ymin < box.ymax <= BASE_HEIGHT

    # Test scaling to 720x1280
    test_box = BoundingBox(xmin=108, ymin=192, xmax=540, ymax=960)
    scaled = scale_bbox(test_box, (BASE_WIDTH, BASE_HEIGHT), (720, 1280))
    assert scaled.xmin == 72
    assert scaled.ymin == 128
    assert scaled.xmax == 360
    assert scaled.ymax == 640

    # Test normalization and inverse conversion
    norm = to_normalized(test_box, (BASE_WIDTH, BASE_HEIGHT))
    assert 0.0 <= norm[0] <= 1.0
    assert 0.0 <= norm[1] <= 1.0
    restored = from_normalized(norm, (BASE_WIDTH, BASE_HEIGHT))
    assert restored.xmin == test_box.xmin
    assert restored.ymin == test_box.ymin
    assert restored.xmax == test_box.xmax
    assert restored.ymax == test_box.ymax

    # Test crop_roi safety
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_roi(dummy_frame, BoundingBox(xmin=10, ymin=20, xmax=40, ymax=60))
    assert crop.shape == (40, 30, 3)

    # Out of bounds clamping
    oob_crop = crop_roi(dummy_frame, BoundingBox(xmin=80, ymin=80, xmax=150, ymax=150))
    assert oob_crop.shape == (20, 20, 3)


def test_card_recognition_accuracy_sense_logic_anomaly(shared_db):
    """Verify card recognition accuracy on synthetic frames for Sense, Logic, and Anomaly plans."""
    for plan in ["sense", "logic", "anomaly"]:
        generator = MockFrameGenerator(db=shared_db, plan=plan)
        recognizer = CardRecognizer(db=shared_db, plan=plan)

        # Get cards available for this plan
        plan_cards = [c for c in shared_db.get_cards_for_plan(plan) if c.get("icon_path")]
        assert len(plan_cards) >= 3, f"Need at least 3 cards for {plan}"

        # Select 3 distinct cards
        selected = plan_cards[:3]
        frame, ground_truth = generator.generate_battle_frame(cards=selected, plan=plan)

        recognized = recognizer.recognize_hand(frame, count=len(selected), plan=plan)
        assert len(recognized) == len(selected)

        # Verify each card ID matches ground truth exactly
        for i in range(len(selected)):
            assert recognized[i].card_id == ground_truth.cards[i].card_id, (
                f"Plan {plan} card {i} mismatch: expected {ground_truth.cards[i].card_id} ({ground_truth.cards[i].name}), "
                f"got {recognized[i].card_id} ({recognized[i].name})"
            )


def test_drink_recognition_filled_and_empty_slots(shared_db):
    """Verify drink recognition on filled slots and accurate identification of empty slots."""
    generator = MockFrameGenerator(db=shared_db, plan="sense")
    recognizer = DrinkRecognizer(db=shared_db, plan="sense")

    plan_drinks = [d for d in shared_db.get_drinks_for_plan("sense") if d.get("icon_path")]
    assert len(plan_drinks) >= 2

    # Case 1: 2 filled drinks, 1 empty slot
    test_drinks = [plan_drinks[0], plan_drinks[1]]
    frame, ground_truth = generator.generate_battle_frame(drinks=test_drinks, plan="sense")

    recognized = recognizer.recognize_drinks(frame, count=3, plan="sense")
    assert len(recognized) == 2, f"Expected 2 drinks recognized (1 empty slot ignored), got {len(recognized)}"
    assert recognized[0].drink_id == plan_drinks[0]["id"]
    assert recognized[1].drink_id == plan_drinks[1]["id"]

    # Case 2: All 3 slots empty
    frame_empty, _ = generator.generate_battle_frame(drinks=[], plan="sense")
    recognized_empty = recognizer.recognize_drinks(frame_empty, count=3, plan="sense")
    assert len(recognized_empty) == 0, f"Expected 0 drinks recognized for all empty slots, got {len(recognized_empty)}"


def test_playable_vs_dimmed_card_detection(shared_db):
    """Verify that dimmed/unplayable cards are accurately detected vs bright active cards."""
    generator = MockFrameGenerator(db=shared_db, plan="sense")
    recognizer = CardRecognizer(db=shared_db, plan="sense")

    plan_cards = [c for c in shared_db.get_cards_for_plan("sense") if c.get("icon_path")][:3]

    # Create 3 cards: card 0 playable, card 1 unplayable (dimmed), card 2 playable
    cards_input = [
        {"card_id": plan_cards[0]["id"], "playable": True, "upgraded": False},
        {"card_id": plan_cards[1]["id"], "playable": False, "upgraded": False},
        {"card_id": plan_cards[2]["id"], "playable": True, "upgraded": False},
    ]

    frame, ground_truth = generator.generate_battle_frame(cards=cards_input, plan="sense")
    recognized = recognizer.recognize_hand(frame, count=3, plan="sense")

    assert len(recognized) == 3
    assert recognized[0].playable is True, "Card 0 must be recognized as playable"
    assert recognized[1].playable is False, "Card 1 must be recognized as unplayable (dimmed)"
    assert recognized[2].playable is True, "Card 2 must be recognized as playable"


def test_upgraded_card_detection(shared_db):
    """Verify that upgraded cards (via badge or upgraded template) are accurately detected."""
    generator = MockFrameGenerator(db=shared_db, plan="sense")
    recognizer = CardRecognizer(db=shared_db, plan="sense")

    plan_cards = [c for c in shared_db.get_cards_for_plan("sense") if c.get("icon_path")][:3]

    # Card 0: unupgraded, Card 1: upgraded (+ badge), Card 2: unupgraded
    cards_input = [
        {"card_id": plan_cards[0]["id"], "playable": True, "upgraded": False},
        {"card_id": plan_cards[1]["id"], "playable": True, "upgraded": True},
        {"card_id": plan_cards[2]["id"], "playable": True, "upgraded": False},
    ]

    frame, ground_truth = generator.generate_battle_frame(cards=cards_input, plan="sense")
    recognized = recognizer.recognize_hand(frame, count=3, plan="sense")

    assert len(recognized) == 3
    assert recognized[0].upgraded is False, "Card 0 should not be upgraded"
    assert recognized[1].upgraded is True, "Card 1 must be recognized as upgraded"
    assert recognized[2].upgraded is False, "Card 2 should not be upgraded"


def test_offline_e2e_round_trip(shared_db):
    """
    Offline end-to-end round trip test:
    MockFrameGenerator synthesizes frame -> recognize_screen extracts ScreenObservation ->
    Verify cards, drinks, status indicators match ground truth with >= 95% accuracy.
    """
    test_cases = [
        {"plan": "sense", "turn": 2, "turns_remaining": 8, "stamina": 24, "max_stamina": 30, "genki": 5, "buffs": {"good_condition": 3}},
        {"plan": "logic", "turn": 4, "turns_remaining": 6, "stamina": 18, "max_stamina": 30, "genki": 12, "buffs": {"good_impression": 6}},
        {"plan": "anomaly", "turn": 1, "turns_remaining": 11, "stamina": 30, "max_stamina": 30, "genki": 0, "buffs": {"strong_spirit": 2}},
    ]

    total_checks = 0
    passed_checks = 0

    for tc in test_cases:
        plan = tc["plan"]
        generator = MockFrameGenerator(db=shared_db, plan=plan)

        cards = [c for c in shared_db.get_cards_for_plan(plan) if c.get("icon_path")][:3]
        drinks = [d for d in shared_db.get_drinks_for_plan(plan) if d.get("icon_path")][:2]

        frame, gt = generator.generate_battle_frame(
            cards=cards,
            drinks=drinks,
            stamina=tc["stamina"],
            max_stamina=tc["max_stamina"],
            genki=tc["genki"],
            turn=tc["turn"],
            turns_remaining=tc["turns_remaining"],
            buffs=tc["buffs"],
            plan=plan,
        )

        obs = recognize_screen(frame, plan=plan, card_count=len(cards), drink_count=3, db=shared_db)

        # 1. Cards match
        total_checks += len(cards)
        for i in range(len(cards)):
            if i < len(obs.cards) and obs.cards[i].card_id == gt.cards[i].card_id:
                passed_checks += 1

        # 2. Drinks match
        total_checks += len(drinks)
        for i in range(len(drinks)):
            if i < len(obs.drinks) and obs.drinks[i].drink_id == gt.drinks[i].drink_id:
                passed_checks += 1

        # 3. Status checks
        total_checks += 5
        if obs.turn == gt.turn:
            passed_checks += 1
        if obs.turns_remaining == gt.turns_remaining:
            passed_checks += 1
        if obs.stamina == gt.stamina:
            passed_checks += 1
        if obs.max_stamina == gt.max_stamina:
            passed_checks += 1
        if obs.genki == gt.genki:
            passed_checks += 1

    accuracy = (passed_checks / float(total_checks)) * 100.0
    print(f"\nRound-trip accuracy: {passed_checks}/{total_checks} ({accuracy:.2f}%)")
    assert accuracy >= 95.0, f"Round-trip accuracy {accuracy:.2f}% is below 95% threshold"


@pytest.mark.parametrize("card_id,expected_plan", [
    (550, "anomaly"),
    (551, "anomaly"),
    (723, "sense"),
    (728, "anomaly"),
    (758, "logic"),
])
def test_null_description_cards_synthesis_and_recognition(shared_db, card_id, expected_plan):
    """
    Verify that cards with NULL description in SQLite can be synthesized
    by MockFrameGenerator and recognized by recognize_screen without ValidationError.
    """
    card_data = shared_db.get_card_by_id(card_id)
    assert card_data is not None, f"Card {card_id} not found in database"
    assert card_data.get("description") is None, f"Expected Card {card_id} description to be None"

    generator = MockFrameGenerator(db=shared_db, plan=expected_plan)
    frame, gt = generator.generate_battle_frame(cards=[card_data], plan=expected_plan)

    assert len(gt.cards) == 1
    assert gt.cards[0].card_id == card_id
    assert gt.cards[0].dsl_summary == ""

    obs = recognize_screen(frame, plan=expected_plan, card_count=1, db=shared_db)
    assert len(obs.cards) == 1
    assert obs.cards[0].card_id == card_id
    assert obs.cards[0].dsl_summary == ""


@pytest.mark.parametrize("hand_count", [1, 2, 4, 5])
def test_variable_hand_counts_synthesis_and_recognition(shared_db, hand_count):
    """
    Verify hand size variations (1, 2, 4, 5) can be synthesized with correct layout
    and correctly recognized by recognize_screen.
    """
    generator = MockFrameGenerator(db=shared_db, plan="sense")
    plan_cards = [c for c in shared_db.get_cards_for_plan("sense") if c.get("icon_path")][:hand_count]
    assert len(plan_cards) == hand_count

    frame, gt = generator.generate_battle_frame(cards=plan_cards, plan="sense")
    assert len(gt.cards) == hand_count

    obs = recognize_screen(frame, plan="sense", card_count=hand_count, db=shared_db)
    assert len(obs.cards) == hand_count

    for i in range(hand_count):
        assert obs.cards[i].card_id == gt.cards[i].card_id
        assert obs.cards[i].bbox.xmin == gt.cards[i].bbox.xmin
        assert obs.cards[i].bbox.xmax == gt.cards[i].bbox.xmax


def test_template_caching_performance(shared_db):
    """
    Verify that template caching reduces subsequent recognize_screen call latency
    to strictly under 50ms.
    """
    generator = MockFrameGenerator(db=shared_db, plan="sense")
    frame, _ = generator.generate_battle_frame(plan="sense")

    # Clear cache first to test clean slate
    clear_recognizer_cache()

    # Call 1: cold call (loads templates into cache)
    t0 = time.perf_counter()
    obs1 = recognize_screen(frame, plan="sense", db=shared_db)
    t1 = time.perf_counter()
    cold_ms = (t1 - t0) * 1000.0

    # Call 2 to 6: warm cached calls
    warm_latencies = []
    for _ in range(5):
        t_start = time.perf_counter()
        obs2 = recognize_screen(frame, plan="sense", db=shared_db)
        t_end = time.perf_counter()
        warm_latencies.append((t_end - t_start) * 1000.0)

    avg_warm_ms = sum(warm_latencies) / len(warm_latencies)
    print(f"\nCold call: {cold_ms:.1f}ms, Warm calls: {warm_latencies}, Avg warm: {avg_warm_ms:.1f}ms")

    # Verify every subsequent call is well under 50ms
    for i, lat in enumerate(warm_latencies):
        assert lat < 50.0, f"Call {i + 2} latency {lat:.1f}ms exceeds 50ms limit"

    # Verify cached output is valid and identical
    assert len(obs2.cards) == len(obs1.cards)
    for c1, c2 in zip(obs1.cards, obs2.cards):
        assert c1.card_id == c2.card_id


def test_recognize_screen_with_explicit_recognizers(shared_db):
    """
    Verify recognize_screen works properly when pre-instantiated recognizers
    are passed explicitly.
    """
    card_rec = CardRecognizer(db=shared_db, plan="logic")
    drink_rec = DrinkRecognizer(db=shared_db, plan="logic")

    generator = MockFrameGenerator(db=shared_db, plan="logic")
    frame, gt = generator.generate_battle_frame(plan="logic")

    t0 = time.perf_counter()
    obs = recognize_screen(
        frame,
        plan="logic",
        card_count=len(gt.cards),
        drink_count=len(gt.drinks),
        db=shared_db,
        card_recognizer=card_rec,
        drink_recognizer=drink_rec,
    )
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    assert elapsed_ms < 50.0, f"Explicit recognizer call took {elapsed_ms:.1f}ms, expected <50ms"
    assert len(obs.cards) == len(gt.cards)
