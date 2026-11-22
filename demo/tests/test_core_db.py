"""
Unit tests for demo/core/db.py read-only SQLite database layer.
"""

import sqlite3
import pytest
from demo.core.db import GakumasDB


def test_db_read_only_enforcement():
    db = GakumasDB()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        db.connection.execute("CREATE TABLE test_table (id INT)")
    db.close()


def test_get_cards_for_plan_counts_and_exclusion_of_hidden():
    with GakumasDB() as db:
        sense_cards = db.get_cards_for_plan("sense")
        logic_cards = db.get_cards_for_plan("logic")
        anomaly_cards = db.get_cards_for_plan("anomaly")

        assert len(sense_cards) == 358
        assert len(logic_cards) == 380
        assert len(anomaly_cards) == 316

        # Confirm hidden cards (IDs 923, 925) are excluded
        sense_ids = {c["id"] for c in sense_cards}
        assert 923 not in sense_ids
        assert 925 not in sense_ids

        # Spot check card structure
        first_card = sense_cards[0]
        assert "id" in first_card
        assert "name" in first_card
        assert "dsl_cost" in first_card
        assert "dsl_conditions" in first_card
        assert "dsl_actions" in first_card
        assert "icon_path" in first_card


def test_get_drinks_for_plan():
    with GakumasDB() as db:
        for plan in ("sense", "logic", "anomaly"):
            drinks = db.get_drinks_for_plan(plan)
            assert len(drinks) == 16
            first_drink = drinks[0]
            assert "id" in first_drink
            assert "name" in first_drink
            assert "dsl_actions" in first_drink
            assert "icon_path" in first_drink


def test_get_card_by_id():
    with GakumasDB() as db:
        # Card 5 is '挑戦'
        card = db.get_card_by_id(5)
        assert card is not None
        assert card["name"] == "挑戦"
        assert card["dsl_cost"] == "cost-=7"
        assert card["dsl_conditions"] == "if:goodConditionTurns>=1"
        assert card["dsl_actions"] == "score+=25"

        # Non-existent card ID
        missing = db.get_card_by_id(999999)
        assert missing is None


def test_get_drink_by_id():
    with GakumasDB() as db:
        # Drink 1 is '初星水'
        drink = db.get_drink_by_id(1)
        assert drink is not None
        assert drink["name"] == "初星水"
        assert drink["dsl_actions"] == "score+=10"

        # Non-existent drink ID
        missing = db.get_drink_by_id(999999)
        assert missing is None


def test_get_all_idols_and_p_idols():
    with GakumasDB() as db:
        idols = db.get_all_idols()
        assert len(idols) == 15

        sense_p_idols = db.get_p_idols("sense")
        assert len(sense_p_idols) > 0
        for p in sense_p_idols:
            assert p["plan"].lower() == "sense"
