"""
Read-only SQLite Knowledge Base Access Layer for Gakumas Demo.
"""

from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional


class GakumasDB:
    """Provides read-only access to gakumas_library.db."""

    def __init__(self, db_path: str = "gakumas_library.db"):
        self.raw_path = db_path
        if db_path.startswith("file:"):
            self.uri = db_path
        else:
            resolved = Path(db_path).resolve()
            # Construct standard SQLite URI with read-only mode flag
            self.uri = resolved.as_uri() + "?mode=ro"

        self._conn: Optional[sqlite3.Connection] = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.uri, uri=True, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "GakumasDB":
        _ = self.connection
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_cards_for_plan(self, plan: str) -> List[Dict[str, Any]]:
        """
        Loads all non-hidden skill cards available for a given plan
        (sense, logic, anomaly).
        """
        query = """
            SELECT 
                c.id, c.base_id, c.upgraded_id, c.name, c.rarity, c.type, 
                c.plan, c.category, c.scope, c.source_type, c.unlock_plv, c.upgraded, 
                c.is_unique, c.is_unique_in_deck, c.is_signature, c.p_idol_id, c.related_p_idol_id, 
                c.generated_by_card_ids_json, c.force_initial_hand, c.limit_count, c.contest_power, 
                c.hidden, c.description, c.dsl_conditions, c.dsl_cost, c.dsl_actions, c.dsl_effects, 
                c.parsed_effects_json, c.available_customization_ids_json, c.raw_json,
                pc.icon_path, pc.details_path, pc.icon_variants_json
            FROM plan_cards pc
            JOIN cards c ON pc.card_id = c.id
            WHERE pc.plan = ? AND c.hidden = 0
            ORDER BY c.id ASC
        """
        cur = self.connection.cursor()
        cur.execute(query, (plan.lower(),))
        return [dict(row) for row in cur.fetchall()]

    def get_drinks_for_plan(self, plan: str) -> List[Dict[str, Any]]:
        """
        Loads all 16 drinks (plan specific + common) available for the plan.
        """
        query = """
            SELECT 
                d.id, d.name, d.rarity, d.plan, d.scope, d.unlock_plv, 
                d.description, d.dsl_actions, d.parsed_effects_json, d.raw_json,
                pd.icon_path, pd.details_path
            FROM plan_drinks pd
            JOIN drinks d ON pd.drink_id = d.id
            WHERE pd.plan = ?
            ORDER BY d.id ASC
        """
        cur = self.connection.cursor()
        cur.execute(query, (plan.lower(),))
        return [dict(row) for row in cur.fetchall()]

    def get_card_by_id(self, card_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetches card metadata, dsl_cost, dsl_conditions, and dsl_actions by card ID.
        """
        query = """
            SELECT 
                c.id, c.base_id, c.upgraded_id, c.name, c.rarity, c.type, 
                c.plan, c.category, c.scope, c.source_type, c.unlock_plv, c.upgraded, 
                c.is_unique, c.is_unique_in_deck, c.is_signature, c.p_idol_id, c.related_p_idol_id, 
                c.generated_by_card_ids_json, c.force_initial_hand, c.limit_count, c.contest_power, 
                c.hidden, c.description, c.dsl_conditions, c.dsl_cost, c.dsl_actions, c.dsl_effects, 
                c.parsed_effects_json, c.available_customization_ids_json, c.raw_json,
                pc.icon_path, pc.details_path, pc.icon_variants_json
            FROM cards c
            LEFT JOIN plan_cards pc ON c.id = pc.card_id
            WHERE c.id = ?
            LIMIT 1
        """
        cur = self.connection.cursor()
        cur.execute(query, (card_id,))
        row = cur.fetchone()
        return dict(row) if row is not None else None

    def get_drink_by_id(self, drink_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetches drink metadata and dsl_actions by drink ID.
        """
        query = """
            SELECT 
                d.id, d.name, d.rarity, d.plan, d.scope, d.unlock_plv, 
                d.description, d.dsl_actions, d.parsed_effects_json, d.raw_json,
                pd.icon_path, pd.details_path
            FROM drinks d
            LEFT JOIN plan_drinks pd ON d.id = pd.drink_id
            WHERE d.id = ?
            LIMIT 1
        """
        cur = self.connection.cursor()
        cur.execute(query, (drink_id,))
        row = cur.fetchone()
        return dict(row) if row is not None else None

    def get_all_idols(self) -> List[Dict[str, Any]]:
        """Returns all 15 idols."""
        cur = self.connection.cursor()
        cur.execute("SELECT id, name FROM idols ORDER BY id ASC")
        return [dict(row) for row in cur.fetchall()]

    def get_p_idols(self, plan: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns P-Idols filtered by plan if specified."""
        cur = self.connection.cursor()
        if plan:
            cur.execute(
                "SELECT * FROM p_idols WHERE plan = ? ORDER BY id ASC", (plan.lower(),)
            )
        else:
            cur.execute("SELECT * FROM p_idols ORDER BY id ASC")
        return [dict(row) for row in cur.fetchall()]

    def get_p_idol_by_id(self, p_idol_id: int) -> Optional[Dict[str, Any]]:
        """Fetches a single P-Idol record including `recommended_effect` by ID."""
        cur = self.connection.cursor()
        cur.execute("SELECT * FROM p_idols WHERE id = ? LIMIT 1", (int(p_idol_id),))
        row = cur.fetchone()
        return dict(row) if row is not None else None

    def get_all_customizations(self) -> List[Dict[str, Any]]:
        """Returns all skill card customization definitions (`customizations` table)."""
        cur = self.connection.cursor()
        cur.execute(
            """
            SELECT
                id, alias, name, type, prices_json, max_level,
                force_initial_hand, limit_count, dsl_conditions,
                dsl_cost, dsl_actions, dsl_effects, parsed_patches_json
            FROM customizations
            ORDER BY id ASC
            """
        )
        return [dict(row) for row in cur.fetchall()]

    def get_customization_by_id(self, cust_id: int) -> Optional[Dict[str, Any]]:
        """Fetches a single skill card customization definition by ID."""
        cur = self.connection.cursor()
        cur.execute(
            """
            SELECT
                id, alias, name, type, prices_json, max_level,
                force_initial_hand, limit_count, dsl_conditions,
                dsl_cost, dsl_actions, dsl_effects, parsed_patches_json
            FROM customizations
            WHERE id = ?
            LIMIT 1
            """,
            (int(cust_id),),
        )
        row = cur.fetchone()
        return dict(row) if row is not None else None
