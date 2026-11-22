"""
Core Data Schemas for Gakumas Autonomous Decision and Automation System.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized or absolute bounding box coordinate container."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def center(self) -> Tuple[int, int]:
        """Compute integer center (x, y) coordinates."""
        return ((self.xmin + self.xmax) // 2, (self.ymin + self.ymax) // 2)

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> "BoundingBox":
        return cls(xmin=x, ymin=y, xmax=x + w, ymax=y + h)


class RecognizedCard(BaseModel):
    """Recognized skill card in player's hand."""

    card_id: int
    name: str
    upgraded: bool = False
    playable: bool = True
    selected: bool = False
    bbox: BoundingBox
    hp_loss: int = 0
    cost_type: str = "green"  # 'green' (盾先抵扣) | 'red' (直接扣血穿盾) | 'buff' | 'free'
    raw_cost: int = 0
    effective_cost: int = 0
    shield_cost: int = 0
    condition_desc: str = ""
    visual_playable: Optional[bool] = None
    dsl_summary: str = ""
    reason: str = ""
    type: str = "active"  # 'active' | 'mental' | 'trouble'
    rarity: str = "N"


class RecognizedDrink(BaseModel):
    """Recognized P-Drink in drink slots."""

    drink_id: int
    name: str
    playable: bool = True
    bbox: BoundingBox
    dsl_summary: str = ""
    reason: str = ""


class ScreenObservation(BaseModel):
    """Complete perception state of the current game screen."""

    scene_type: str = "battle_turn"  # 'battle_turn' | 'produce_schedule' | 'card_reward' | 'shop' | 'popup'
    turn: int = 1
    turns_remaining: int = 10
    stamina: int = 30
    max_stamina: int = 30
    genki: int = 0
    score: int = 0
    target_score: int = 0
    p_points: int = 0
    card_uses_remaining: int = 1
    plan: str = "sense"  # 'sense' | 'logic' | 'anomaly'
    scenario_id: str = "hajime_regular"  # 'hajime_regular' | 'hajime_pro' | 'hajime_master' | 'nia_master'
    week: int = 1
    total_weeks: int = 13
    week_activity: str = ""
    idol_stats: Dict[str, int] = Field(
        default_factory=lambda: {"vocal": 180, "dance": 160, "visual": 150}
    )
    p_items: List[Dict[str, str]] = Field(default_factory=list)
    stage_bonuses: Dict[str, Any] = Field(default_factory=dict)
    buffs: Dict[str, int] = Field(default_factory=dict)
    buff_hud_occluded: bool = False
    cards: List[RecognizedCard] = Field(default_factory=list)
    drinks: List[RecognizedDrink] = Field(default_factory=list)
    options: List[Dict[str, Any]] = Field(default_factory=list)
    raw_text: Optional[str] = None


class LegalAction(BaseModel):
    """Executable candidate action presented to decision engine."""

    choice_id: int
    action_type: str  # 'play_card' | 'use_drink' | 'end_turn' | 'pick_option'
    description: str
    bbox: BoundingBox
    payload: Dict[str, Any] = Field(default_factory=dict)


class DecisionResult(BaseModel):
    """Structured decision output from LLM, rule skip, or heuristic fallback."""

    choice_id: int
    reason: str
    source: str = "llm"  # 'llm' | 'rule_skip' | 'fallback'
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
