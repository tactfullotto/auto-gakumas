"""
Custom Action module for MaaFramework and Pure-Python Simulator.
Translates decisions into controller touch events and updates game state tracking.
"""

from dataclasses import dataclass
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from demo.schemas import BoundingBox, DecisionResult, LegalAction, ScreenObservation
from demo.core.db import GakumasDB
from demo.core.tracker import GameStateTracker
from demo.core.prompt import PromptBuilder
from demo.agent.client import AgentConfig, AgentDecisionMaker
from demo.maa_bridge.base_controller import BaseController

logger = logging.getLogger(__name__)

try:
    from maa.custom_action import CustomAction as MaaCustomAction, RunResult
    _HAS_MAA = True
except ImportError:
    class MaaCustomAction:  # type: ignore
        pass

    @dataclass
    class RunResult:  # type: ignore
        success: bool

    _HAS_MAA = False


class GakumasDecisionAction(MaaCustomAction):
    """
    Executes autonomous decisions across all game scenes:
    1. Reconstructs ScreenObservation from recognition detail.
    2. Builds prompt and legal actions via PromptBuilder.
    3. Obtains decision from AgentDecisionMaker (or fallback heuristic).
    4. Maps chosen choice_id to target BoundingBox center (cx, cy).
    5. Dispatches click via controller.
    6. Updates GameStateTracker if play_card action executed.
    7. Overrides next state via context.override_next.
    8. Invokes optional on_decision / on_action_executed callbacks.
    """

    def __init__(
        self,
        controller: Optional[BaseController] = None,
        agent: Optional[AgentDecisionMaker] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        tracker: Optional[GameStateTracker] = None,
        db: Optional[GakumasDB] = None,
        on_decision: Optional[Callable[[DecisionResult, Optional[LegalAction]], None]] = None,
        on_action_executed: Optional[Callable[[Optional[LegalAction], Tuple[int, int]], None]] = None,
    ):
        super().__init__()
        self.controller = controller
        self.db = db or GakumasDB()
        self.agent = agent or AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True))
        self.prompt_builder = prompt_builder or PromptBuilder(db=self.db)
        self.tracker = tracker or GameStateTracker(plan="sense")
        self.on_decision = on_decision
        self.on_action_executed = on_action_executed

    def execute_decision(
        self,
        decision: DecisionResult,
        legal_actions: List[LegalAction],
    ) -> bool:
        """
        Execute physical/simulated click given a decision and candidate action list.
        Maps choice_id to the exact center coordinates of the target element's BoundingBox.
        """
        chosen_action: Optional[LegalAction] = None
        for act in legal_actions:
            if act.choice_id == decision.choice_id:
                chosen_action = act
                break

        if chosen_action is None and legal_actions:
            # Fallback to action 0 if decision choice_id is invalid
            chosen_action = legal_actions[0]

        if chosen_action is not None:
            click_coords = chosen_action.bbox.center()
        else:
            click_coords = (540, 960)

        if self.on_decision:
            try:
                self.on_decision(decision, chosen_action)
            except Exception as e:
                logger.error("Error in on_decision callback: %s", e)

        # Dispatch click via controller (double-tap for play_card and produce_schedule choices on live devices)
        success = True
        if self.controller:
            from demo.maa_bridge.mock_controller import MockController
            success = self.controller.click(click_coords[0], click_coords[1])
            needs_double_tap = (
                not isinstance(self.controller, MockController)
                and chosen_action is not None
                and (
                    chosen_action.action_type == "play_card"
                    or (
                        chosen_action.action_type == "pick_option"
                        and chosen_action.payload.get("scene_type") in ("produce_schedule", "produce_event")
                    )
                )
            )
            if success and needs_double_tap:
                import time
                time.sleep(0.58)
                self.controller.click(click_coords[0], click_coords[1])

        # Track limit_count=1 card if played
        if chosen_action and chosen_action.action_type == "play_card" and self.tracker:
            card_id = chosen_action.payload.get("card_id")
            if card_id is not None:
                self.tracker.record_card_play(card_id, limit_count=1)

        if self.on_action_executed:
            try:
                self.on_action_executed(chosen_action, click_coords)
            except Exception as e:
                logger.error("Error in on_action_executed callback: %s", e)

        return success

    def run(self, context: Any, argv: Any) -> Any:
        """
        MaaFramework and Simulator action callback.
        Reconstructs observation, invokes decision engine, executes click,
        and overrides next state machine candidate.
        """
        node_name = getattr(argv, "node_name", "BattleTurn")
        obs: Optional[ScreenObservation] = None

        # 1. Parse observation from reco_detail
        reco_detail = getattr(argv, "reco_detail", None)
        if reco_detail is not None:
            detail_val = None
            if hasattr(reco_detail, "best_result") and hasattr(reco_detail.best_result, "detail"):
                detail_val = reco_detail.best_result.detail
            elif isinstance(reco_detail, dict):
                detail_val = reco_detail.get("best", {}).get("detail")
            elif isinstance(reco_detail, str):
                detail_val = reco_detail

            if detail_val:
                try:
                    if isinstance(detail_val, str):
                        data = json.loads(detail_val)
                    else:
                        data = detail_val
                    obs = ScreenObservation.model_validate(data)
                except Exception as e:
                    logger.warning("Failed to parse ScreenObservation from detail: %s", e)

        # Fallback observation if parsing failed
        if obs is None:
            obs = ScreenObservation(scene_type="battle_turn", turn=1, turns_remaining=10)

        # Update tracker
        if self.tracker:
            self.tracker.update_observation(obs)

        # 2. Build prompt and legal actions
        prompt_text, legal_actions = self.prompt_builder.build_prompt(obs, self.tracker)

        # 3. Request decision from agent
        decision = self.agent.decide(prompt=prompt_text, legal_actions=legal_actions, obs=obs)

        # 4. Execute click
        click_ok = self.execute_decision(decision, legal_actions)

        # 5. Advance state machine routing via context.override_next
        if hasattr(context, "override_next"):
            next_targets = ["SceneRouting"]
            if obs.scene_type == "battle_turn":
                next_targets = ["BattleTurn", "PopupConfirmation", "SceneRouting"]
            elif obs.scene_type == "produce_schedule":
                next_targets = ["PopupConfirmation", "BattleTurn", "SceneRouting"]
            elif obs.scene_type == "card_reward":
                next_targets = ["PopupConfirmation", "ProduceSchedule", "SceneRouting"]
            elif obs.scene_type == "popup":
                next_targets = ["SceneRouting", "BattleTurn"]

            try:
                context.override_next(node_name, next_targets)
            except Exception as e:
                logger.warning("context.override_next failed: %s", e)

        return True
