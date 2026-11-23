"""
Custom Recognition module for MaaFramework and Pure-Python Simulator.
Performs scene classification and state extraction across:
- SceneRouting
- BattleTurn
- ProduceSchedule
- CardReward
- Shop
- PopupConfirmation
- SkipCutscene
"""

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from demo.schemas import BoundingBox, ScreenObservation
from demo.core.db import GakumasDB
from demo.vision.recognizer import enrich_observation_with_week, recognize_screen
from demo.vision.roi import BASE_WIDTH, BASE_HEIGHT

logger = logging.getLogger(__name__)

# Try importing MaaFramework definitions
try:
    from maa.custom_recognition import CustomRecognition as MaaCustomRecognition, CustomRecognitionResult
    from maa.define import Rect
    _HAS_MAA = True
except ImportError:
    class MaaCustomRecognition:  # type: ignore
        pass

    @dataclass
    class Rect:  # type: ignore
        x: int
        y: int
        w: int
        h: int

    @dataclass
    class CustomRecognitionResult:  # type: ignore
        box: Rect
        detail: str

    _HAS_MAA = False


def classify_scene(frame: np.ndarray) -> str:
    """
    Empirical color and contour heuristic to classify current scene type from RGB frame.
    Returns: 'popup' | 'battle_turn' | 'produce_schedule' | 'card_reward' | 'shop' | 'skip' | 'unknown'
    """
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return "unknown"

    h, w = frame.shape[:2]
    if h < 1100 or w < 860:
        return "unknown"

    # 1. Popup check: Confirm button at [1000:1100, 560:880] (Green button)
    pop_zone = frame[1020:1080, 580:860]
    if pop_zone.size > 0:
        green_mask = (pop_zone[:, :, 1] > 120) & (np.int16(pop_zone[:, :, 1]) - np.int16(pop_zone[:, :, 0]) > 30)
        if np.mean(green_mask) > 0.35:
            return "popup"

    # 2. Skip cutscene check: Skip button at top right [40:120, 900:1040]
    if h >= 120 and w >= 1040:
        skip_zone = frame[40:120, 900:1040]
        if skip_zone.size > 0:
            white_mask = (skip_zone[:, :, 0] > 200) & (skip_zone[:, :, 1] > 200) & (skip_zone[:, :, 2] > 200)
            if np.mean(white_mask) > 0.08:
                return "skip"

    # 3. Battle turn check: End turn button at bottom right [1770:1870, 870:1030] (Red button)
    if h >= 1880 and w >= 1040:
        end_turn_zone = frame[1770:1870, 870:1030]
        if end_turn_zone.size > 0:
            red_mask = (end_turn_zone[:, :, 0] > 130) & (np.int16(end_turn_zone[:, :, 0]) - np.int16(end_turn_zone[:, :, 1]) > 40)
            if np.mean(red_mask) > 0.35:
                return "battle_turn"

    # 4. Produce schedule check: Option select buttons at [520:620, 760:940] (Orange button)
    if h >= 650 and w >= 940:
        orange_zone = frame[520:620, 760:940]
        if orange_zone.size > 0:
            orange_mask = (orange_zone[:, :, 0] > 150) & (orange_zone[:, :, 1] > 100) & (orange_zone[:, :, 2] < 100)
            if np.mean(orange_mask) > 0.25:
                return "produce_schedule"

    # 5. Shop check: Item slot region has high variance at [600:700, 100:980]
    if h >= 700 and w >= 980:
        shop_zone = frame[600:700, 100:980]
        if shop_zone.size > 0 and np.std(shop_zone) > 40.0:
            return "shop"

    # 6. Card reward check: Reward cards region [750:1110, 100:980]
    if h >= 1110 and w >= 980:
        reward_zone = frame[750:1110, 100:980]
        if reward_zone.size > 0 and np.std(reward_zone) > 20.0:
            return "card_reward"

    return "battle_turn"


class BaseGakumasRecognition(MaaCustomRecognition):
    """
    Base class compatible with both MaaFramework CustomRecognition and
    pure-Python state machine simulator.
    """

    def __init__(self, db: Optional[GakumasDB] = None, plan: str = "sense"):
        super().__init__()
        self.db = db or GakumasDB()
        self.plan = plan.lower()

    def recognize(self, frame: np.ndarray) -> Optional[ScreenObservation]:
        """Direct Python recognition method returning ScreenObservation."""
        raise NotImplementedError

    def analyze(self, context: Any, argv: Any) -> Optional[Any]:
        """MaaFramework recognition callback."""
        image = getattr(argv, "image", None)
        if image is None:
            if isinstance(argv, np.ndarray):
                image = argv
            else:
                return None

        obs = self.recognize(image)
        if obs is None:
            return None

        detail_json = obs.model_dump_json() if hasattr(obs, "model_dump_json") else json.dumps(obs)
        return CustomRecognitionResult(
            box=Rect(0, 0, BASE_WIDTH, BASE_HEIGHT),
            detail=detail_json,
        )


class GakumasSceneRoutingRecognition(BaseGakumasRecognition):
    """Entry scene classifier that determines current scene and routes accordingly."""

    def __init__(self, db: Optional[GakumasDB] = None, plan: str = "sense", scenario_id: str = "hajime_regular"):
        super().__init__(db=db, plan=plan)
        self.scenario_id = scenario_id
        self._live_recognizer: Optional[Any] = None
        self._banned_drink_ids: List[int] = [19]

    def set_banned_drink_ids(self, drink_ids: Optional[Any]) -> None:
        self._banned_drink_ids = [int(x) for x in (drink_ids or ())]
        if self._live_recognizer is not None and hasattr(self._live_recognizer, "set_banned_drink_ids"):
            self._live_recognizer.set_banned_drink_ids(self._banned_drink_ids)

    def _get_live_recognizer(self) -> Any:
        if self._live_recognizer is None:
            from demo.vision.live_adb_recognizer import LiveScreenRecognizer
            self._live_recognizer = LiveScreenRecognizer(db=self.db, plan=self.plan, scenario_id=self.scenario_id)
            self._live_recognizer.set_banned_drink_ids(self._banned_drink_ids)
        return self._live_recognizer

    def export_state(self) -> Dict[str, Any]:
        """Serialize live recognizer fallback state (if initialized) for checkpoint recovery."""
        if self._live_recognizer is None or not hasattr(self._live_recognizer, "export_state"):
            return {}
        return self._live_recognizer.export_state()

    def load_state(self, data: Dict[str, Any]) -> None:
        """Restore live recognizer fallback state on resume."""
        if not data:
            return
        reco = self._get_live_recognizer()
        if hasattr(reco, "load_state"):
            reco.load_state(data)

    def recognize(self, frame: np.ndarray) -> Optional[ScreenObservation]:
        # Real Android / Desktop capture resolutions (e.g. 720x1600) use LiveScreenRecognizer
        if frame is not None and frame.ndim == 3 and frame.shape[:2] != (BASE_HEIGHT, BASE_WIDTH):
            return self._get_live_recognizer().recognize(frame, plan=self.plan)

        scene = classify_scene(frame)
        obs: Optional[ScreenObservation] = None
        if scene == "battle_turn":
            obs = recognize_screen(frame, plan=self.plan, db=self.db)
        elif scene == "produce_schedule":
            reco = GakumasProduceRecognition(db=self.db, plan=self.plan)
            obs = reco.recognize(frame)
        elif scene == "card_reward":
            reco = GakumasRewardRecognition(db=self.db, plan=self.plan)
            obs = reco.recognize(frame)
        elif scene == "shop":
            reco = GakumasShopRecognition(db=self.db, plan=self.plan)
            obs = reco.recognize(frame)
        elif scene == "popup":
            reco = GakumasPopupRecognition(db=self.db, plan=self.plan)
            obs = reco.recognize(frame)
        elif scene == "skip":
            reco = GakumasSkipRecognition(db=self.db, plan=self.plan)
            obs = reco.recognize(frame)
        else:
            obs = recognize_screen(frame, plan=self.plan, db=self.db)

        if obs is not None and scene != "battle_turn":
            enrich_observation_with_week(obs, frame)
        return obs


class GakumasBattleRecognition(BaseGakumasRecognition):
    """Succeeds only on battle turn; populates cards, drinks, and combat status indicators."""

    def recognize(self, frame: np.ndarray) -> Optional[ScreenObservation]:
        # Perform full battle recognition
        obs = recognize_screen(frame, plan=self.plan, db=self.db)
        return obs


class GakumasProduceRecognition(BaseGakumasRecognition):
    """Succeeds on produce schedule; populates lesson selection options with explicit BoundingBox."""

    def recognize(self, frame: np.ndarray) -> Optional[ScreenObservation]:
        # 3 standard lesson options
        options = [
            {
                "choice_id": 0,
                "name": "Vocal Lesson",
                "title": "Vocal Lesson",
                "type": "lesson",
                "bonus": "+30 Vo",
                "bbox": BoundingBox(xmin=100, ymin=450, xmax=980, ymax=690),
            },
            {
                "choice_id": 1,
                "name": "Dance Lesson",
                "title": "Dance Lesson",
                "type": "lesson",
                "bonus": "+30 Da",
                "bbox": BoundingBox(xmin=100, ymin=750, xmax=980, ymax=990),
            },
            {
                "choice_id": 2,
                "name": "Visual Lesson",
                "title": "Visual Lesson",
                "type": "lesson",
                "bonus": "+30 Vi",
                "bbox": BoundingBox(xmin=100, ymin=1050, xmax=980, ymax=1290),
            },
        ]
        return ScreenObservation(
            scene_type="produce_schedule",
            turn=1,
            turns_remaining=10,
            stamina=30,
            max_stamina=30,
            genki=0,
            plan=self.plan,
            options=options,
        )


class GakumasRewardRecognition(BaseGakumasRecognition):
    """Succeeds on 3-choice card reward screen."""

    def recognize(self, frame: np.ndarray) -> Optional[ScreenObservation]:
        cards_pool = self.db.get_cards_for_plan(self.plan)
        card_0 = cards_pool[0] if len(cards_pool) > 0 else {"id": 101, "name": "Card 1"}
        card_1 = cards_pool[1] if len(cards_pool) > 1 else {"id": 102, "name": "Card 2"}
        card_2 = cards_pool[2] if len(cards_pool) > 2 else {"id": 103, "name": "Card 3"}

        options = [
            {
                "choice_id": 0,
                "card_id": card_0.get("id", 101),
                "name": card_0.get("name", "Card 1"),
                "title": card_0.get("name", "Card 1"),
                "bbox": BoundingBox(xmin=100, ymin=750, xmax=360, ymax=1110),
            },
            {
                "choice_id": 1,
                "card_id": card_1.get("id", 102),
                "name": card_1.get("name", "Card 2"),
                "title": card_1.get("name", "Card 2"),
                "bbox": BoundingBox(xmin=410, ymin=750, xmax=670, ymax=1110),
            },
            {
                "choice_id": 2,
                "card_id": card_2.get("id", 103),
                "name": card_2.get("name", "Card 3"),
                "title": card_2.get("name", "Card 3"),
                "bbox": BoundingBox(xmin=720, ymin=750, xmax=980, ymax=1110),
            },
        ]
        return ScreenObservation(
            scene_type="card_reward",
            turn=1,
            turns_remaining=10,
            stamina=30,
            max_stamina=30,
            genki=0,
            plan=self.plan,
            options=options,
        )


class GakumasShopRecognition(BaseGakumasRecognition):
    """Succeeds on item shop screen."""

    def recognize(self, frame: np.ndarray) -> Optional[ScreenObservation]:
        drinks_pool = self.db.get_drinks_for_plan(self.plan)
        drink_0 = drinks_pool[0] if len(drinks_pool) > 0 else {"id": 1, "name": "Drink 1"}
        drink_1 = drinks_pool[1] if len(drinks_pool) > 1 else {"id": 2, "name": "Drink 2"}

        options = [
            {
                "choice_id": 0,
                "drink_id": drink_0.get("id", 1),
                "name": drink_0.get("name", "Drink 1"),
                "title": drink_0.get("name", "Drink 1"),
                "price": 120,
                "bbox": BoundingBox(xmin=100, ymin=600, xmax=500, ymax=920),
            },
            {
                "choice_id": 1,
                "drink_id": drink_1.get("id", 2),
                "name": drink_1.get("name", "Drink 2"),
                "title": drink_1.get("name", "Drink 2"),
                "price": 120,
                "bbox": BoundingBox(xmin=580, ymin=600, xmax=980, ymax=920),
            },
        ]
        return ScreenObservation(
            scene_type="shop",
            turn=1,
            turns_remaining=10,
            stamina=30,
            max_stamina=30,
            genki=0,
            plan=self.plan,
            options=options,
        )


class GakumasPopupRecognition(BaseGakumasRecognition):
    """Succeeds on modal confirmation dialogs."""

    def recognize(self, frame: np.ndarray) -> Optional[ScreenObservation]:
        btn_cancel = BoundingBox(xmin=200, ymin=1000, xmax=520, ymax=1100)
        btn_confirm = BoundingBox(xmin=560, ymin=1000, xmax=880, ymax=1100)

        options = [
            {
                "choice_id": 0,
                "name": "CANCEL",
                "title": "CANCEL",
                "action": "cancel",
                "bbox": btn_cancel,
            },
            {
                "choice_id": 1,
                "name": "CONFIRM",
                "title": "CONFIRM",
                "action": "confirm",
                "bbox": btn_confirm,
            },
        ]
        return ScreenObservation(
            scene_type="popup",
            turn=1,
            turns_remaining=10,
            stamina=30,
            max_stamina=30,
            genki=0,
            plan=self.plan,
            options=options,
        )


class GakumasSkipRecognition(BaseGakumasRecognition):
    """Succeeds when cutscene / skip button is present."""

    def recognize(self, frame: np.ndarray) -> Optional[ScreenObservation]:
        btn_skip = BoundingBox(xmin=900, ymin=40, xmax=1040, ymax=120)
        options = [
            {
                "choice_id": 0,
                "name": "SKIP",
                "title": "SKIP",
                "action": "skip",
                "bbox": btn_skip,
            }
        ]
        return ScreenObservation(
            scene_type="skip",
            turn=1,
            turns_remaining=10,
            stamina=30,
            max_stamina=30,
            genki=0,
            plan=self.plan,
            options=options,
        )
