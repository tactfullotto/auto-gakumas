"""
Vision Recognizer for Gakumas cards, drinks, and screen states.
Supports plan whitelist filtering (sense, logic, anomaly) for high speed and accuracy.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from demo.schemas import BoundingBox, RecognizedCard, RecognizedDrink, ScreenObservation
from demo.core.db import GakumasDB
from demo.vision.roi import (
    BASE_WIDTH,
    BASE_HEIGHT,
    extract_hand_rois,
    extract_drink_rois,
    StatusOCRExtractor,
)


class CardRecognizer:
    """
    Recognizes skill cards in player's hand using plan-filtered template matching.
    Detects card playability (brightness/dimming) and upgraded '+' status.
    """

    FEATURE_SIZE = (48, 48)
    PLAYABLE_BRIGHTNESS_THRESHOLD = 0.70

    _GLOBAL_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {}
    _GLOBAL_FEATURE_MATRICES: Dict[str, np.ndarray] = {}

    def __init__(self, db: Optional[GakumasDB] = None, plan: Optional[str] = "sense"):
        self.db = db or GakumasDB()
        self._templates_by_plan: Dict[str, List[Dict[str, Any]]] = {}
        self._feature_matrices: Dict[str, np.ndarray] = {}
        if plan:
            self.load_templates(plan)

    def load_templates(self, plan: str) -> None:
        """Load and cache card templates and normalized feature vectors for a plan."""
        plan_lower = plan.lower()
        if plan_lower in self._templates_by_plan:
            return

        if plan_lower in CardRecognizer._GLOBAL_TEMPLATES:
            self._templates_by_plan[plan_lower] = CardRecognizer._GLOBAL_TEMPLATES[plan_lower]
            self._feature_matrices[plan_lower] = CardRecognizer._GLOBAL_FEATURE_MATRICES[plan_lower]
            return

        cards = self.db.get_cards_for_plan(plan_lower)
        loaded: List[Dict[str, Any]] = []
        vecs: List[np.ndarray] = []

        for card in cards:
            icon_path = card.get("icon_path")
            if not icon_path or not Path(icon_path).exists():
                continue

            try:
                img = Image.open(icon_path).convert("RGB")
                # Downsample to FEATURE_SIZE for fast vectorized dot product
                thumb = img.resize(self.FEATURE_SIZE)
                arr = np.array(thumb, dtype=np.float32).flatten()
                arr_mean = arr.mean()
                arr_norm = arr - arr_mean
                norm = np.linalg.norm(arr_norm)
                if norm < 1e-4:
                    continue
                unit_vec = arr_norm / norm

                loaded.append({
                    "card": card,
                    "img_size": img.size,
                    "mean_brightness": float(arr_mean),
                    "icon_path": icon_path,
                })
                vecs.append(unit_vec)
            except Exception:
                continue

        mat = np.array(vecs, dtype=np.float32) if vecs else np.zeros((0, self.FEATURE_SIZE[0] * self.FEATURE_SIZE[1] * 3), dtype=np.float32)

        CardRecognizer._GLOBAL_TEMPLATES[plan_lower] = loaded
        CardRecognizer._GLOBAL_FEATURE_MATRICES[plan_lower] = mat

        self._templates_by_plan[plan_lower] = loaded
        self._feature_matrices[plan_lower] = mat

    def detect_upgraded_badge(self, card_roi: np.ndarray) -> bool:
        """
        Check for '+' badge in the top-right corner of card ROI.
        Badge is typically a yellow/gold circle with high contrast '+' sign.
        """
        h, w = card_roi.shape[:2]
        if h < 40 or w < 40:
            return False

        # Corner ROI: top right
        corner = card_roi[int(h * 0.02):int(h * 0.18), int(w * 0.75):int(w * 0.98)]
        if corner.size == 0:
            return False

        # Check for yellow/gold badge color: R > 170, G > 130, B < 90
        if corner.ndim == 3:
            r = corner[:, :, 0].astype(float)
            g = corner[:, :, 1].astype(float)
            b = corner[:, :, 2].astype(float)
            gold_mask = (r > 160) & (g > 120) & (b < 100) & (r - b > 50)
            gold_pixels = np.count_nonzero(gold_mask)
            # If significant cluster of golden badge pixels exists
            if gold_pixels > 25:
                return True

        return False

    def detect_playability(self, card_roi: np.ndarray, template_mean: float) -> bool:
        """
        Detect card playability. Unplayable/dimmed cards have significantly lower
        brightness compared to template.
        """
        if card_roi.size == 0 or template_mean <= 1e-3:
            return True

        roi_mean = float(np.mean(card_roi))
        ratio = roi_mean / template_mean
        return ratio >= self.PLAYABLE_BRIGHTNESS_THRESHOLD

    def recognize_card(
        self,
        card_roi: np.ndarray,
        bbox: BoundingBox,
        plan: Optional[str] = "sense",
    ) -> Optional[RecognizedCard]:
        """Recognize a single skill card from a cropped hand card slot ROI."""
        plan_lower = (plan or "sense").lower()
        if plan_lower not in self._templates_by_plan:
            self.load_templates(plan_lower)

        templates = self._templates_by_plan.get(plan_lower, [])
        feature_matrix = self._feature_matrices.get(plan_lower)
        if not templates or feature_matrix is None or len(templates) == 0:
            return None

        h, w = card_roi.shape[:2]
        if h < 20 or w < 20:
            return None

        # Extract central icon region from card ROI (roughly y: [10%, 70%], x: [10%, 90%])
        y1, y2 = int(h * 0.10), int(h * 0.70)
        x1, x2 = int(w * 0.10), int(w * 0.90)
        icon_crop = card_roi[y1:y2, x1:x2]
        if icon_crop.size == 0:
            icon_crop = card_roi

        # Convert crop to PIL and resize to feature size
        pil_crop = Image.fromarray(icon_crop if icon_crop.ndim == 3 else np.stack([icon_crop]*3, axis=-1))
        thumb = pil_crop.resize(self.FEATURE_SIZE)
        arr = np.array(thumb, dtype=np.float32).flatten()
        arr_norm = arr - arr.mean()
        norm = np.linalg.norm(arr_norm)
        if norm < 1e-4:
            return None
        roi_vec = arr_norm / norm

        # Vectorized dot product against all templates for the plan
        scores = feature_matrix @ roi_vec
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        # Top candidate card
        matched_item = templates[best_idx]
        card_data = matched_item["card"]

        # Check upgraded status: template upgraded or corner badge
        badge_present = self.detect_upgraded_badge(card_roi)
        is_upgraded = bool(card_data.get("upgraded", 0) == 1) or badge_present

        # If '+' badge detected on an unupgraded template match, promote to upgraded card
        final_card = card_data
        if badge_present and card_data.get("upgraded", 0) == 0:
            upgraded_id = card_data.get("upgraded_id")
            if upgraded_id:
                up_card = self.db.get_card_by_id(upgraded_id)
                if up_card:
                    final_card = up_card
                    is_upgraded = True

        # Check playability status
        playable = self.detect_playability(icon_crop, matched_item["mean_brightness"])

        return RecognizedCard(
            card_id=final_card["id"],
            name=final_card.get("name") or "",
            upgraded=is_upgraded,
            playable=playable,
            bbox=bbox,
            hp_loss=0,
            dsl_summary=final_card.get("description") or "",
            type=final_card.get("type") or "active",
            rarity=final_card.get("rarity") or "N",
        )

    def recognize_hand(
        self,
        frame: np.ndarray,
        count: int = 3,
        plan: Optional[str] = "sense",
    ) -> List[RecognizedCard]:
        """Extract and recognize all cards in the player's hand."""
        hand_rois = extract_hand_rois(frame, count=count)
        cards: List[RecognizedCard] = []
        for bbox, crop in hand_rois:
            rec = self.recognize_card(crop, bbox, plan=plan)
            if rec is not None:
                cards.append(rec)
        return cards


class DrinkRecognizer:
    """
    Recognizes P-Drinks in drink slots using plan-filtered template matching.
    Identifies empty slots and filled drink slots.
    """

    FEATURE_SIZE = (48, 48)
    CONFIDENCE_THRESHOLD = 0.60
    EMPTY_SLOT_STD_THRESHOLD = 15.0

    _GLOBAL_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {}
    _GLOBAL_FEATURE_MATRICES: Dict[str, np.ndarray] = {}

    def __init__(self, db: Optional[GakumasDB] = None, plan: Optional[str] = "sense"):
        self.db = db or GakumasDB()
        self._templates_by_plan: Dict[str, List[Dict[str, Any]]] = {}
        self._feature_matrices: Dict[str, np.ndarray] = {}
        if plan:
            self.load_templates(plan)

    def load_templates(self, plan: str) -> None:
        """Load and cache 16 drink templates for a plan."""
        plan_lower = plan.lower()
        if plan_lower in self._templates_by_plan:
            return

        if plan_lower in DrinkRecognizer._GLOBAL_TEMPLATES:
            self._templates_by_plan[plan_lower] = DrinkRecognizer._GLOBAL_TEMPLATES[plan_lower]
            self._feature_matrices[plan_lower] = DrinkRecognizer._GLOBAL_FEATURE_MATRICES[plan_lower]
            return

        drinks = self.db.get_drinks_for_plan(plan_lower)
        loaded: List[Dict[str, Any]] = []
        vecs: List[np.ndarray] = []

        for drink in drinks:
            icon_path = drink.get("icon_path")
            if not icon_path or not Path(icon_path).exists():
                continue

            try:
                img = Image.open(icon_path).convert("RGB")
                thumb = img.resize(self.FEATURE_SIZE)
                arr = np.array(thumb, dtype=np.float32).flatten()
                arr_norm = arr - arr.mean()
                norm = np.linalg.norm(arr_norm)
                if norm < 1e-4:
                    continue
                unit_vec = arr_norm / norm

                loaded.append({
                    "drink": drink,
                    "img": img,
                    "icon_path": icon_path,
                })
                vecs.append(unit_vec)
            except Exception:
                continue

        mat = np.array(vecs, dtype=np.float32) if vecs else np.zeros((0, self.FEATURE_SIZE[0] * self.FEATURE_SIZE[1] * 3), dtype=np.float32)

        DrinkRecognizer._GLOBAL_TEMPLATES[plan_lower] = loaded
        DrinkRecognizer._GLOBAL_FEATURE_MATRICES[plan_lower] = mat

        self._templates_by_plan[plan_lower] = loaded
        self._feature_matrices[plan_lower] = mat

    def is_empty_slot(self, drink_roi: np.ndarray) -> bool:
        """Check if drink slot is empty (low variance or dark uniform background)."""
        if drink_roi.size == 0:
            return True
        std = float(np.std(drink_roi))
        return std < self.EMPTY_SLOT_STD_THRESHOLD

    def recognize_drink(
        self,
        drink_roi: np.ndarray,
        bbox: BoundingBox,
        plan: Optional[str] = "sense",
    ) -> Optional[RecognizedDrink]:
        """Recognize a P-Drink from a drink slot ROI or identify empty slot."""
        if self.is_empty_slot(drink_roi):
            return None

        plan_lower = (plan or "sense").lower()
        if plan_lower not in self._templates_by_plan:
            self.load_templates(plan_lower)

        templates = self._templates_by_plan.get(plan_lower, [])
        feature_matrix = self._feature_matrices.get(plan_lower)
        if not templates or feature_matrix is None or len(templates) == 0:
            return None

        # Take inner crop (removing outer slot border)
        h, w = drink_roi.shape[:2]
        pad_y = int(h * 0.06)
        pad_x = int(w * 0.06)
        inner = drink_roi[pad_y:h - pad_y, pad_x:w - pad_x]
        if inner.size == 0:
            inner = drink_roi

        # Resize ROI to feature size and normalize
        pil_crop = Image.fromarray(inner if inner.ndim == 3 else np.stack([inner]*3, axis=-1))
        thumb = pil_crop.resize(self.FEATURE_SIZE)
        arr = np.array(thumb, dtype=np.float32).flatten()
        arr_norm = arr - arr.mean()
        norm = np.linalg.norm(arr_norm)
        if norm < 1e-4:
            return None
        roi_vec = arr_norm / norm

        # Vectorized match
        scores = feature_matrix @ roi_vec
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        # If below threshold, treat as empty or unconfirmed slot
        if best_score < self.CONFIDENCE_THRESHOLD:
            return None

        drink_data = templates[best_idx]["drink"]
        return RecognizedDrink(
            drink_id=drink_data["id"],
            name=drink_data.get("name") or "",
            playable=True,
            bbox=bbox,
            dsl_summary=drink_data.get("description") or "",
        )

    def recognize_drinks(
        self,
        frame: np.ndarray,
        count: int = 3,
        plan: Optional[str] = "sense",
    ) -> List[RecognizedDrink]:
        """Extract and recognize all drinks in HUD slots (skipping empty slots)."""
        drink_rois = extract_drink_rois(frame, count=count)
        drinks: List[RecognizedDrink] = []
        for bbox, crop in drink_rois:
            rec = self.recognize_drink(crop, bbox, plan=plan)
            if rec is not None:
                drinks.append(rec)
        return drinks


_SCREEN_CARD_RECOGNIZERS: Dict[str, CardRecognizer] = {}
_SCREEN_DRINK_RECOGNIZERS: Dict[str, DrinkRecognizer] = {}
_SHARED_OCR: Optional[StatusOCRExtractor] = None


def get_card_recognizer(plan: str = "sense", db: Optional[GakumasDB] = None) -> CardRecognizer:
    """Get or create cached CardRecognizer instance for a plan."""
    plan_lower = plan.lower()
    rec = _SCREEN_CARD_RECOGNIZERS.get(plan_lower)
    if rec is None:
        rec = CardRecognizer(db=db, plan=plan_lower)
        _SCREEN_CARD_RECOGNIZERS[plan_lower] = rec
    elif db is not None:
        rec.db = db
    return rec


def get_drink_recognizer(plan: str = "sense", db: Optional[GakumasDB] = None) -> DrinkRecognizer:
    """Get or create cached DrinkRecognizer instance for a plan."""
    plan_lower = plan.lower()
    rec = _SCREEN_DRINK_RECOGNIZERS.get(plan_lower)
    if rec is None:
        rec = DrinkRecognizer(db=db, plan=plan_lower)
        _SCREEN_DRINK_RECOGNIZERS[plan_lower] = rec
    elif db is not None:
        rec.db = db
    return rec


def clear_recognizer_cache() -> None:
    """Clear all module-level and class-level recognizer caches."""
    _SCREEN_CARD_RECOGNIZERS.clear()
    _SCREEN_DRINK_RECOGNIZERS.clear()
    CardRecognizer._GLOBAL_TEMPLATES.clear()
    CardRecognizer._GLOBAL_FEATURE_MATRICES.clear()
    DrinkRecognizer._GLOBAL_TEMPLATES.clear()
    DrinkRecognizer._GLOBAL_FEATURE_MATRICES.clear()


def enrich_observation_with_week(
    obs: ScreenObservation,
    frame: np.ndarray,
    scenario_id: Optional[str] = None,
) -> ScreenObservation:
    """
    Recognize the current week number from the frame header ROI (`STATUS_WEEK_BBOX`)
    and populate scenario/week flow metadata (`week_activity`, `idol_stats`, `p_items`,
    `stage_bonuses`, and schedule options) from `demo.core.scenarios`.
    """
    global _SHARED_OCR
    if _SHARED_OCR is None:
        _SHARED_OCR = StatusOCRExtractor()
    from demo.vision.roi import extract_status_rois
    from demo.core.scenarios import compute_week_stats_and_stage_bonus

    status_rois = extract_status_rois(frame)
    w, tot = _SHARED_OCR.ocr_week(status_rois["week"][1])
    if w is None:
        return obs

    inferred_scenario = scenario_id
    if not inferred_scenario:
        if tot == 16:
            inferred_scenario = "hajime_pro"
        elif tot == 18:
            inferred_scenario = "hajime_master"
        else:
            inferred_scenario = "hajime_regular"

    ctx = compute_week_stats_and_stage_bonus(
        scenario_id=inferred_scenario,
        week=w,
        plan=obs.plan,
    )
    node = ctx["week_node"]
    obs.scenario_id = ctx["scenario_id"]
    obs.week = node.week
    obs.total_weeks = ctx["total_weeks"]
    obs.week_activity = node.title
    obs.idol_stats = ctx["idol_stats"]
    obs.p_items = ctx["p_items"]
    obs.stage_bonuses = ctx["stage_bonuses"]
    if obs.scene_type == "produce_schedule" and node.scene_type == "produce_schedule":
        for i, act_title in enumerate(node.activities):
            if i < len(obs.options):
                obs.options[i]["name"] = act_title
                obs.options[i]["title"] = act_title
                obs.options[i]["description"] = node.operation_guide
    return obs


def recognize_screen(
    frame: np.ndarray,
    plan: str = "sense",
    card_count: int = 3,
    drink_count: int = 3,
    db: Optional[GakumasDB] = None,
    card_recognizer: Optional[CardRecognizer] = None,
    drink_recognizer: Optional[DrinkRecognizer] = None,
) -> ScreenObservation:
    """
    End-to-end vision pipeline: parses status indicators, hand cards, and drinks
    into a structured ScreenObservation.
    Uses in-memory template and recognizer caching for high-speed throughput (<50ms).
    """
    global _SHARED_OCR
    db_inst = db or GakumasDB()
    if _SHARED_OCR is None:
        _SHARED_OCR = StatusOCRExtractor()
    status_data = _SHARED_OCR.extract_status(frame)

    card_rec = card_recognizer or get_card_recognizer(plan=plan, db=db_inst)
    cards = card_rec.recognize_hand(frame, count=card_count, plan=plan)

    drink_rec = drink_recognizer or get_drink_recognizer(plan=plan, db=db_inst)
    drinks = drink_rec.recognize_drinks(frame, count=drink_count, plan=plan)

    obs = ScreenObservation(
        scene_type="battle_turn",
        turn=status_data.get("turn") or 1,
        turns_remaining=status_data.get("turns_remaining") or 10,
        stamina=status_data.get("stamina") if status_data.get("stamina") is not None else 30,
        max_stamina=status_data.get("max_stamina") or 30,
        genki=status_data.get("genki") if status_data.get("genki") is not None else 0,
        score=status_data.get("score") if status_data.get("score") is not None else 0,
        target_score=status_data.get("target_score") or 5000,
        plan=plan.lower(),
        buffs=status_data.get("buffs") or {},
        cards=cards,
        drinks=drinks,
        options=[],
    )
    if status_data.get("week") is not None:
        enrich_observation_with_week(obs, frame)
    return obs
