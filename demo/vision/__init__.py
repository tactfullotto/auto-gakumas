"""
Vision and Mock Frame Generator package for Gakumas Demo.
"""

from demo.vision.roi import (
    BASE_WIDTH,
    BASE_HEIGHT,
    BoundingBox,
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
    get_card_recognizer,
    get_drink_recognizer,
    clear_recognizer_cache,
)
from demo.vision.mock_generator import (
    MockFrameGenerator,
)

__all__ = [
    "BASE_WIDTH",
    "BASE_HEIGHT",
    "BoundingBox",
    "scale_bbox",
    "to_normalized",
    "from_normalized",
    "crop_roi",
    "get_hand_bboxes",
    "get_drink_bboxes",
    "get_status_bboxes",
    "extract_hand_rois",
    "extract_drink_rois",
    "extract_status_rois",
    "StatusOCRExtractor",
    "CardRecognizer",
    "DrinkRecognizer",
    "recognize_screen",
    "get_card_recognizer",
    "get_drink_recognizer",
    "clear_recognizer_cache",
    "MockFrameGenerator",
]
