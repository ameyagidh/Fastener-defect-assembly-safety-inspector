"""Tests for the pipeline wiring (src/pipeline.py) using a fake detector —
these validate that detect -> match -> verdict -> explain are wired
correctly, without requiring ultralytics, a trained model, or network access.
"""

from __future__ import annotations

from PIL import Image

from src.detect import FastenerDetector
from src.models import BoundingBox, Detection, DetectionClass
from src.pipeline import DEFAULT_TEMPLATE_PATH, inspect


class FakeDetector(FastenerDetector):
    """Returns a fixed detection list instead of running a real model."""

    def __init__(self, detections: list[Detection], is_fine_tuned: bool = True):
        self._fixed_detections = detections
        self.is_fine_tuned = is_fine_tuned
        self.model_path = None
        self.confidence_threshold = 0.25

    def detect(self, image_path):
        return self._fixed_detections


def _make_test_image(path, size=(1000, 1000)):
    Image.new("RGB", size, color="white").save(path)


class TestInspectWiring:
    def test_all_four_corners_present_and_ok_is_safe(self, tmp_path):
        image_path = tmp_path / "test.jpg"
        _make_test_image(image_path)

        # Detections placed exactly at the 4 template corner positions
        # (bracket_4bolt.json uses a 1000x1000 reference frame, matching
        # our test image, so no rescaling math needed here).
        detections = [
            Detection(DetectionClass.BOLT_OK, 0.95, BoundingBox(145, 145, 10, 10)),
            Detection(DetectionClass.BOLT_OK, 0.95, BoundingBox(845, 145, 10, 10)),
            Detection(DetectionClass.BOLT_OK, 0.95, BoundingBox(145, 845, 10, 10)),
            Detection(DetectionClass.BOLT_OK, 0.95, BoundingBox(845, 845, 10, 10)),
        ]
        result = inspect(
            image_path,
            template_path=DEFAULT_TEMPLATE_PATH,
            detector=FakeDetector(detections),
            include_explanation=False,
        )

        assert len(result.statuses) == 4
        assert all(not s.is_missing for s in result.statuses)
        assert result.verdict.overall.value == "safe"
        assert result.explanation == ""  # explanation skipped
        assert result.model_is_fine_tuned is True

    def test_one_missing_bolt_is_unsafe(self, tmp_path):
        image_path = tmp_path / "test.jpg"
        _make_test_image(image_path)

        # Only 3 of 4 expected bolts detected
        detections = [
            Detection(DetectionClass.BOLT_OK, 0.95, BoundingBox(145, 145, 10, 10)),
            Detection(DetectionClass.BOLT_OK, 0.95, BoundingBox(845, 145, 10, 10)),
            Detection(DetectionClass.BOLT_OK, 0.95, BoundingBox(145, 845, 10, 10)),
        ]
        result = inspect(
            image_path,
            detector=FakeDetector(detections),
            include_explanation=False,
        )

        missing = [s for s in result.statuses if s.is_missing]
        assert len(missing) == 1
        assert missing[0].position.label == "bottom_right"
        assert result.verdict.overall.value == "unsafe"

    def test_explanation_falls_back_without_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no_such_config"))

        image_path = tmp_path / "test.jpg"
        _make_test_image(image_path)
        detections = [
            Detection(DetectionClass.BOLT_OK, 0.95, BoundingBox(145, 145, 10, 10)),
        ]
        result = inspect(
            image_path,
            detector=FakeDetector(detections),
            include_explanation=True,
        )
        # Falls back to the deterministic templated string rather than raising
        # or leaving the explanation empty.
        assert result.explanation.startswith("Verdict:")
