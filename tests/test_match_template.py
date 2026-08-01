"""Unit tests for src/match_template.py — pure logic, hand-built detection
lists, no model or image required.
"""

from __future__ import annotations

import json

import pytest

from src.match_template import (
    TemplateError,
    expected_positions_from_template,
    load_template,
    match,
)
from src.models import BoundingBox, Detection, DetectionClass, ExpectedPosition


def _det(x: float, y: float, cls=DetectionClass.BOLT_OK, confidence: float = 0.9) -> Detection:
    # bbox centered on (x, y)
    return Detection(
        detection_class=cls,
        confidence=confidence,
        bbox=BoundingBox(x=x - 5, y=y - 5, width=10, height=10),
    )


def _pos(label: str, x: float, y: float, tolerance_px: float = 40.0) -> ExpectedPosition:
    return ExpectedPosition(label=label, x=x, y=y, tolerance_px=tolerance_px)


class TestLoadTemplate:
    def test_loads_bundled_bracket_template(self):
        path_root = __file__
        import os

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(path_root)), "templates", "bracket_4bolt.json"
        )
        template = load_template(__import__("pathlib").Path(template_path))
        assert template["name"] == "bracket_4bolt"
        assert len(template["expected_positions"]) == 4

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(TemplateError, match="not found"):
            load_template(tmp_path / "nope.json")

    def test_missing_required_key_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"name": "x", "reference_frame": {"width": 1, "height": 1}}))
        with pytest.raises(TemplateError, match="missing required keys"):
            load_template(bad)

    def test_empty_positions_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps(
                {
                    "name": "x",
                    "reference_frame": {"width": 1, "height": 1},
                    "expected_positions": [],
                }
            )
        )
        with pytest.raises(TemplateError, match="empty expected_positions"):
            load_template(bad)


class TestExpectedPositionsFromTemplate:
    def test_rescales_normalized_coordinates(self):
        template = {
            "reference_frame": {"width": 1000, "height": 1000},
            "expected_positions": [{"label": "a", "x": 500, "y": 500, "tolerance_px": 40}],
        }
        positions = expected_positions_from_template(template, image_width=2000, image_height=1000)
        assert positions[0].x == 1000  # 500 * (2000/1000)
        assert positions[0].y == 500  # 500 * (1000/1000)


class TestMatch:
    def test_all_positions_matched_when_detections_present(self):
        positions = [_pos("a", 100, 100), _pos("b", 500, 500)]
        detections = [_det(500, 500), _det(100, 100)]
        statuses = match(positions, detections)
        assert len(statuses) == 2
        assert all(not s.is_missing for s in statuses)
        by_label = {s.position.label: s for s in statuses}
        assert by_label["a"].distance_px == pytest.approx(0.0, abs=1e-6)
        assert by_label["b"].distance_px == pytest.approx(0.0, abs=1e-6)

    def test_unmatched_position_is_missing(self):
        positions = [_pos("a", 100, 100), _pos("b", 900, 900)]
        detections = [_det(100, 100)]  # nothing near "b"
        statuses = match(positions, detections)
        by_label = {s.position.label: s for s in statuses}
        assert not by_label["a"].is_missing
        assert by_label["b"].is_missing
        assert by_label["b"].detection is None

    def test_detection_outside_tolerance_does_not_match(self):
        positions = [_pos("a", 100, 100, tolerance_px=10)]
        detections = [_det(200, 200)]  # far outside tolerance
        statuses = match(positions, detections)
        assert statuses[0].is_missing

    def test_two_positions_compete_for_closer_detection(self):
        # A single detection sits closer to "b" than to "a"; the greedy
        # nearest-match must award it to "b", leaving "a" missing —
        # not the reverse, and not double-assigned.
        positions = [_pos("a", 0, 0), _pos("b", 100, 100)]
        detections = [_det(90, 90)]
        statuses = match(positions, detections)
        by_label = {s.position.label: s for s in statuses}
        assert not by_label["b"].is_missing
        assert by_label["a"].is_missing

    def test_no_detections_all_missing(self):
        positions = [_pos("a", 0, 0), _pos("b", 100, 100)]
        statuses = match(positions, [])
        assert all(s.is_missing for s in statuses)
        assert len(statuses) == 2

    def test_extra_unclaimed_detections_are_simply_ignored(self):
        # More detections than expected positions (e.g. a stray false
        # positive) should not create phantom statuses or crash.
        positions = [_pos("a", 100, 100)]
        detections = [_det(100, 100), _det(900, 900)]
        statuses = match(positions, detections)
        assert len(statuses) == 1
        assert not statuses[0].is_missing
