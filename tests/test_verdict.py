"""Unit tests for the safety-critical rule engine (src/verdict.py).

These tests run against hand-built FastenerStatus lists — no model, no
image, no network call — because this is the safety-critical decision path
and it must be fully verifiable on every commit, fast, in CI.
"""

from __future__ import annotations

import pytest

from src.models import (
    BoundingBox,
    Detection,
    DetectionClass,
    ExpectedPosition,
    FastenerStatus,
    Severity,
    VerdictLabel,
)
from src.verdict import RulesError, load_rules, evaluate

RULES = load_rules()


def _status(label: str, cls: DetectionClass | None, confidence: float = 0.9) -> FastenerStatus:
    """Build a FastenerStatus; cls=None produces a "missing" status."""
    pos = ExpectedPosition(label=label, x=0, y=0)
    if cls is None:
        return FastenerStatus(position=pos, detection=None)
    det = Detection(
        detection_class=cls,
        confidence=confidence,
        bbox=BoundingBox(x=0, y=0, width=10, height=10),
    )
    return FastenerStatus(position=pos, detection=det, distance_px=2.0)


class TestLoadRules:
    def test_loads_successfully(self):
        rules = load_rules()
        assert "class_severity" in rules
        assert set(rules["class_severity"]) == {c.value for c in DetectionClass}

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(RulesError, match="not found"):
            load_rules(tmp_path / "nope.yaml")

    def test_rejects_incomplete_class_severity(self, tmp_path):
        bad = tmp_path / "rules.yaml"
        bad.write_text(
            "class_severity:\n  bolt_ok: none\n"
            "missing_severity: critical\n"
            "min_confidence: 0.5\n"
            "verdict_thresholds:\n  - if_worst_severity: none\n    verdict: safe\n"
            "low_confidence_verdict_cap: needs_inspection\n"
        )
        with pytest.raises(RulesError, match="missing class_severity entries"):
            load_rules(bad)


class TestEvaluate:
    def test_all_ok_is_safe(self):
        statuses = [_status("a", DetectionClass.BOLT_OK), _status("b", DetectionClass.BOLT_OK)]
        v = evaluate(statuses, RULES)
        assert v.overall == VerdictLabel.SAFE
        assert v.worst_severity == Severity.NONE
        assert v.findings == []

    def test_missing_fastener_is_unsafe(self):
        statuses = [_status("a", DetectionClass.BOLT_OK), _status("b", None)]
        v = evaluate(statuses, RULES)
        assert v.overall == VerdictLabel.UNSAFE
        assert v.worst_severity == Severity.CRITICAL
        assert len(v.findings) == 1
        assert "not detected" in v.findings[0].reason

    def test_corroded_bolt_is_unsafe_major(self):
        statuses = [_status("a", DetectionClass.BOLT_CORRODED)]
        v = evaluate(statuses, RULES)
        assert v.overall == VerdictLabel.UNSAFE
        assert v.worst_severity == Severity.MAJOR

    def test_damaged_bolt_is_unsafe_critical(self):
        statuses = [_status("a", DetectionClass.BOLT_DAMAGED)]
        v = evaluate(statuses, RULES)
        assert v.overall == VerdictLabel.UNSAFE
        assert v.worst_severity == Severity.CRITICAL

    def test_worst_severity_wins_across_multiple_fasteners(self):
        # one fine, one corroded (major), one missing (critical) -> overall critical/unsafe
        statuses = [
            _status("a", DetectionClass.BOLT_OK),
            _status("b", DetectionClass.BOLT_CORRODED),
            _status("c", None),
        ]
        v = evaluate(statuses, RULES)
        assert v.worst_severity == Severity.CRITICAL
        assert v.overall == VerdictLabel.UNSAFE
        assert len(v.findings) == 2  # corroded + missing; the "ok" one contributes nothing

    def test_low_confidence_caps_verdict_even_when_class_is_ok(self):
        # bolt_ok would normally be "safe", but low confidence must never be
        # reported as an unqualified "safe" — this is the false-negative guard.
        statuses = [_status("a", DetectionClass.BOLT_OK, confidence=0.2)]
        v = evaluate(statuses, RULES)
        assert v.low_confidence_flagged is True
        assert v.overall == VerdictLabel.NEEDS_INSPECTION

    def test_low_confidence_does_not_downgrade_an_already_worse_verdict(self):
        # unsafe stays unsafe even if it's also low-confidence — the cap only
        # ever raises the floor, never lowers an already-worse verdict.
        statuses = [_status("a", DetectionClass.BOLT_DAMAGED, confidence=0.1)]
        v = evaluate(statuses, RULES)
        assert v.overall == VerdictLabel.UNSAFE

    def test_empty_statuses_is_safe(self):
        v = evaluate([], RULES)
        assert v.overall == VerdictLabel.SAFE
        assert v.findings == []

    @pytest.mark.parametrize(
        "cls,expected_verdict",
        [
            (DetectionClass.BOLT_OK, VerdictLabel.SAFE),
            (DetectionClass.BOLT_MISALIGNED, VerdictLabel.UNSAFE),
            (DetectionClass.BOLT_CORRODED, VerdictLabel.UNSAFE),
            (DetectionClass.BOLT_DAMAGED, VerdictLabel.UNSAFE),
        ],
    )
    def test_single_fastener_verdict_matrix(self, cls, expected_verdict):
        v = evaluate([_status("a", cls)], RULES)
        assert v.overall == expected_verdict
