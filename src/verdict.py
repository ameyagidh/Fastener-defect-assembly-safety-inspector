"""Rule-based safety verdict engine.

This module is the entire safety-critical decision path in the system. It
contains NO model calls and NO network I/O — every function here is a pure
function of its inputs, driven entirely by rules.yaml. That is deliberate:
whether an assembly is "safe" is a domain judgment that must be transparent,
auditable, and testable without a GPU, not a black-box model output.

See rules.yaml for the actual severity/verdict mapping this module applies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.models import (
    DefectFinding,
    DetectionClass,
    FastenerStatus,
    Severity,
    Verdict,
    VerdictLabel,
)

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "rules.yaml"


class RulesError(ValueError):
    """Raised when rules.yaml is missing a required key or has an invalid shape."""


def load_rules(path: Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    """Load and lightly validate rules.yaml.

    Fails loudly on a malformed rules file rather than falling back to a
    silent default — a wrong severity mapping is exactly the kind of bug
    that must not fail silently in a safety verdict.
    """
    if not path.exists():
        raise RulesError(f"rules file not found: {path}")

    with path.open() as f:
        rules = yaml.safe_load(f)

    required_keys = {
        "class_severity",
        "missing_severity",
        "min_confidence",
        "verdict_thresholds",
        "low_confidence_verdict_cap",
    }
    missing = required_keys - rules.keys()
    if missing:
        raise RulesError(f"rules.yaml missing required keys: {sorted(missing)}")

    known_classes = {c.value for c in DetectionClass}
    unknown_classes = set(rules["class_severity"]) - known_classes
    if unknown_classes:
        raise RulesError(
            f"rules.yaml has class_severity entries for unknown DetectionClass "
            f"values: {sorted(unknown_classes)}"
        )
    missing_classes = known_classes - set(rules["class_severity"])
    if missing_classes:
        raise RulesError(
            f"rules.yaml is missing class_severity entries for: "
            f"{sorted(missing_classes)} — every DetectionClass must have a "
            f"severity mapping so no defect class is silently treated as safe"
        )

    return rules


def _severity_for_status(status: FastenerStatus, rules: dict[str, Any]) -> Severity:
    if status.is_missing:
        return Severity(rules["missing_severity"])
    return Severity(rules["class_severity"][status.detection.detection_class.value])


def _reason_for_status(status: FastenerStatus, severity: Severity) -> str:
    if status.is_missing:
        return f"Expected fastener at '{status.position.label}' was not detected."
    det = status.detection
    return (
        f"Fastener at '{status.position.label}' classified as "
        f"'{det.detection_class.value}' (confidence {det.confidence:.2f}, "
        f"severity {severity.value})."
    )


def evaluate(
    statuses: list[FastenerStatus],
    rules: dict[str, Any] | None = None,
) -> Verdict:
    """Apply rules.yaml to a list of matched fastener statuses and return a Verdict.

    `statuses` is expected to come from src.match_template.match, but this
    function accepts any list of FastenerStatus — including hand-built ones —
    which is what makes it directly unit-testable without a model or an image.
    """
    if rules is None:
        rules = load_rules()

    min_confidence = float(rules["min_confidence"])
    findings: list[DefectFinding] = []
    worst = Severity.NONE
    low_confidence_flagged = False

    for status in statuses:
        severity = _severity_for_status(status, rules)

        if not status.is_missing and status.detection.confidence < min_confidence:
            low_confidence_flagged = True

        if severity.rank > worst.rank:
            worst = severity

        if severity != Severity.NONE:
            findings.append(
                DefectFinding(
                    label=status.position.label,
                    severity=severity,
                    reason=_reason_for_status(status, severity),
                )
            )

    overall = _lookup_verdict(worst, rules)

    if low_confidence_flagged:
        capped = VerdictLabel(rules["low_confidence_verdict_cap"])
        if _verdict_rank(overall) < _verdict_rank(capped):
            overall = capped
            findings.append(
                DefectFinding(
                    label="_confidence",
                    severity=Severity.MINOR,
                    reason=(
                        "One or more detections fell below the minimum "
                        f"confidence threshold ({min_confidence:.2f}); verdict "
                        "capped rather than asserting 'safe' on weak evidence."
                    ),
                )
            )

    return Verdict(
        overall=overall,
        worst_severity=worst,
        findings=findings,
        low_confidence_flagged=low_confidence_flagged,
    )


def _lookup_verdict(worst: Severity, rules: dict[str, Any]) -> VerdictLabel:
    for row in rules["verdict_thresholds"]:
        if row["if_worst_severity"] == worst.value:
            return VerdictLabel(row["verdict"])
    raise RulesError(
        f"verdict_thresholds in rules.yaml has no entry for severity "
        f"'{worst.value}' — every Severity value must map to a verdict"
    )


def _verdict_rank(v: VerdictLabel) -> int:
    return {"safe": 0, "needs_inspection": 1, "unsafe": 2}[v.value]
