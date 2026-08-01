"""Shared data structures for the fastener inspection pipeline.

Kept deliberately flat and dependency-light (dataclasses only) so every stage
of the pipeline — detection, template matching, verdict rules, explanation —
can be unit-tested against hand-built instances of these types without
needing a model, an image, or a network call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DetectionClass(str, Enum):
    """Per-fastener classes the vision model predicts."""

    BOLT_OK = "bolt_ok"
    BOLT_CORRODED = "bolt_corroded"
    BOLT_MISALIGNED = "bolt_misaligned"
    BOLT_DAMAGED = "bolt_damaged"


class Severity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"none": 0, "minor": 1, "major": 2, "critical": 3}[self.value]


class VerdictLabel(str, Enum):
    SAFE = "safe"
    NEEDS_INSPECTION = "needs_inspection"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space box, (x, y) = top-left corner."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass(frozen=True)
class Detection:
    """One fastener detected in an image by the vision model."""

    detection_class: DetectionClass
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class ExpectedPosition:
    """One fastener position from an assembly template."""

    label: str
    x: float
    y: float
    tolerance_px: float = 40.0


@dataclass(frozen=True)
class FastenerStatus:
    """Result of matching one expected template position against detections."""

    position: ExpectedPosition
    detection: Detection | None  # None means "missing" — no matching detection found
    distance_px: float | None = None

    @property
    def is_missing(self) -> bool:
        return self.detection is None


@dataclass(frozen=True)
class DefectFinding:
    """One reason contributing to the overall verdict."""

    label: str
    severity: Severity
    reason: str


@dataclass(frozen=True)
class Verdict:
    overall: VerdictLabel
    worst_severity: Severity
    findings: list[DefectFinding] = field(default_factory=list)
    low_confidence_flagged: bool = False
