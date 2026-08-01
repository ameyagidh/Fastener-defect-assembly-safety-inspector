"""Match a list of model detections against a supplied assembly template.

A pure object detector can only report what it sees ("3 bolts, all fine") —
it cannot know a 4th bolt is *missing* unless something tells it 4 were
expected. That "something" is a template: a small, human-authored JSON file
per part type describing where fasteners should be. This module diffs
detections against that template so a truly absent fastener is reported as
"missing" rather than silently disappearing from the result.

No model, no image I/O — this operates purely on Detection/ExpectedPosition
data structures, so it is directly unit-testable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.models import Detection, ExpectedPosition, FastenerStatus


class TemplateError(ValueError):
    """Raised when a template file is missing a required field or is malformed."""


def load_template(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise TemplateError(f"template file not found: {path}")

    with path.open() as f:
        template = json.load(f)

    required = {"name", "reference_frame", "expected_positions"}
    missing = required - template.keys()
    if missing:
        raise TemplateError(f"template {path} missing required keys: {sorted(missing)}")

    if not template["expected_positions"]:
        raise TemplateError(f"template {path} has an empty expected_positions list")

    return template


def expected_positions_from_template(
    template: dict[str, Any],
    image_width: float,
    image_height: float,
) -> list[ExpectedPosition]:
    """Rescale a template's normalized positions into actual image pixel space."""
    ref = template["reference_frame"]
    scale_x = image_width / ref["width"]
    scale_y = image_height / ref["height"]

    positions = []
    for entry in template["expected_positions"]:
        positions.append(
            ExpectedPosition(
                label=entry["label"],
                x=entry["x"] * scale_x,
                y=entry["y"] * scale_y,
                # Tolerance scales with the smaller axis so it doesn't blow up
                # on extreme aspect ratios.
                tolerance_px=entry.get("tolerance_px", 40.0) * min(scale_x, scale_y),
            )
        )
    return positions


def _distance(pos: ExpectedPosition, det: Detection) -> float:
    cx, cy = det.bbox.center
    return math.hypot(cx - pos.x, cy - pos.y)


def match(
    positions: list[ExpectedPosition],
    detections: list[Detection],
) -> list[FastenerStatus]:
    """Greedy nearest-match: each expected position claims its closest
    unclaimed detection within tolerance; unclaimed positions are "missing".

    Greedy-by-distance (rather than a full bipartite optimal assignment) is a
    deliberate simplicity choice: with 4-8 well-separated fastener positions
    per assembly, a greedy nearest-match and an optimal assignment agree in
    practice, and greedy is trivial to reason about and unit-test. If this
    is ever extended to templates with tightly-clustered fasteners, swap in
    `scipy.optimize.linear_sum_assignment` here — the function signature
    would not need to change.
    """
    unclaimed = list(detections)
    statuses: list[FastenerStatus] = []

    # Process positions in order of their closest available detection first,
    # so two positions competing for the same detection resolve to whichever
    # is actually closer, not whichever appears first in the list.
    remaining_positions = list(positions)
    while remaining_positions:
        best_pos = None
        best_det = None
        best_dist = math.inf

        for pos in remaining_positions:
            for det in unclaimed:
                d = _distance(pos, det)
                if d <= pos.tolerance_px and d < best_dist:
                    best_pos, best_det, best_dist = pos, det, d

        if best_pos is None:
            # No remaining position has any detection within tolerance —
            # everything left is missing.
            for pos in remaining_positions:
                statuses.append(FastenerStatus(position=pos, detection=None))
            break

        statuses.append(
            FastenerStatus(position=best_pos, detection=best_det, distance_px=best_dist)
        )
        remaining_positions.remove(best_pos)
        unclaimed.remove(best_det)

    return statuses
