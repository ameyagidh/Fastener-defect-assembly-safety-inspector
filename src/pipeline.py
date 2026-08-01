"""End-to-end pipeline: photo -> detections -> matched statuses -> verdict
-> (optional) plain-language explanation.

This module is the single entry point both the CLI usage below and app.py
(the Streamlit GUI) call into, so the demo and the command line can never
drift out of sync with each other.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.detect import FastenerDetector
from src.explain import explain_or_fallback
from src.match_template import expected_positions_from_template, load_template, match
from src.models import Detection, FastenerStatus, Verdict
from src.verdict import evaluate, load_rules

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "bracket_4bolt.json"


@dataclass
class InspectionResult:
    """Full output of one pipeline run — everything the GUI needs to render."""

    image_path: Path
    detections: list[Detection]
    statuses: list[FastenerStatus]
    verdict: Verdict
    explanation: str
    model_is_fine_tuned: bool


def inspect(
    image_path: str | Path,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    detector: FastenerDetector | None = None,
    include_explanation: bool = True,
) -> InspectionResult:
    """Run the full pipeline on one image and return an InspectionResult.

    `detector` can be injected (e.g. a fake in tests) so this function does
    not require a real model to be exercised in unit tests of the wiring —
    only the eval/demo paths need an actual FastenerDetector.
    """
    image_path = Path(image_path)
    template_path = Path(template_path)

    if detector is None:
        detector = FastenerDetector()

    with Image.open(image_path) as img:
        width, height = img.size

    template = load_template(template_path)
    positions = expected_positions_from_template(template, width, height)

    detections = detector.detect(image_path)
    statuses = match(positions, detections)

    rules = load_rules()
    verdict = evaluate(statuses, rules)

    explanation = explain_or_fallback(verdict) if include_explanation else ""

    return InspectionResult(
        image_path=image_path,
        detections=detections,
        statuses=statuses,
        verdict=verdict,
        explanation=explanation,
        model_is_fine_tuned=detector.is_fine_tuned,
    )


def _print_result(result: InspectionResult) -> None:
    print(f"Image: {result.image_path}")
    print(
        f"Model fine-tuned on fastener classes: "
        f"{'yes' if result.model_is_fine_tuned else 'NO (using stock/demo weights — see README)'}"
    )
    print(f"Detections found: {len(result.detections)}")
    print()
    print("Per-position status:")
    for status in result.statuses:
        if status.is_missing:
            print(f"  [MISSING] {status.position.label}")
        else:
            det = status.detection
            print(
                f"  [{det.detection_class.value:>15}] {status.position.label} "
                f"(confidence={det.confidence:.2f}, distance={status.distance_px:.1f}px)"
            )
    print()
    print(f"VERDICT: {result.verdict.overall.value.upper()} "
          f"(worst severity: {result.verdict.worst_severity.value})")
    if result.verdict.findings:
        print("Findings:")
        for f in result.verdict.findings:
            print(f"  - [{f.severity.value}] {f.reason}")
    print()
    print("Explanation:")
    print(result.explanation)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the fastener safety inspection pipeline on one image."
    )
    parser.add_argument("image", help="Path to the assembly photo to inspect")
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE_PATH),
        help="Path to the assembly template JSON (default: bracket_4bolt.json)",
    )
    parser.add_argument(
        "--no-explanation",
        action="store_true",
        help="Skip the LLM explanation call (structured verdict only)",
    )
    args = parser.parse_args()

    result = inspect(
        args.image,
        template_path=args.template,
        include_explanation=not args.no_explanation,
    )
    _print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
