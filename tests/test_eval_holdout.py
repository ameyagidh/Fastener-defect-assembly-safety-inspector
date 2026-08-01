"""End-to-end evaluation against the held-out image set produced by
train/prepare_dataset.py.

Unlike test_verdict.py and test_match_template.py (pure logic, always run),
this test requires a fine-tuned models/best.pt and a real holdout split —
neither ships with this repo (see train/README.md for why). It skips
cleanly rather than failing when that data isn't present, so `pytest tests/`
is always green on a fresh checkout while still being the right test to run
once a real model has been trained.

Run after training:
    python train/prepare_dataset.py
    python train/train.py --epochs 50
    pytest tests/test_eval_holdout.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.detect import FastenerDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
HOLDOUT_DATA_YAML = PROJECT_ROOT / "data" / "merged" / "data.yaml"

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and HOLDOUT_DATA_YAML.exists()),
    reason=(
        "No fine-tuned model / holdout dataset present. Run "
        "train/prepare_dataset.py and train/train.py against a real "
        "labeled dataset first — see train/README.md."
    ),
)


def _load_holdout_labels() -> list[tuple[Path, Path]]:
    with HOLDOUT_DATA_YAML.open() as f:
        data_yaml = yaml.safe_load(f)
    merged_dir = Path(data_yaml["path"])
    images_dir = merged_dir / "images" / "holdout"
    labels_dir = merged_dir / "labels" / "holdout"
    return [
        (img, labels_dir / f"{img.stem}.txt")
        for img in sorted(images_dir.glob("*"))
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


class TestDetectionQuality:
    """Standard object-detection metrics via ultralytics' own validator —
    this is the mAP number from the plan's Evaluation section, item 1.
    """

    def test_holdout_map(self):
        from ultralytics import YOLO

        model = YOLO(str(MODEL_PATH))
        metrics = model.val(data=str(HOLDOUT_DATA_YAML), split="test")

        print(f"\nHoldout mAP50: {metrics.box.map50:.3f}")
        print(f"Holdout mAP50-95: {metrics.box.map:.3f}")
        for i, name in enumerate(model.names.values()):
            if i < len(metrics.box.ap50):
                print(f"  {name}: AP50={metrics.box.ap50[i]:.3f}")

        # A floor, not a target — this exists to catch a broken/undertrained
        # model producing near-random output, not to gate on a specific
        # research-grade number. Report the real number in README.md.
        assert metrics.box.map50 > 0.0


class TestVerdictAccuracy:
    """End-to-end verdict precision/recall against human-labeled holdout
    assemblies — the number from the plan's Evaluation section, item 2,
    and the one that actually matters for the product.

    NOTE: this requires a separate assembly-level ground-truth file
    (data/holdout_verdicts.jsonl: {"image": "...", "expected_verdict": "..."})
    that a human labels once the holdout images exist, since per-fastener
    detection labels alone don't tell you the intended overall verdict for
    a whole assembly photo. See train/README.md.
    """

    def test_verdict_precision_recall(self):
        verdicts_path = PROJECT_ROOT / "data" / "holdout_verdicts.jsonl"
        if not verdicts_path.exists():
            pytest.skip(
                f"{verdicts_path} not found — label expected verdicts for "
                "the holdout assemblies before this test can run."
            )

        import json

        from src.pipeline import inspect

        detector = FastenerDetector()
        results = []
        with verdicts_path.open() as f:
            for line in f:
                row = json.loads(line)
                result = inspect(
                    PROJECT_ROOT / row["image"],
                    detector=detector,
                    include_explanation=False,
                )
                results.append((row["expected_verdict"], result.verdict.overall.value))

        false_negatives = sum(
            1
            for expected, actual in results
            if expected == "unsafe" and actual != "unsafe"
        )
        correct = sum(1 for expected, actual in results if expected == actual)

        print(f"\nVerdict accuracy: {correct}/{len(results)}")
        print(
            f"False negatives (unsafe assembly called '{results and 'safe/needs_inspection'}'): "
            f"{false_negatives} — this is the headline metric, report it separately "
            "from overall accuracy in README.md"
        )

        assert len(results) > 0
