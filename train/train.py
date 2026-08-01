#!/usr/bin/env python3
"""Fine-tune YOLOv8n on the merged dataset produced by prepare_dataset.py.

Writes the trained weights to models/best.pt (picked up automatically by
src/detect.py) and prints final validation metrics.

Usage:
    python train/train.py --epochs 50
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-yaml",
        default=str(PROJECT_ROOT / "data" / "merged" / "data.yaml"),
        help="Path to the merged dataset's data.yaml (from prepare_dataset.py)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--base-model", default="yolov8n.pt", help="Starting checkpoint")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        print(
            f"{data_yaml} not found — run train/prepare_dataset.py first "
            "against a real labeled dataset (see train/README.md)."
        )
        return 1

    from ultralytics import YOLO  # imported here: heavy dependency, only needed for training

    model = YOLO(args.base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(PROJECT_ROOT / "train" / "runs"),
        name="fastener_detector",
    )

    # Ultralytics writes the best checkpoint under runs/<name>/weights/best.pt;
    # copy it to the fixed path src/detect.py expects.
    trained_best = Path(results.save_dir) / "weights" / "best.pt"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    dest = models_dir / "best.pt"
    shutil.copy2(trained_best, dest)
    print(f"\nTrained model copied to {dest}")

    # Print final validation metrics for the record — these belong in the
    # README's evaluation table.
    metrics = model.val(data=str(data_yaml))
    print(f"\nValidation mAP50: {metrics.box.map50:.3f}")
    print(f"Validation mAP50-95: {metrics.box.map:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
