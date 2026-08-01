#!/usr/bin/env python3
"""Merge YOLO-format source datasets under data/raw/ and data/self_collected/
into a single train/val/holdout split at data/merged/.

See train/README.md for the full workflow this script is one step of, and
for why it does not fetch data itself.

Expected input layout (per source dataset, already in YOLO format — this is
what a Roboflow YOLOv8 export produces):

    data/raw/<dataset_name>/
        images/*.jpg
        labels/*.txt          # one .txt per image, YOLO box format
        data.yaml             # must declare the same 4 class names used here,
                               # in the same order — see CLASS_NAMES below

    data/self_collected/
        images/*.jpg
        labels/*.txt

Output:

    data/merged/
        images/{train,val,holdout}/*.jpg
        labels/{train,val,holdout}/*.txt
        data.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["bolt_ok", "bolt_corroded", "bolt_misaligned", "bolt_damaged"]

# Fixed split ratios. Holdout is never used in training and is what
# tests/test_eval_holdout.py evaluates against.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# Remainder (0.15) goes to holdout.

RANDOM_SEED = 42  # fixed so the split is reproducible across runs


def discover_source_datasets(data_dir: Path) -> list[Path]:
    sources = []
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        sources.extend(sorted(p for p in raw_dir.iterdir() if p.is_dir()))
    self_collected = data_dir / "self_collected"
    if (self_collected / "images").exists():
        sources.append(self_collected)
    return sources


def collect_image_label_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    pairs = []
    if not images_dir.exists():
        return pairs
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"  skipping {image_path.name}: no matching label file")
            continue
        pairs.append((image_path, label_path))
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]], seed: int
) -> dict[str, list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "holdout": shuffled[n_train + n_val :],
    }


def write_split(split_name: str, pairs: list[tuple[Path, Path]], merged_dir: Path) -> None:
    images_out = merged_dir / "images" / split_name
    labels_out = merged_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in pairs:
        # Prefix filenames with the source dataset name to avoid collisions
        # when multiple sources happen to share a filename convention.
        prefix = image_path.parent.parent.name
        dest_name = f"{prefix}_{image_path.name}"
        shutil.copy2(image_path, images_out / dest_name)
        shutil.copy2(label_path, labels_out / f"{prefix}_{label_path.name}")


def write_data_yaml(merged_dir: Path) -> None:
    data_yaml = {
        "path": str(merged_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/holdout",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with (merged_dir / "data.yaml").open("w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    merged_dir = data_dir / "merged"

    sources = discover_source_datasets(data_dir)
    if not sources:
        print(
            "No source datasets found under data/raw/ or data/self_collected/.\n"
            "See train/README.md for how to source and lay out training data "
            "before running this script."
        )
        return 1

    all_pairs: list[tuple[Path, Path]] = []
    for source in sources:
        pairs = collect_image_label_pairs(source)
        print(f"{source.name}: {len(pairs)} labeled image(s)")
        all_pairs.extend(pairs)

    if not all_pairs:
        print("No labeled image/label pairs found across any source dataset.")
        return 1

    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    splits = split_pairs(all_pairs, args.seed)
    for split_name, pairs in splits.items():
        write_split(split_name, pairs, merged_dir)
        print(f"{split_name}: {len(pairs)} images")

    write_data_yaml(merged_dir)
    print(f"\nMerged dataset written to {merged_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
