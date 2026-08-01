# Training the fastener detector

This directory fine-tunes a YOLOv8n model on labeled fastener images to
produce `models/best.pt`, which `src/detect.py` will pick up automatically
once it exists.

## Honest status

**This repository does not ship a fine-tuned model.** Producing one requires
a labeled dataset of bolt/fastener images across four classes
(`bolt_ok`, `bolt_corroded`, `bolt_misaligned`, `bolt_damaged`) — data this
environment does not have and cannot ethically fabricate (a "corroded bolt"
detector trained on invented images would report a false confidence it
hasn't earned). The scripts below are complete, runnable, and documented so
that plugging in real data is the *only* remaining step; running them without
real data will not produce a usable model.

Until `models/best.pt` exists, `src/detect.py` falls back to a stock
COCO-pretrained `yolov8n.pt` — this keeps the full pipeline runnable
end-to-end (see `src/pipeline.py`), but it does **not** detect fastener
classes and every detection it returns is deliberately forced to
near-zero confidence by `FastenerDetector.detect()` so the verdict engine's
`low_confidence_verdict_cap` rule kicks in rather than asserting a false
"safe". Check `InspectionResult.model_is_fine_tuned` (surfaced in the CLI
output and the GUI) before trusting a verdict.

## Steps to produce a real model

1. **Source data.** Search Roboflow Universe for bolt/fastener/corrosion
   detection datasets (e.g. search terms: "bolt detection", "nut bolt
   defect", "rust corrosion detection"). Check each dataset's license before
   use — prefer CC BY or public domain. Download in YOLO format.
2. **Self-collect the gap classes.** Public data is unlikely to cover
   `bolt_misaligned` well. Take 20-30 of your own photos of a bolted bracket
   in good and bad states, and label them with Roboflow's free annotation
   tool (bounding box + one of the four classes).
3. **`python train/prepare_dataset.py`** — merges every source under
   `data/raw/<dataset_name>/` and `data/self_collected/` into a single
   YOLO-format dataset at `data/merged/`, with a fixed train/val/holdout
   split (holdout is never touched by training — see `tests/test_eval_holdout.py`).
4. **`python train/train.py --epochs 50`** — fine-tunes `yolov8n.pt` on
   `data/merged/`, writes `models/best.pt`, and prints final validation mAP.
5. Re-run `pytest tests/test_eval_holdout.py` — this is the test that
   actually validates the trained model against the held-out set (mAP,
   per-class precision/recall, and end-to-end verdict accuracy — see the
   main README's Evaluation section for what to report).

## Why the scripts are separate from a Roboflow API call

`prepare_dataset.py` intentionally does not fetch datasets over the network
itself (no hardcoded Roboflow API key, no silent download of an unspecified
dataset into a demo). You choose and vet the source datasets, download them
yourself, and the script's job is purely the local merge/reformat/split step
— that boundary keeps "which data trained this model" auditable from the
README rather than buried in a script.
