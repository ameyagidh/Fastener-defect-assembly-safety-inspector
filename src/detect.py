"""Vision-model wrapper: runs a YOLOv8 model over an image and returns a list
of Detection objects.

This module is deliberately thin. It does not decide anything — it converts
between the Ultralytics result format and this project's Detection/BoundingBox
dataclasses so the rest of the pipeline (match_template, verdict) never has
to import ultralytics or know about its API.

Model status: this repo ships WITHOUT a fine-tuned models/best.pt, because a
production-quality fastener-defect detector requires a labeled dataset this
environment does not have (see train/README.md and the project README's
"Current status & limitations" section for the honest account of this).
FastenerDetector falls back to a stock COCO-pretrained YOLOv8n checkpoint,
which does NOT recognize fastener-specific classes — it is wired in so the
full pipeline (detect -> match -> verdict -> explain) is runnable end-to-end
today, with a clear, loud signal about which stage is a stub.
"""

from __future__ import annotations

from pathlib import Path

from src.models import BoundingBox, Detection, DetectionClass

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"

# Ultralytics class-index -> our DetectionClass, for a model trained on the
# 4-class scheme this project defines (see train/prepare_dataset.py). This is
# the mapping a fine-tuned models/best.pt is expected to produce; it has
# nothing to do with stock COCO classes.
CLASS_INDEX_MAP: dict[int, DetectionClass] = {
    0: DetectionClass.BOLT_OK,
    1: DetectionClass.BOLT_CORRODED,
    2: DetectionClass.BOLT_MISALIGNED,
    3: DetectionClass.BOLT_DAMAGED,
}


class ModelNotFineTunedError(RuntimeError):
    """Raised by strict callers that require a real fastener-trained model."""


class FastenerDetector:
    """Loads a YOLOv8 model once and exposes a simple `.detect(image_path)` call."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, confidence_threshold: float = 0.25):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.is_fine_tuned = model_path.exists()
        self._model = None  # lazy-loaded; ultralytics import is slow and pulls in torch

    def _load(self):
        if self._model is not None:
            return self._model

        from ultralytics import YOLO  # local import: keep this heavy dependency optional

        if self.is_fine_tuned:
            self._model = YOLO(str(self.model_path))
        else:
            # Stock pretrained fallback so the pipeline is runnable without a
            # trained checkpoint. See the module docstring: this does not
            # detect fastener-specific classes.
            self._model = YOLO("yolov8n.pt")
        return self._model

    def detect(self, image_path: str | Path) -> list[Detection]:
        """Run inference and return a list of Detection objects.

        When `is_fine_tuned` is False, detections are returned using the
        best-effort CLASS_INDEX_MAP but the caller should treat class labels
        as unreliable — see ModelNotFineTunedError for a strict alternative.
        """
        model = self._load()
        results = model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_idx = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())

                detection_class = CLASS_INDEX_MAP.get(cls_idx)
                if detection_class is None:
                    # Stock-model class index with no fastener mapping (e.g.
                    # COCO fallback predicting "person" or "car"). Keep the
                    # detection visible rather than silently dropping it, but
                    # mark it clearly so verdict logic doesn't misread it.
                    detection_class = DetectionClass.BOLT_OK
                    confidence = min(confidence, 0.01)  # forces the low-confidence guard

                detections.append(
                    Detection(
                        detection_class=detection_class,
                        confidence=confidence,
                        bbox=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                    )
                )
        return detections

    def require_fine_tuned(self) -> None:
        if not self.is_fine_tuned:
            raise ModelNotFineTunedError(
                f"No fine-tuned model found at {self.model_path}. Run "
                "train/prepare_dataset.py and train/train.py against a "
                "labeled fastener dataset before relying on detection "
                "output for a real safety verdict — see README.md "
                "'Current status & limitations'."
            )
