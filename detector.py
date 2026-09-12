"""YOLO object detection for a single frame.

Stage one of the pipeline. This file knows about YOLO and nothing about
tracking; sort_tracker.py knows about tracking and nothing about YOLO. They
meet only at a plain numpy array of [x1, y1, x2, y2, score, class_id] rows,
which is why the tracker can be tested on synthetic boxes with no model loaded
and no video decoded.

The model is pre-trained on COCO and is used as-is. Training a detector from
scratch would need a labelled dataset, a GPU and days of compute to arrive
somewhere worse than the published weights, and the task here is detection and
tracking, not reproducing the detector.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

# The COCO classes worth tracking in the sample footage. Passing this to the
# model rather than filtering afterwards is deliberate: YOLO skips the
# non-max suppression work for classes it has been told to ignore, so it is
# faster as well as tidier.
DEFAULT_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# yolov8n is the smallest of the family. On CPU it is the difference between a
# few frames per second and a slideshow, and on the sample clips its accuracy
# is not the limiting factor. yolov8s or yolov8m are drop-in replacements if
# a GPU is available.
DEFAULT_WEIGHTS = "yolov8n.pt"


class Detector:
    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        conf: float = 0.35,
        classes: list[str] | None = None,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        # Downloads the weights on first use and caches them. They are in
        # .gitignore: they are tens of MB, they are not mine to redistribute,
        # and a fresh clone will fetch them itself.
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

        # Map class names to the integer ids the model actually uses. Doing
        # this by name means the code does not depend on COCO's ordering.
        self.names: dict[int, str] = self.model.names
        name_to_id = {name: idx for idx, name in self.names.items()}

        if classes is None:
            classes = DEFAULT_CLASSES
        unknown = [c for c in classes if c not in name_to_id]
        if unknown:
            raise ValueError(
                f"Unknown class name(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(name_to_id))}"
            )
        self.class_ids = [name_to_id[c] for c in classes] if classes else None

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect objects in one frame.

        Returns an (N, 6) float array of [x1, y1, x2, y2, score, class_id],
        which is exactly what Sort.update expects. An empty (0, 6) array is
        returned when nothing is found, rather than None, so callers never have
        to special-case the empty frame.
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            classes=self.class_ids,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,  # otherwise it prints a line per frame
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = results.boxes.xyxy.cpu().numpy()
        conf = results.boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = results.boxes.cls.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, conf, cls]).astype(np.float32)

    def class_name(self, cls_id: int) -> str:
        return self.names.get(int(cls_id), str(cls_id))
