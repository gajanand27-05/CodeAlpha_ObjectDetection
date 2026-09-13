"""Real-time object detection and tracking.

    python detect_track.py                                        webcam
    python detect_track.py --source videos/people-detection.mp4   video file
    python detect_track.py --source 0 --save output/demo.mp4      record the output
    python detect_track.py --source clip.mp4 --no-show            headless, no window

Press q or Esc in the window to stop early.

The loop is: read a frame, detect objects in it, hand those detections to the
tracker, draw what the tracker reports. The detector finds objects but has no
memory between frames; the tracker is what turns a sequence of unrelated
detections into an object with an identity that persists.
"""

from __future__ import annotations

import argparse
import colorsys
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from detector import DEFAULT_CLASSES, DEFAULT_WEIGHTS, Detector
from sort_tracker import Sort

FONT = cv2.FONT_HERSHEY_SIMPLEX


def colour_for_id(track_id: int) -> tuple[int, int, int]:
    """A stable colour per tracking ID.

    Derived from the ID rather than chosen at random, so a track keeps the same
    colour for its whole life and the same clip always renders identically.
    The golden-ratio step spreads consecutive IDs far apart on the hue circle,
    which keeps neighbouring tracks visually distinct instead of three shades
    of the same green.
    """
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)  # OpenCV wants BGR


def draw_track(frame: np.ndarray, label: str, bbox: np.ndarray, colour,
               predicted: bool = False) -> None:
    """Draw one box with its label sitting above it.

    A coasted box, one whose position came from the Kalman filter rather than
    from a detection this frame, is drawn thinner and dimmer. The distinction
    is worth showing: a solid box means the detector saw the object, a faint
    one means the tracker believes it is still there.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))

    if predicted:
        colour = tuple(int(c * 0.55) for c in colour)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1 if predicted else 2)

    (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
    box_w, box_h = tw + 8, th + baseline + 4

    # Flip the label inside the box when the object is at the top of the frame,
    # otherwise it would be drawn off-screen and be invisible.
    top = y1 - box_h
    if top < 0:
        top = y1 + 2

    # Pull the label left when the object is near the right edge. Without this
    # the label of anything entering from the right is cut off by the frame
    # boundary, which is exactly when you most want to read its ID.
    left = max(0, min(x1, w - box_w))

    cv2.rectangle(frame, (left, top), (left + box_w, top + box_h), colour, -1)
    cv2.putText(frame, label, (left + 4, top + th + 2), FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    """Draw the status lines in the top-left corner on a dark panel."""
    pad, line_h = 8, 20
    width = max(cv2.getTextSize(t, FONT, 0.5, 1)[0][0] for t in lines) + pad * 2
    height = line_h * len(lines) + pad

    panel = frame[0:height, 0:width]
    # Darken rather than fill, so the footage stays visible underneath.
    frame[0:height, 0:width] = (panel * 0.35).astype(np.uint8)

    for i, text in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + line_h * i + 10), FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)


def open_source(source: str) -> tuple[cv2.VideoCapture, bool]:
    """Open a webcam index or a video file path."""
    if source.isdigit():
        # CAP_DSHOW avoids a multi-second startup delay opening webcams on
        # Windows through the default backend.
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW if sys.platform == "win32" else 0)
        return cap, True
    if not Path(source).exists():
        raise FileNotFoundError(f"No such video file: {source}")
    return cv2.VideoCapture(source), False


def main() -> int:
    p = argparse.ArgumentParser(description="Detect and track objects in a video or webcam feed.")
    p.add_argument("--source", default="0", help="webcam index (0) or path to a video file")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO weights to use")
    p.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    p.add_argument("--device", default=None, help="'cpu', '0' for the first GPU, or leave unset")
    p.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES,
                   help="class names to track, or 'all'")
    p.add_argument("--max-age", type=int, default=30,
                   help="frames a track survives unmatched before deletion")
    p.add_argument("--min-hits", type=int, default=3,
                   help="consecutive detections before a track is shown")
    p.add_argument("--iou", type=float, default=0.3, help="IoU threshold for association")
    p.add_argument("--coast", type=int, default=3,
                   help="frames a track keeps being drawn from prediction after a missed "
                        "detection (0 disables)")
    p.add_argument("--save", default=None, help="write the annotated video to this path")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    p.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    args = p.parse_args()

    classes = None if args.classes == ["all"] else args.classes

    # Mistyping a class name or a file path is a user error, not a crash. A
    # traceback buries the useful sentence under a stack that the person who
    # made the typo has no use for.
    try:
        print(f"Loading {args.weights} ...")
        detector = Detector(args.weights, conf=args.conf, classes=classes,
                            imgsz=args.imgsz, device=args.device)
        tracker = Sort(max_age=args.max_age, min_hits=args.min_hits,
                       iou_threshold=args.iou, coast=args.coast)
        cap, is_webcam = open_source(args.source)
    except (ValueError, FileNotFoundError) as exc:
        print(f"\n{exc}")
        return 1
    if not cap.isOpened():
        print(f"Could not open source: {args.source}")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Webcams commonly report -1 here rather than a real rate. `or 30.0` does
    # not catch that, since -1 is truthy, and the bogus value then reaches
    # VideoWriter and produces a file with an invalid frame rate.
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0 or not math.isfinite(src_fps):
        src_fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    print(f"Source: {args.source} ({width}x{height} at {src_fps:.1f} fps"
          + (f", {total} frames)" if total else ")"))
    print("Press q to stop." if not args.no_show else "Running headless.")

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (width, height))
        if not writer.isOpened():
            print(f"Could not open {args.save} for writing.")
            return 1

    frames = 0
    seen_ids: set[int] = set()
    class_of_id: dict[int, str] = {}
    recent = []  # rolling per-frame durations, for a stable FPS readout
    started = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            recent.append(time.perf_counter() - t0)
            recent = recent[-30:]

            frames += 1
            fps = len(recent) / max(sum(recent), 1e-9)

            # Counts are updated before anything is drawn, so the HUD reports
            # this frame rather than the previous one.
            for t in tracks:
                seen_ids.add(t["id"])
                class_of_id[t["id"]] = detector.class_name(t["cls_id"])

            # The HUD is drawn first so that boxes and labels land on top of it.
            # With the order reversed, the panel covered the label of any object
            # in the top-left corner, hiding the ID of a real tracked object
            # behind a status readout.
            draw_hud(frame, [
                f"frame {frames}" + (f"/{total}" if total else ""),
                f"{fps:.1f} fps",
                f"tracking {len(tracks)}   unique so far {len(seen_ids)}",
            ])

            for t in tracks:
                label = f"{class_of_id[t['id']]} {t['id']}"
                draw_track(frame, label, t["bbox"], colour_for_id(t["id"]),
                           predicted=t.get("predicted", False))

            if writer is not None:
                writer.write(frame)

            if not args.no_show:
                cv2.imshow("Detection and tracking", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("Stopped by user.")
                    break

            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print(f"\nProcessed {frames} frames in {elapsed:.1f}s "
          f"({frames / max(elapsed, 1e-9):.1f} fps overall)")
    print(f"Unique objects tracked: {len(seen_ids)}")

    per_class: dict[str, int] = defaultdict(int)
    for name in class_of_id.values():
        per_class[name] += 1
    for name, count in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {count}")

    if args.save:
        print(f"Saved to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
