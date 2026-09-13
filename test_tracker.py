"""Checks the SORT tracker on synthetic motion. Run with:  python test_tracker.py

The tracker is tested on generated box trajectories rather than on video. A
video test would depend on the detector, the weights and the clip all being
correct, so a failure would not say which part broke. Here the input is exact,
so any failure is the tracker's.

What is being tested is the property that actually matters: an object keeps one
ID for as long as it is the same object, and does not inherit somebody else's.
"""

import numpy as np

from sort_tracker import Sort, iou_batch

PASS, FAIL = "PASS", "FAIL"
failures = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if condition else FAIL}  {name}{('  ' + detail) if detail else ''}")
    if not condition:
        failures += 1


def box(cx: float, cy: float, w: float = 40, h: float = 80, score: float = 0.9, cls_id: int = 0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, cls_id]


# --------------------------------------------------------------- IoU geometry

def test_iou() -> None:
    print("\n--- IoU ---")
    a = np.array([[0, 0, 10, 10]], dtype=float)
    check("identical boxes score 1.0", abs(iou_batch(a, a)[0, 0] - 1.0) < 1e-6)

    b = np.array([[20, 20, 30, 30]], dtype=float)
    check("disjoint boxes score 0.0", iou_batch(a, b)[0, 0] == 0.0)

    # Half-overlapping squares: intersection 50, union 150, so 1/3.
    c = np.array([[5, 0, 15, 10]], dtype=float)
    check("half overlap scores 1/3", abs(iou_batch(a, c)[0, 0] - 1 / 3) < 1e-6,
          f"got {iou_batch(a, c)[0, 0]:.4f}")

    check("empty input gives empty matrix",
          iou_batch(np.empty((0, 4)), a).shape == (0, 1))


# ------------------------------------------------------------- ID persistence

def test_single_object_keeps_one_id() -> None:
    print("\n--- one object moving in a straight line ---")
    Sort.reset_ids()
    tracker = Sort(max_age=30, min_hits=3)

    seen_ids, reported_frames = set(), 0
    for frame in range(40):
        out = tracker.update(np.array([box(100 + frame * 5, 200)]))
        if out:
            reported_frames += 1
            seen_ids.update(o["id"] for o in out)

    check("exactly one ID issued", len(seen_ids) == 1, f"ids={sorted(seen_ids)}")
    check("reported on every frame", reported_frames == 40, f"{reported_frames}/40")


def test_two_objects_do_not_swap() -> None:
    print("\n--- two objects passing each other ---")
    Sort.reset_ids()
    tracker = Sort(max_age=30, min_hits=3)

    # One moves right, the other left, on separate rows so they pass without
    # fully overlapping. Their paths cross in x around frame 15.
    id_left_to_right, id_right_to_left = set(), set()
    for frame in range(30):
        a = box(50 + frame * 8, 150)
        b = box(290 - frame * 8, 260)
        out = tracker.update(np.array([a, b]))
        for o in out:
            cx = (o["bbox"][0] + o["bbox"][2]) / 2
            cy = (o["bbox"][1] + o["bbox"][3]) / 2
            (id_left_to_right if cy < 200 else id_right_to_left).add(o["id"])

    check("upper object holds a single ID", len(id_left_to_right) == 1, f"ids={sorted(id_left_to_right)}")
    check("lower object holds a single ID", len(id_right_to_left) == 1, f"ids={sorted(id_right_to_left)}")
    check("the two objects have different IDs", id_left_to_right != id_right_to_left)


def test_id_survives_brief_occlusion() -> None:
    print("\n--- object hidden for 5 frames, max_age 30 ---")
    Sort.reset_ids()
    tracker = Sort(max_age=30, min_hits=3)

    before = after = None
    for frame in range(30):
        occluded = 12 <= frame < 17
        dets = np.empty((0, 6)) if occluded else np.array([box(60 + frame * 6, 200)])
        out = tracker.update(dets)
        if out and frame == 11:
            before = out[0]["id"]
        if out and frame == 20:
            after = out[0]["id"]

    check("an ID was held before the gap", before is not None)
    check("an ID was held after the gap", after is not None)
    check("it is the same ID", before == after, f"{before} then {after}")


def test_track_retired_after_max_age() -> None:
    print("\n--- object leaves for good, max_age 5 ---")
    Sort.reset_ids()
    tracker = Sort(max_age=5, min_hits=3)

    for frame in range(15):
        tracker.update(np.array([box(100 + frame * 4, 200)]))
    for _ in range(10):
        out = tracker.update(np.empty((0, 6)))

    check("nothing is still reported", out == [], f"got {out}")
    check("the track was deleted, not just hidden", len(tracker.tracks) == 0,
          f"{len(tracker.tracks)} left")


def test_single_frame_false_positive_suppressed() -> None:
    print("\n--- one-frame detector blip ---")
    Sort.reset_ids()
    tracker = Sort(max_age=30, min_hits=3)

    # Run a real object long enough that frame_count is past the startup
    # exception, then inject a blip that appears for exactly one frame.
    for frame in range(10):
        tracker.update(np.array([box(100, 200)]))

    out = tracker.update(np.array([box(100, 200), box(400, 400)]))
    reported = [(o["bbox"][0] + o["bbox"][2]) / 2 for o in out]
    check("the blip is not reported", all(abs(cx - 400) > 50 for cx in reported),
          f"centres={[round(c) for c in reported]}")

    for _ in range(5):
        out = tracker.update(np.array([box(100, 200)]))
    check("the real object is still reported", len(out) == 1)


def test_confirmed_track_survives_flickering_detector() -> None:
    print("\n--- detector drops single frames on an established track ---")
    Sort.reset_ids()
    tracker = Sort(max_age=30, min_hits=3)

    # Establish the track properly first.
    for frame in range(10):
        tracker.update(np.array([box(100 + frame * 5, 200)]))

    # Now alternate: detected, missed, detected, missed. This is what a real
    # detector does when confidence sits near the threshold.
    reported, ids = 0, set()
    detected_frames = 0
    for frame in range(10, 30):
        missing = frame % 2 == 1
        dets = np.empty((0, 6)) if missing else np.array([box(100 + frame * 5, 200)])
        out = tracker.update(dets)
        if not missing:
            detected_frames += 1
            if out:
                reported += 1
                ids.update(o["id"] for o in out)

    # The regression this guards: with `hit_streak >= min_hits` as the display
    # rule, a track never rebuilds a 3-frame streak under this pattern and is
    # reported on none of these frames despite being alive and detected.
    check("shown on every frame it was detected", reported == detected_frames,
          f"{reported}/{detected_frames}")
    check("and kept one ID throughout", len(ids) == 1, f"ids={sorted(ids)}")


def test_class_label_survives_a_misclassified_frame() -> None:
    print("\n--- detector misclassifies a single frame ---")
    Sort.reset_ids()
    tracker = Sort(max_age=30, min_hits=3)

    # 2 is "car", 5 is "bus" in COCO. In the sample footage YOLO calls the car
    # a bus on exactly one frame out of hundreds.
    labels = []
    for frame in range(20):
        cls = 5 if frame == 12 else 2
        out = tracker.update(np.array([box(100 + frame * 5, 200, cls_id=cls)]))
        if out:
            labels.append(out[0]["cls_id"])

    check("never reports the one-frame misclassification",
          all(c == 2 for c in labels), f"saw classes {sorted(set(labels))}")

    # And the vote must still be able to change if the evidence really changes.
    Sort.reset_ids()
    t2 = Sort(max_age=30, min_hits=3)
    for frame in range(3):
        t2.update(np.array([box(100, 200, cls_id=2)]))
    for frame in range(3, 20):
        out = t2.update(np.array([box(100 + frame * 5, 200, cls_id=5)]))
    check("a sustained class change is eventually adopted", out[0]["cls_id"] == 5,
          f"got {out[0]['cls_id']}")


def test_coasting_marks_predicted_boxes() -> None:
    print("\n--- coasting through a detection gap ---")
    Sort.reset_ids()
    tracker = Sort(max_age=30, min_hits=3, coast=3)

    for frame in range(10):
        out = tracker.update(np.array([box(100 + frame * 5, 200)]))
    check("a matched box is not flagged predicted", out[0]["predicted"] is False)

    # Three missed frames: still reported, flagged as predictions.
    flags, positions = [], []
    for _ in range(3):
        out = tracker.update(np.empty((0, 6)))
        flags.append(bool(out) and out[0]["predicted"])
        if out:
            positions.append((out[0]["bbox"][0] + out[0]["bbox"][2]) / 2)

    check("still reported during the gap", all(flags), f"{flags}")
    check("and flagged as predicted", flags == [True, True, True])
    check("the prediction keeps moving", len(positions) == 3 and positions[2] > positions[0],
          f"{[round(p) for p in positions]}")

    # Past the coast window it goes quiet, though the track is still alive.
    out = tracker.update(np.empty((0, 6)))
    check("silent once past the coast window", out == [], f"got {out}")
    check("but the track still exists", len(tracker.tracks) == 1)

    Sort.reset_ids()
    strict = Sort(max_age=30, min_hits=3, coast=0)
    for frame in range(10):
        strict.update(np.array([box(100 + frame * 5, 200)]))
    check("coast=0 reports nothing on a missed frame",
          strict.update(np.empty((0, 6))) == [])


def test_empty_input_is_safe() -> None:
    print("\n--- degenerate input ---")
    Sort.reset_ids()
    tracker = Sort()

    check("empty array does not crash", tracker.update(np.empty((0, 6))) == [])
    check("None does not crash", tracker.update(None) == [])

    # A zero-area box is malformed but a detector can emit one, and it must not
    # produce NaN and take the whole tracker down with it.
    tracker.update(np.array([[10.0, 10.0, 10.0, 10.0, 0.9, 0]]))
    for _ in range(3):
        out = tracker.update(np.array([[10.0, 10.0, 10.0, 10.0, 0.9, 0]]))
    check("degenerate box produces no NaN",
          all(np.all(np.isfinite(o["bbox"])) for o in out))

    # The (N, 6) shape is the whole contract between the detector and the
    # tracker, and it used to be enforced with reshape(-1, 6). That accepts any
    # array whose size divides by 6, so six rows missing the class column became
    # five rows with every field shifted along by one: no error, plausible
    # boxes, entirely wrong. A wrong shape has to be loud.
    try:
        Sort().update(np.array([[10.0, 10.0, 50.0, 90.0, 0.9]] * 6))
        check("a missing column is rejected, not reshaped", False)
    except ValueError:
        check("a missing column is rejected, not reshaped", True)

    check("a single flat row of six is still accepted",
          len(Sort(min_hits=1).update(np.array([10.0, 10.0, 50.0, 90.0, 0.9, 0.0]))) == 1)


def test_class_and_score_carried() -> None:
    print("\n--- class and score are carried through ---")
    Sort.reset_ids()
    tracker = Sort(min_hits=1)

    for _ in range(4):
        out = tracker.update(np.array([box(100, 200, score=0.77, cls_id=2)]))
    check("class id preserved", out[0]["cls_id"] == 2, f"got {out[0]['cls_id']}")
    check("score preserved", abs(out[0]["score"] - 0.77) < 1e-6, f"got {out[0]['score']:.3f}")


def main() -> int:
    test_iou()
    test_single_object_keeps_one_id()
    test_two_objects_do_not_swap()
    test_id_survives_brief_occlusion()
    test_track_retired_after_max_age()
    test_single_frame_false_positive_suppressed()
    test_confirmed_track_survives_flickering_detector()
    test_class_label_survives_a_misclassified_frame()
    test_coasting_marks_predicted_boxes()
    test_empty_input_is_safe()
    test_class_and_score_carried()

    print(f"\n{'all checks passed' if not failures else str(failures) + ' FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
