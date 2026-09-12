"""SORT: Simple Online and Realtime Tracking.

The detector runs on one frame at a time and has no memory. It reports "there
is a person here" in frame 10 and "there is a person here" in frame 11 without
any notion that these are the same person. Tracking is what supplies that
missing link, and a tracking ID is the visible result of it.

SORT does this in three steps per frame:

  1. PREDICT   Move every existing track forward using its estimated velocity,
               giving a guess of where it should be in this frame.
  2. ASSOCIATE Match this frame's detections to those predictions by overlap,
               choosing the set of pairings with the best total overlap.
  3. UPDATE    Correct matched tracks toward their detection, start tracks for
               unmatched detections, and retire tracks that have gone missing.

Written from the algorithm rather than pulled from a library, so every constant
here is one this file has to justify. The Kalman filter is plain numpy: it is
seven lines of linear algebra and adding a dependency to avoid them would hide
the part worth understanding.

Reference: Bewley et al., "Simple Online and Realtime Tracking" (2016).
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Intersection over union between every box in A and every box in B.

    Returns an (len(A), len(B)) matrix. IoU is the overlap area divided by the
    combined area, so it is 1.0 for identical boxes and 0.0 for boxes that do
    not touch. It is used instead of centre distance because it accounts for
    size: two boxes can share a centre while being wildly different objects.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.expand_dims(boxes_a, 1)  # (N, 1, 4)
    b = np.expand_dims(boxes_b, 0)  # (1, M, 4)

    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])

    # Clamped at zero: negative width means the boxes do not overlap at all.
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def bbox_to_measurement(bbox: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] to [centre x, centre y, area, aspect ratio].

    The filter tracks area and aspect ratio rather than width and height
    because an object moving towards the camera changes area smoothly while
    keeping its aspect ratio roughly fixed. Splitting them this way lets the
    filter treat "getting closer" and "changing shape" as separate events.
    """
    x1, y1, x2, y2 = bbox[:4]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w * h, w / h], dtype=np.float64)


def measurement_to_bbox(z: np.ndarray) -> np.ndarray:
    """Inverse of bbox_to_measurement, clamped so a bad state cannot produce NaN."""
    cx, cy, s, r = z[:4]
    s = max(float(s), 1e-6)
    r = max(float(r), 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)


class KalmanBoxTracker:
    """Constant velocity Kalman filter following one object.

    State is [cx, cy, area, ratio, d_cx, d_cy, d_area]: four measured
    quantities plus the rate of change of three of them. Aspect ratio has no
    velocity term, since an object's shape is assumed not to drift steadily in
    one direction.

    The filter matters most when the detector misses a frame. With no filter a
    track would simply freeze in place; here it keeps moving at its estimated
    velocity, so when the detection reappears a few frames later it is still
    close enough to be matched to the same ID.
    """

    _next_id = 1

    def __init__(self, bbox: np.ndarray, cls_id: int, score: float) -> None:
        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self.score = float(score)

        # Every class this track has been assigned, and how often. The reported
        # class is the majority vote rather than the latest detection.
        #
        # A detector misclassifies on the odd frame: in the sample footage a car
        # is labelled "bus" for a single frame out of hundreds. Taking the most
        # recent answer makes the on-screen label flicker between the two, which
        # looks like the tracker has lost the object when it has not. The vote
        # is over the whole life of the track, so one bad frame cannot outvote
        # the accumulated evidence.
        self.class_votes: Counter[int] = Counter([int(cls_id)])

        # State transition: position += velocity on each step.
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        # Measurement matrix: a detection observes the four position terms and
        # says nothing directly about velocity.
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        # Measurement noise. Area and aspect ratio are trusted less than centre
        # position because detector boxes tend to breathe at the edges even
        # when the object is still.
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0

        # Initial covariance. Velocities start unobservable, so their variance
        # starts very high and the filter is free to learn them from motion.
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0

        # Process noise: how much the model expects reality to deviate from
        # constant velocity between frames.
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[-1, -1] *= 0.01

        self.x = np.zeros((7, 1))
        self.x[:4, 0] = bbox_to_measurement(bbox)

        self.time_since_update = 0   # frames since a detection last matched
        self.hits = 0                # total detections matched to this track
        self.hit_streak = 0          # consecutive frames matched, resets on a miss
        self.age = 0                 # frames since the track was created

        # Latched once the track has proved itself, and never cleared. See the
        # note in Sort.update for why a track that has been confirmed must not
        # be demoted by a single missed frame.
        self.confirmed = False

    def predict(self) -> np.ndarray:
        """Step the state forward one frame and return the predicted box."""
        # Area must not be allowed to go negative: a box cannot have negative
        # size, and letting it happen produces NaN in the width calculation.
        if self.x[2, 0] + self.x[6, 0] <= 0:
            self.x[6, 0] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            # A track that missed last frame has its streak broken. This is what
            # stops a long-lost track from being treated as reliably confirmed.
            self.hit_streak = 0
        self.time_since_update += 1

        return measurement_to_bbox(self.x[:4, 0])

    def update(self, bbox: np.ndarray, cls_id: int, score: float) -> None:
        """Correct the state towards a detection that was matched to this track."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.class_votes[int(cls_id)] += 1
        self.score = float(score)

        z = bbox_to_measurement(bbox).reshape(4, 1)
        y = z - self.H @ self.x                      # innovation: measured minus predicted
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def bbox(self) -> np.ndarray:
        return measurement_to_bbox(self.x[:4, 0])

    @property
    def cls_id(self) -> int:
        """The class this track has been called most often."""
        return self.class_votes.most_common(1)[0][0]


def associate(
    detections: np.ndarray, predictions: np.ndarray, iou_threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match detections to predicted track positions.

    Uses the Hungarian algorithm rather than greedy nearest-match. Greedy
    matching takes the single best pair first and can strand a later track with
    a poor partner; the Hungarian algorithm minimises total cost across all
    pairs at once, which is what keeps IDs stable when two objects pass close
    to each other.

    Returns (matches, unmatched detection indices, unmatched track indices).
    """
    if len(predictions) == 0 or len(detections) == 0:
        return [], list(range(len(detections))), list(range(len(predictions)))

    iou = iou_batch(detections[:, :4], predictions)

    # linear_sum_assignment minimises, and a high IoU is a good match, so the
    # cost is negated.
    det_idx, trk_idx = linear_sum_assignment(-iou)

    matches, matched_dets, matched_trks = [], set(), set()
    for d, t in zip(det_idx, trk_idx):
        # The Hungarian algorithm returns a complete assignment, including pairs
        # that barely overlap. Anything under the threshold is thrown back and
        # treated as unmatched, otherwise a new object entering the frame would
        # be handed the ID of an unrelated departing one.
        if iou[d, t] < iou_threshold:
            continue
        matches.append((int(d), int(t)))
        matched_dets.add(int(d))
        matched_trks.add(int(t))

    unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]
    unmatched_trks = [t for t in range(len(predictions)) if t not in matched_trks]
    return matches, unmatched_dets, unmatched_trks


class Sort:
    """Multi-object tracker.

    Args:
        max_age: frames a track survives without a detection before deletion.
            Higher values ride out longer occlusions but risk holding a stale
            box after an object has genuinely left.
        min_hits: consecutive detections before a new track is reported. This
            suppresses the one-frame false positives that every detector
            produces, at the cost of a short delay before a real object appears.
        iou_threshold: minimum overlap for a detection to be considered the
            same object as a track.
        coast: frames a confirmed track keeps being reported after a missed
            detection, using its predicted position. Set to 0 to report only
            tracks matched in the current frame.

            This is what the Kalman filter is for. A detector drops the
            occasional frame even on a large, obvious object, and without
            coasting the box simply vanishes and reappears, which reads as a
            broken tracker. Reported positions during a gap are predictions
            rather than measurements, so they are flagged as such in the output
            and the caller can draw them differently.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        coast: int = 3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.coast = coast
        self.tracks: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: np.ndarray | None = None) -> list[dict]:
        """Advance the tracker by one frame.

        `detections` is an (N, 6) array of [x1, y1, x2, y2, score, class_id].
        Pass an empty array for a frame in which nothing was detected: the
        tracker still needs to age its tracks, and skipping the call would
        freeze them in place.

        Returns one dict per confirmed track with keys id, bbox, cls_id, score.
        """
        self.frame_count += 1
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 6), dtype=np.float32)
        detections = np.asarray(detections, dtype=np.float64).reshape(-1, 6)

        # 1. PREDICT. Any track whose filter produces a non-finite box is
        # dropped rather than propagated: one NaN would poison every IoU
        # comparison it takes part in.
        predictions, stale = [], []
        for i, track in enumerate(self.tracks):
            box = track.predict()
            if np.all(np.isfinite(box)):
                predictions.append(box)
            else:
                stale.append(i)
        for i in reversed(stale):
            self.tracks.pop(i)
        predictions = np.array(predictions).reshape(-1, 4)

        # 2. ASSOCIATE.
        matches, unmatched_dets, _ = associate(detections, predictions, self.iou_threshold)

        # 3. UPDATE.
        for d, t in matches:
            track = self.tracks[t]
            track.update(detections[d, :4], int(detections[d, 5]), float(detections[d, 4]))
            if track.hits >= self.min_hits:
                track.confirmed = True

        for d in unmatched_dets:
            self.tracks.append(
                KalmanBoxTracker(detections[d, :4], int(detections[d, 5]), float(detections[d, 4]))
            )

        # Retire tracks that have gone unmatched for too long.
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        results = []
        for track in self.tracks:
            # A track is reported once it has been matched min_hits times, and
            # stays reportable from then on.
            #
            # The obvious condition is `hit_streak >= min_hits`, which is what
            # the SORT paper describes, but it behaves badly against a real
            # detector. A missed frame resets hit_streak, so a track that had
            # been followed for hundreds of frames would have to earn three
            # consecutive detections again before being drawn. On the sample
            # street clip, where confidence hovers near the threshold and the
            # detector drops the occasional frame, that blanked out a live
            # track on 21 frames: the object was detected and its ID was intact,
            # but nothing was displayed. Latching the confirmation fixes that
            # while still suppressing one-frame false positives, which never
            # reach min_hits at all.
            #
            # The frame_count exception covers the opening frames, where there
            # has not yet been time to accumulate any hits.
            established = track.confirmed or self.frame_count <= self.min_hits
            if track.time_since_update <= self.coast and established:
                results.append(
                    {
                        "id": track.id,
                        "bbox": track.bbox,
                        "cls_id": track.cls_id,
                        "score": track.score,
                        # False when this frame's box came from a detection,
                        # True when it is the filter's prediction during a gap.
                        "predicted": track.time_since_update > 0,
                    }
                )
        return results

    @staticmethod
    def reset_ids() -> None:
        """Restart ID numbering. Used by the tests so each case starts at 1."""
        KalmanBoxTracker._next_id = 1
