"""Convert BEVDet's 9-value box tensor to API detections (pure numpy, testable without torch).

BEVDet's CenterHead.get_bboxes stores the *bottom* center in z (it subtracts half the box
height before building LiDARInstance3DBoxes; centerpoint_head.py, get_bboxes). The API returns the
geometric center, so z is shifted back up by half the height here. x, y, sizes, yaw and velocity
are passed through unchanged. All values are in the key-ego frame (ego@CAM_FRONT_LEFT).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bevserve.detector import Detection

BOX_DIM = 9  # x, y, z_bottom, x_size, y_size, z_size, yaw, vx, vy


def detections_from_arrays(
    boxes_in: npt.ArrayLike, scores_in: npt.ArrayLike, labels_in: npt.ArrayLike
) -> list[Detection]:
    boxes = np.asarray(boxes_in, dtype=np.float64)
    scores = np.asarray(scores_in, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels_in).reshape(-1)
    if boxes.ndim != 2 or boxes.shape[1] != BOX_DIM:
        raise ValueError(f"expected (N, {BOX_DIM}) boxes, got {boxes.shape}")
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("boxes, scores and labels differ in length")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("non-finite values in detector output")
    out = []
    for box, score, label in zip(boxes, scores, labels, strict=True):
        x, y, z_bottom, dx, dy, dz, yaw, vx, vy = (float(v) for v in box)
        out.append(
            Detection(
                label_id=int(label),
                score=float(score),
                center=(x, y, z_bottom + 0.5 * dz),
                size=(dx, dy, dz),
                yaw=yaw,
                velocity=(vx, vy),
            )
        )
    return out
