# Coordinate frames

Every box BEVServe returns is in **`ego@CAM_FRONT_LEFT`**: the ego-vehicle frame at the moment
CAM_FRONT_LEFT captured its image. Units are meters, radians and meters per second.

## Frames

| Frame | Axes | Defined by |
| --- | --- | --- |
| camera *i* | x right, y down, z forward (optical axis) | the camera |
| ego *i* | x forward, y left, z up (nuScenes ego-vehicle frame) | the vehicle pose when camera *i* fired |
| key ego | same axes as ego | ego 0 = ego at CAM_FRONT_LEFT, the first camera in the model's order |
| global | world frame of the calibration source | `ego2global` poses |

Canonical camera order (fixed by the trained model):
`CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT`.
Requests name cameras explicitly; the order inside a request does not matter.

## Calibration fields

- `intrinsic`: 3x3 pinhole matrix for the **1600x900** source image (no distortion terms).
- `sensor2ego`: camera *i* -> ego *i*. Quaternion `rotation_wxyz` = (w, x, y, z) and
  `translation_m`, in the nuScenes convention.
- `ego2global` (optional, all six or none): ego *i* -> global.

## Transform used by the model

For camera *i*, BEVDet maps camera coordinates to the key-ego frame as

```
T(key ego <- cam i) = inv(T(global <- ego 0)) @ T(global <- ego i) @ T(ego i <- cam i)
```

(`BEVDet.prepare_inputs`; reproduced in `bevserve.calibration.sensor_to_key_ego`). Consequences:

- With `ego2global` omitted, every ego pose is the identity and the key-ego frame is simply the
  rig's ego frame. This is exact for a synchronized rig and an approximation for nuScenes, whose
  six cameras fire at slightly different times.
- Moving the whole rig in the global frame does not change the result: only relative ego poses
  matter.

Tested in `tests/unit/test_calibration.py` with deliberately different per-camera ego poses, and
compared bit for bit with BEVDet's own `prepare_inputs` by `tools/reference_check_bevdet.py`.

## Box fields

| Field | Meaning |
| --- | --- |
| `center` | geometric box center (x, y, z) in `ego@CAM_FRONT_LEFT` (see below) |
| `size` | (x_size, y_size, z_size); nuScenes `wlh` = (y_size, x_size, z_size) |
| `yaw` | rotation about z of the output frame |
| `velocity` | (vx, vy) in the output frame |

BEVDet's decoder (`CenterHead.get_bboxes`) stores the **bottom** center in z: it subtracts
half the box height before building the box tensor, and BEVDet's own nuScenes formatter then
writes that bottom z as the box center (harmless for nuScenes mAP/NDS, which use x and y only).
BEVServe returns the geometric center: `center.z = bottom z + z_size / 2`
(`bevserve.boxes.detections_from_arrays`). x, y, sizes, yaw and velocity are unchanged.

To convert to the global frame, apply CAM_FRONT_LEFT's `ego2global` pose. Note that BEVDet's
own nuScenes formatter applies CAM_FRONT's pose instead; BEVServe does not convert, so clients
choose explicitly.
