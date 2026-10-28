"""Experimental parallel decode + preprocessing of the six cameras (M6, PREPROCESS_THREADS).

The serial path (`api._decode_and_build`: decode all six, then `build_inputs`) stays the default.
This module runs the *same* functions, `imaging.decode_rgb` and `preprocess.camera_plane`, for each
camera on a bounded, process-wide thread pool, then assembles the planes with the same
`preprocess.assemble_inputs`. Model inputs are therefore byte-identical to the serial path; no
image-processing arithmetic is duplicated here.

Guarantees:
- Order: planes are placed by camera index (CAMERA_ORDER), never by completion order.
- Isolation: a task reads one immutable JPEG `bytes` object and returns a new array; decoded
  images and intermediate buffers are task-local. Tasks share no mutable state.
- Errors, in the serial path's precedence: (1) the first camera in CAMERA_ORDER whose *decode*
  failed, (2) calibration (`to_rig`), (3) the first camera whose per-camera preprocessing failed.
  Every task of the request has finished before any error is raised, so no task outlives the
  request or touches its data afterwards.
- Bounded work: N worker threads (1..32); queued tasks are bounded by admission (at most
  6 x max_in_flight requests' cameras). OpenCV is set to one thread while the pool exists, to
  avoid N x OpenCV-pool oversubscription; the previous setting is restored on shutdown.
- Lifecycle: created with the service, shut down (pending tasks cancelled, workers joined) with
  it; worker threads are named "bevserve-prep" and start lazily on first use.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import cv2

from bevserve.calibration import RigCalibration
from bevserve.constants import CAMERA_ORDER
from bevserve.imaging import decode_rgb
from bevserve.preprocess import FloatArray, ModelInputs, assemble_inputs, camera_plane

MAX_THREADS = 32
THREAD_PREFIX = "bevserve-prep"

# OpenCV's thread count is process-global: one thread while any pool exists, then restored.
_cv_lock = threading.Lock()
_cv_users = 0
_cv_saved: int | None = None


def _cv_acquire() -> None:
    global _cv_users, _cv_saved
    with _cv_lock:
        if _cv_users == 0:
            _cv_saved = cv2.getNumThreads()
            cv2.setNumThreads(1)
        _cv_users += 1


def _cv_release() -> None:
    global _cv_users
    with _cv_lock:
        _cv_users -= 1
        if _cv_users == 0 and _cv_saved is not None:
            cv2.setNumThreads(_cv_saved)


@dataclass(frozen=True)
class _CameraResult:
    plane: FloatArray | None = None
    decode_error: Exception | None = None
    build_error: Exception | None = None
    decoded_at: float | None = None


class CameraDecodeError(Exception):
    """Decode failure of one camera, raised in CAMERA_ORDER precedence (wraps the original)."""

    def __init__(self, camera: str, error: Exception) -> None:
        super().__init__(f"{camera}: {error}")
        self.camera = camera
        self.error = error


def _camera_task(name: str, data: bytes) -> _CameraResult:
    try:
        image = decode_rgb(data)
    except Exception as exc:
        return _CameraResult(decode_error=exc)
    decoded_at = time.perf_counter()
    try:
        return _CameraResult(plane=camera_plane(name, image), decoded_at=decoded_at)
    except Exception as exc:
        return _CameraResult(build_error=exc, decoded_at=decoded_at)


class CameraPool:
    def __init__(self, threads: int) -> None:
        if not 1 <= threads <= MAX_THREADS:
            raise ValueError(f"preprocess threads must be 1..{MAX_THREADS}, got {threads}")
        self.threads = threads
        self._lock = threading.Lock()
        self._closed = False
        _cv_acquire()
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix=THREAD_PREFIX
        )

    def build(
        self, raw: Mapping[str, bytes], to_rig: Callable[[], RigCalibration]
    ) -> tuple[ModelInputs, float, float]:
        """raw JPEG/PNG bytes by camera -> (inputs, decode_ms, preprocess_ms).

        decode_ms is the time until the last camera is decoded; preprocess_ms the rest. The
        per-camera stages overlap, so they are not the serial path's per-stage sums.
        """
        if tuple(raw) != CAMERA_ORDER:
            raise ValueError(f"images must be exactly {CAMERA_ORDER}, in that order")
        t0 = time.perf_counter()
        with self._lock:
            if self._closed:
                raise RuntimeError("camera pool is shut down")
            futures = [self._pool.submit(_camera_task, n, raw[n]) for n in CAMERA_ORDER]
        results = [f.result() for f in futures]  # waits for every task of this request
        for name, r in zip(CAMERA_ORDER, results, strict=True):
            if r.decode_error is not None:
                raise CameraDecodeError(name, r.decode_error)
        decoded = max(r.decoded_at for r in results if r.decoded_at is not None)
        rig = to_rig()
        for r in results:
            if r.build_error is not None:
                raise r.build_error
        planes = [r.plane for r in results if r.plane is not None]
        inputs = assemble_inputs(planes, rig)
        end = time.perf_counter()
        return inputs, (decoded - t0) * 1000, (end - decoded) * 1000

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)
        _cv_release()
