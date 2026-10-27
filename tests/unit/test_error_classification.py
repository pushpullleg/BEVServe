"""Fatal-vs-ordinary classification of model errors (review correction 2). Pure: no torch needed."""

from __future__ import annotations

import pytest

from bevserve.detector import DetectorError, FatalDetectorError
from bevserve.model.bevdet_detector import classify_runtime_error


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: an illegal memory access was encountered",
        "CUDA error: device-side assert triggered",
        "CUDA error: unspecified launch failure",
        "CUDA error: misaligned address",
        "CUDNN_STATUS_EXECUTION_FAILED",
        "CUBLAS_STATUS_EXECUTION_FAILED when calling cublasSgemm",
        "uncorrectable ECC error encountered",
    ],
)
def test_unrecoverable_cuda_errors_are_fatal(message: str) -> None:
    assert classify_runtime_error(RuntimeError(message)) is FatalDetectorError


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory. Tried to allocate 2.00 GiB",
        "CUDA error: out of memory",
        "shape '[1, 6, 3]' is invalid for input of size 7",
        "Expected all tensors to be on the same device",
    ],
)
def test_recoverable_errors_are_ordinary(message: str) -> None:
    assert classify_runtime_error(RuntimeError(message)) is DetectorError
