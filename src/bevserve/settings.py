"""Service configuration from BEVSERVE_* environment variables."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MiB = 1024 * 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BEVSERVE_", frozen=True, extra="forbid")

    # Network (review correction 3): loopback by default; anything else must be allow-listed.
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    allowed_bind_hosts: tuple[str, ...] = ()

    # Input limits (review correction 5), enforced before any image is decoded.
    max_request_bytes: int = Field(default=50 * MiB, ge=1)
    max_image_bytes: int = Field(default=8 * MiB, ge=1)
    max_calibration_bytes: int = Field(default=64 * 1024, ge=1)

    # Execution slot (review correction 1): jobs in flight = 1 running + queued.
    max_in_flight: int = Field(default=5, ge=1)
    deadline_s: float = Field(default=10.0, gt=0)

    # Experimental (M6): 0 = serial decode + preprocessing (default, as in M5); N = the six
    # cameras' decode + preprocessing on a shared pool of N threads (byte-identical inputs).
    preprocess_threads: int = Field(default=0, ge=0, le=32)
    stall_after_s: float = Field(default=30.0, gt=0)

    # Fatal runtime failure (review correction 2): exit so the supervisor restarts the process.
    fatal_exit_grace_s: float = Field(default=0.5, ge=0)
    fatal_exit_code: int = Field(default=70, ge=1, le=255)

    # Detector backend. "fake" returns fixed synthetic boxes; "bevdet" loads BEVDet-R50 (M2).
    detector: Literal["fake", "bevdet"] = "fake"

    # Optional benchmark instrumentation: adds a Server-Timing response header with stage
    # durations (and CUDA-event forward timing for the real model). Off by default; does not
    # change the response body or any model output.
    server_timing: bool = False
    git_commit: str | None = None

    # BEVDet backend (operator-supplied; never taken from a request).
    bevdet_config: str | None = None  # path to configs/bevdet/bevdet-r50-cbgs.py
    checkpoint: str | None = None  # path to the trusted .pth
    checkpoint_sha256: str | None = None  # trusted reference value, required
    checkpoint_allowed_dirs: tuple[str, ...] = ()  # optional: checkpoint must resolve inside one
    checkpoint_max_bytes: int = Field(default=2 * 1024 * MiB, ge=1)
    device: str = "cuda:0"
    warmup_runs: int = Field(default=2, ge=1, le=20)
