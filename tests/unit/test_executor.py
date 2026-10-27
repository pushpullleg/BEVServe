"""Execution-slot ownership, bounded admission and deadline behaviour (review correction 1)."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from bevserve.detector import DetectorError, FakeDetector, FatalDetectorError
from bevserve.executor import DeadlineExceededError, InferenceExecutor, QueueFullError
from bevserve.preprocess import ModelInputs

pytestmark = pytest.mark.anyio


def _executor(
    fake: FakeDetector, max_in_flight: int = 4
) -> tuple[InferenceExecutor, list[Exception]]:
    fatals: list[Exception] = []
    return InferenceExecutor(fake, max_in_flight, fatals.append), fatals


async def _wait_until(predicate: object, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not predicate():  # type: ignore[operator]
        if time.monotonic() > end:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


async def test_one_job_at_a_time_under_concurrent_load(model_inputs: ModelInputs) -> None:
    fake = FakeDetector(delay_s=0.01)
    ex, _ = _executor(fake, max_in_flight=20)

    async def one() -> int:
        with ex.try_admit() as adm:
            return len(await ex.run(adm, model_inputs, deadline_s=10))

    results = await asyncio.gather(*(one() for _ in range(20)))
    assert results == [3] * 20
    assert fake.calls == 20 and fake.max_active == 1
    assert ex.in_flight == 0
    ex.shutdown()


async def test_timed_out_running_job_keeps_the_slot(model_inputs: ModelInputs) -> None:
    gate = threading.Event()
    fake = FakeDetector(gate=gate)
    ex, _ = _executor(fake)

    with ex.try_admit() as adm_a, pytest.raises(DeadlineExceededError) as err_a:
        await ex.run(adm_a, model_inputs, deadline_s=0.2)
    assert err_a.value.stage == "running"
    assert ex.running_for() is not None  # job A still owns the slot

    # Job B is admitted but cannot start while A runs; its deadline passes in the queue.
    with ex.try_admit() as adm_b, pytest.raises(DeadlineExceededError) as err_b:
        await ex.run(adm_b, model_inputs, deadline_s=0.2)
    assert err_b.value.stage == "queued"
    assert fake.calls == 1  # B never started
    assert ex.in_flight == 1  # B's token was returned at cancel; A's is still held

    gate.set()
    await _wait_until(lambda: ex.in_flight == 0)
    with ex.try_admit() as adm_c:
        assert len(await ex.run(adm_c, model_inputs, deadline_s=5)) == 3
    assert fake.calls == 2 and fake.max_active == 1  # A and C only, never together
    ex.shutdown()


async def test_admission_bound_counts_abandoned_running_job(model_inputs: ModelInputs) -> None:
    gate = threading.Event()
    ex, _ = _executor(FakeDetector(gate=gate), max_in_flight=1)
    with ex.try_admit() as adm, pytest.raises(DeadlineExceededError):
        await ex.run(adm, model_inputs, deadline_s=0.1)
    with pytest.raises(QueueFullError):
        ex.try_admit()
    gate.set()
    await _wait_until(lambda: ex.in_flight == 0)
    ex.try_admit().release()
    ex.shutdown()


async def test_unsubmitted_admission_is_returned(model_inputs: ModelInputs) -> None:
    ex, _ = _executor(FakeDetector(), max_in_flight=1)
    with pytest.raises(RuntimeError), ex.try_admit():
        raise RuntimeError("decode failed before submit")
    assert ex.in_flight == 0
    ex.try_admit().release()
    ex.shutdown()


async def test_bound_rejects_excess_requests(model_inputs: ModelInputs) -> None:
    gate = threading.Event()
    fake = FakeDetector(gate=gate)
    ex, _ = _executor(fake, max_in_flight=2)
    adm1, adm2 = ex.try_admit(), ex.try_admit()
    t1 = asyncio.create_task(ex.run(adm1, model_inputs, deadline_s=5))
    t2 = asyncio.create_task(ex.run(adm2, model_inputs, deadline_s=5))
    await _wait_until(lambda: fake.calls == 1)
    with pytest.raises(QueueFullError):
        ex.try_admit()
    gate.set()
    assert [len(r) for r in await asyncio.gather(t1, t2)] == [3, 3]
    assert fake.max_active == 1
    ex.shutdown()


async def test_fatal_error_disables_executor_without_calling_detector_again(
    model_inputs: ModelInputs,
) -> None:
    fake = FakeDetector(fail="fatal")
    ex, fatals = _executor(fake)
    with ex.try_admit() as adm, pytest.raises(FatalDetectorError):
        await ex.run(adm, model_inputs, deadline_s=5)
    assert len(fatals) == 1 and ex.fatal
    with pytest.raises(FatalDetectorError):
        ex.try_admit()
    assert fake.calls == 1
    ex.shutdown()


async def test_ordinary_error_leaves_executor_usable(model_inputs: ModelInputs) -> None:
    fake = FakeDetector(fail="error")
    ex, fatals = _executor(fake)
    with ex.try_admit() as adm, pytest.raises(DetectorError):
        await ex.run(adm, model_inputs, deadline_s=5)
    fake.fail = None
    with ex.try_admit() as adm:
        assert len(await ex.run(adm, model_inputs, deadline_s=5)) == 3
    assert fatals == [] and not ex.fatal
    ex.shutdown()


async def test_running_for_reports_a_long_job(model_inputs: ModelInputs) -> None:
    gate = threading.Event()
    fake = FakeDetector(gate=gate)
    ex, _ = _executor(fake)
    assert ex.running_for() is None
    adm = ex.try_admit()
    task = asyncio.create_task(ex.run(adm, model_inputs, deadline_s=5))
    await _wait_until(lambda: fake.calls == 1)
    await asyncio.sleep(0.15)
    running = ex.running_for()
    assert running is not None and running >= 0.1
    gate.set()
    await task
    assert ex.running_for() is None
    ex.shutdown()
