from __future__ import annotations

import pytest

from bevserve import faultinject
from bevserve.detector import FakeDetector, FatalDetectorError
from bevserve.preprocess import ModelInputs


def test_fatal_on_the_configured_call_only(model_inputs: ModelInputs) -> None:
    wrapped = faultinject.FatalOnCall(FakeDetector(), call=2)
    assert len(wrapped.detect(model_inputs)) == 3
    with pytest.raises(FatalDetectorError, match="call 2"):
        wrapped.detect(model_inputs)
    assert len(wrapped.detect(model_inputs)) == 3
    assert wrapped.info.name == "fake-detector"


def test_refuses_without_acknowledgement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BEVSERVE_FAULT_INJECTION_ACK", raising=False)

    def must_not_run(*_: object, **__: object) -> None:
        raise AssertionError("server started")

    monkeypatch.setattr(faultinject.uvicorn, "run", must_not_run)
    assert faultinject.main(["--fatal-on-call", "2"]) == 2
