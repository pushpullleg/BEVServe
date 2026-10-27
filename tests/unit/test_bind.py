"""Bind policy (review correction 3)."""

from __future__ import annotations

import pytest

from bevserve import cli
from bevserve.bind import BindNotAllowedError, check_bind_host


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
def test_loopback_is_allowed(host: str) -> None:
    check_bind_host(host, ())


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "10.0.0.5", "example.org"])
def test_other_interfaces_are_refused(host: str) -> None:
    with pytest.raises(BindNotAllowedError):
        check_bind_host(host, ())


def test_allow_list_authorizes_a_specific_address() -> None:
    check_bind_host("10.0.0.5", ("10.0.0.5",))
    with pytest.raises(BindNotAllowedError):
        check_bind_host("10.0.0.6", ("10.0.0.5",))


def test_cli_refuses_unauthorized_host_before_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEVSERVE_HOST", "0.0.0.0")

    def must_not_run(*_: object, **__: object) -> None:
        raise AssertionError("server started")

    monkeypatch.setattr(cli.uvicorn, "run", must_not_run)
    assert cli.main() == 2


def test_cli_accepts_allow_listed_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEVSERVE_HOST", "0.0.0.0")
    monkeypatch.setenv("BEVSERVE_ALLOWED_BIND_HOSTS", '["0.0.0.0"]')
    started: list[str] = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, host, **_: started.append(host))
    assert cli.main() == 0 and started == ["0.0.0.0"]


def test_request_logs_survive_mmcv_silencing_root_handlers() -> None:
    """mmcv.get_logger raises root StreamHandlers to ERROR while BEVDet loads."""
    import io
    import logging

    stream = io.StringIO()
    cli.configure_logging(stream)
    try:
        for handler in logging.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)
        logging.getLogger("bevserve.access").info("request_id=abc status=200")
        assert "request_id=abc status=200" in stream.getvalue()
    finally:
        logging.getLogger("bevserve").handlers.clear()
        logging.getLogger("bevserve").propagate = True
