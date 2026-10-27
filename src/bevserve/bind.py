"""Bind-address policy (review correction 3): serve on loopback unless explicitly authorized."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable


class BindNotAllowedError(ValueError):
    pass


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_bind_host(host: str, allowed: Iterable[str]) -> None:
    """Accept loopback addresses; anything else only if listed in BEVSERVE_ALLOWED_BIND_HOSTS.

    `--network none` is for offline verification runs and is unrelated to this check: a
    serving process always binds a real interface, and that interface must be authorized.
    """
    if is_loopback(host) or host in set(allowed):
        return
    raise BindNotAllowedError(
        f"refusing to bind {host!r}: not a loopback address and not in BEVSERVE_ALLOWED_BIND_HOSTS"
    )
