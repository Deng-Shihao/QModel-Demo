"""Environment variable helpers used across NanoModel."""

from __future__ import annotations

import os


_TRUTHY = {"1", "true", "yes", "on", "y"}


def env_flag(name: str, default: str | bool | None = "0") -> bool:
    """Return ``True`` when an env var is set to a truthy value."""

    value = os.getenv(name)
    if value is None:
        if default is None:
            return False
        if isinstance(default, bool):
            return default
        value = default
    return str(value).strip().lower() in _TRUTHY


__all__ = ["env_flag"]
