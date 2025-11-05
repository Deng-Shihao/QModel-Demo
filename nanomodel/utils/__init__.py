from __future__ import annotations

"""Top-level exports and shared utilities for the `nanomodel.utils` package."""

from .backend import BACKEND
from .logger import setup_logger
from .python import (
    gte_python_3_13_3,
    gte_python_3_14,
    has_gil_control,
    has_gil_disabled,
    log_gil_requirements_for,
)
from .threads import AsyncManager, SerialWorker

logger = setup_logger("nanomodel.utils")

# Reusable worker pools that modules import for background work scheduling.
ASYNC_BG_QUEUE: AsyncManager = AsyncManager(threads=4)
SERIAL_BG_QUEUE: SerialWorker = SerialWorker()


def _register_optional_perplexity() -> None:
    """Expose Perplexity helper only when Python runs without the GIL."""
    if has_gil_disabled():
        from .perplexity import Perplexity as _Perplexity

        globals()["Perplexity"] = _Perplexity
        return

    if has_gil_control():
        logger.warning.once(
            "Python reports GIL control support but it is still enabled; "
            "Perplexity remains unavailable."
        )

    log_gil_requirements_for("utils/Perplexity")


_register_optional_perplexity()

__all__ = [
    "ASYNC_BG_QUEUE",
    "BACKEND",
    "SERIAL_BG_QUEUE",
    "AsyncManager",
    "SerialWorker",
    "gte_python_3_13_3",
    "gte_python_3_14",
    "has_gil_control",
    "has_gil_disabled",
    "log_gil_requirements_for",
    "logger",
    "setup_logger",
]

if "Perplexity" in globals():
    __all__.append("Perplexity")
