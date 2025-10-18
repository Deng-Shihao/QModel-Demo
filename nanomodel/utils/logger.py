from __future__ import annotations

import contextlib
import inspect
import itertools
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Iterable, Iterator, Optional, Sequence

from tqdm import tqdm


_LOGGER_CACHE: dict[str, "_LoggerWrapper"] = {}
_LOGGING_CONFIGURED = False


def _configure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _LOGGING_CONFIGURED = True


class _LoggerMethod:
    def __init__(self, logger: logging.Logger, method_name: str, once_registry: set[str]):
        self._logger = logger
        self._method_name = method_name
        self._method = getattr(logger, method_name)
        self._once_registry = once_registry

    def __call__(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._method(msg, *args, **kwargs)

    def once(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        key = self._make_key(msg, args, kwargs)
        if key in self._once_registry:
            return
        self._once_registry.add(key)
        self._method(msg, *args, **kwargs)

    def _make_key(self, msg: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        # Include the method name so info.once and warn.once keep separate caches.
        payload = [self._method_name, repr(msg)]
        if args:
            payload.append("|args:" + "|".join(repr(a) for a in args))
        if kwargs:
            serialized = "|".join(f"{k}={repr(v)}" for k, v in sorted(kwargs.items()))
            payload.append("|kwargs:" + serialized)
        return "".join(payload)


class _ProgressBar:
    """tqdm-backed progress helper compatible with legacy logbar usage."""

    def __init__(
        self,
        sequence: Iterable[Any] | int,
        *,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: str = "steps",
        leave: bool = False,
    ) -> None:
        if isinstance(sequence, int):
            self._iterable: Iterable[Any] = range(sequence)
            computed_total = sequence
        else:
            self._iterable = sequence
            if total is None:
                try:
                    computed_total = len(sequence)  # type: ignore[arg-type]
                except (TypeError, AttributeError):
                    computed_total = None
            else:
                computed_total = total
        self._tqdm = tqdm(total=computed_total, desc=desc, unit=unit, leave=leave)
        self._displayed = 0
        self._total = computed_total
        self._title = desc or ""
        self._subtitle = ""
        self.current_iter_step = 0
        self._closed = False
        self._left_steps_offset = 0
        self._manual_mode = False

    def __iter__(self) -> Iterator[Any]:
        for item in self._iterable:
            yield item
            self.current_iter_step += 1
            self._increment(1)
        if not self._manual_mode:
            self.close()

    def __len__(self) -> int:
        if self._total is not None:
            return int(self._total)
        try:
            return len(self._iterable)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            return 0

    def __enter__(self) -> "_ProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def manual(self) -> "_ProgressBar":
        self._manual_mode = True
        return self

    def set(self, **kwargs: Any) -> "_ProgressBar":
        if "left_steps_offset" in kwargs:
            try:
                self._left_steps_offset = int(kwargs["left_steps_offset"])
            except (ValueError, TypeError):
                self._left_steps_offset = 0
            self._sync_progress()
        return self

    def title(self, text: str) -> "_ProgressBar":
        self._title = text or ""
        self._refresh()
        return self

    def subtitle(self, text: str) -> "_ProgressBar":
        self._subtitle = text or ""
        self._sync_progress()
        self._refresh()
        return self

    def draw(self) -> "_ProgressBar":
        self._refresh()
        return self

    def next(self, amount: int = 1) -> "_ProgressBar":
        if amount <= 0:
            return self
        self.current_iter_step += amount
        self._increment(amount)
        return self

    def close(self) -> None:
        if not self._closed:
            self._tqdm.close()
            self._closed = True

    def _increment(self, amount: int) -> None:
        if self._closed or amount <= 0:
            return
        self._tqdm.update(amount)
        self._displayed += amount

    def _sync_progress(self) -> None:
        if self._closed:
            return
        target = max(int(self.current_iter_step + self._left_steps_offset), 0)
        delta = target - self._displayed
        if delta > 0:
            self._increment(delta)

    def _refresh(self) -> None:
        if self._closed:
            return
        if self._title:
            self._tqdm.set_description_str(self._title, refresh=False)
        if self._subtitle:
            self._tqdm.set_postfix_str(self._subtitle, refresh=False)
        else:
            self._tqdm.set_postfix_str("", refresh=False)
        self._tqdm.refresh()


class _Spinner:
    """Background spinner built on tqdm to indicate ongoing work."""

    _FRAMES = itertools.cycle("|/-\\")

    def __init__(self, title: str, interval: float = 0.1) -> None:
        self._interval = max(interval, 0.05)
        self._tqdm = tqdm(total=0, desc=title, bar_format="{desc} {postfix}", leave=False)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def __enter__(self) -> "_Spinner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self._interval * 2)
        self._tqdm.close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            frame = next(self._FRAMES)
            self._tqdm.set_postfix_str(frame, refresh=True)
            time.sleep(self._interval)


class _LoggerWrapper:
    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._once_registry: set[str] = set()
        self.debug = _LoggerMethod(logger, "debug", self._once_registry)
        self.info = _LoggerMethod(logger, "info", self._once_registry)
        self.warning = _LoggerMethod(logger, "warning", self._once_registry)
        self.error = _LoggerMethod(logger, "error", self._once_registry)
        self.exception = _LoggerMethod(logger, "exception", self._once_registry)
        self.critical = _LoggerMethod(logger, "critical", self._once_registry)

        # Legacy aliases
        self.warn = self.warning
        self.info_once = self.info.once
        self.warn_once = self.warning.once
        self.warning_once = self.warning.once

    def pb(
        self,
        sequence: Iterable[Any] | int,
        *,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: str = "steps",
        leave: bool = False,
    ) -> _ProgressBar:
        return _ProgressBar(sequence, total=total, desc=desc, unit=unit, leave=leave)

    def spinner(self, title: str, interval: float = 0.1) -> _Spinner:
        return _Spinner(title=title, interval=interval)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._logger, item)


def setup_logger(name: Optional[str] = None) -> _LoggerWrapper:
    """Return a shared logger wrapper backed by Python's logging module."""
    _configure_logging()
    if name is None:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        module = inspect.getmodule(caller) if caller is not None else None
        resolved = module.__name__ if module and module.__name__ else "nanomodel"
        if caller is not None:
            del caller
        if frame is not None:
            del frame
        name = resolved
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    wrapper = _LoggerWrapper(logger)
    _LOGGER_CACHE[name] = wrapper
    return wrapper


class QuantizationRegionTimer:
    """Accumulate timing statistics for named quantisation regions."""

    DEFAULT_REGIONS: Sequence[tuple[str, str]] = (
        ("model_load", "Model load"),
        ("model_reload", "Turtle reload"),
        ("capture_inputs", "Capture inputs"),
        ("forward_hook", "Forward hook"),
        ("pre_quant_forward", "Pre-quant forward"),
        ("process_quant", "Process quant"),
        ("post_quant_forward", "Post-quant replay"),
        ("submodule_finalize", "Submodule finalize"),
        ("submodule_finalize_create", "Finalize create"),
        ("submodule_finalize_pack", "Finalize pack"),
        ("submodule_finalize_offload", "Finalize offload"),
        ("process_finalize", "Process finalize"),
        ("model_save", "Model save"),
    )

    def __init__(self, logger: Optional[_LoggerWrapper] = None) -> None:
        self._logger = logger or setup_logger("nanomodel.timer")
        self._lock = threading.Lock()
        self._region_labels = OrderedDict(self.DEFAULT_REGIONS)
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._stats: OrderedDict[str, dict[str, Any]] = OrderedDict(
                (region, self._fresh_stat()) for region in self._region_labels.keys()
            )

    def record(self, region: str, duration: float, *, source: Optional[str] = None) -> None:
        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            return
        if duration_value < 0:
            duration_value = 0.0

        with self._lock:
            if region not in self._stats:
                if region not in self._region_labels:
                    self._region_labels[region] = region.replace("_", " ").title()
                self._stats[region] = self._fresh_stat()

            stat = self._stats[region]
            stat["total"] = float(stat.get("total", 0.0)) + duration_value
            stat["count"] = int(stat.get("count", 0)) + 1
            stat["last"] = duration_value
            if source is not None:
                stat["source"] = source

    def flush(self) -> None:
        snapshot = self.snapshot()
        populated = [
            (region, stat)
            for region, stat in snapshot.items()
            if stat.get("count", 0) > 0
        ]
        if not populated:
            return
        overall = sum(stat["total"] for _, stat in populated)
        lines = []
        for region, stat in populated:
            label = self._region_labels.get(region, region)
            total = stat["total"]
            count = stat["count"]
            last = stat["last"]
            avg = total / count if count else 0.0
            pct = (total / overall * 100.0) if overall > 0 else 0.0
            source = stat.get("source") or "-"
            lines.append(
                f"  - {label:<24} total={total:.3f}s avg={avg:.3f}s last={last:.3f}s count={count:>3} pct={pct:>5.1f}% src={source}"
            )
        self._logger.info("Quantisation timing summary:\n%s", "\n".join(lines))

    def snapshot(self) -> OrderedDict[str, dict[str, Any]]:
        with self._lock:
            return OrderedDict(
                (
                    region,
                    {
                        "total": float(stat.get("total", 0.0)),
                        "count": int(stat.get("count", 0)),
                        "last": float(stat.get("last", 0.0)),
                        "source": stat.get("source"),
                    },
                )
                for region, stat in self._stats.items()
            )

    @contextlib.contextmanager
    def measure(self, region: str, *, source: Optional[str] = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.record(region, elapsed, source=source)

    def _fresh_stat(self) -> dict[str, Any]:
        return {"total": 0.0, "count": 0, "last": 0.0, "source": None}


@contextlib.contextmanager
def log_time_block(
    block_name: str,
    *,
    logger: Optional[_LoggerWrapper | logging.Logger] = None,
    module_name: Optional[str] = None,
) -> Iterator[None]:
    active_logger = logger or setup_logger()
    emit = active_logger.info if hasattr(active_logger, "info") else logging.getLogger("nanomodel").info
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if module_name:
            emit(f"[{block_name}] ({module_name}) took {elapsed:.3f}s")
        else:
            emit(f"[{block_name}] took {elapsed:.3f}s")
