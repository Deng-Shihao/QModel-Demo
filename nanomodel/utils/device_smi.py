from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional

import torch

try:  # optional dependency
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil may be missing
    psutil = None


def _cpu_memory_used_bytes() -> int:
    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            return int(mem.total - mem.available)
        except Exception:  # pragma: no cover - defensive, psutil quirks
            pass

    meminfo_path = "/proc/meminfo"
    if os.name == "posix" and os.path.exists(meminfo_path):
        totals: dict[str, int] = {}
        try:
            with open(meminfo_path, "r", encoding="ascii", errors="ignore") as fh:
                for line in fh:
                    parts = line.split(":")
                    if len(parts) != 2:
                        continue
                    key, value = parts
                    tokens = value.strip().split()
                    if not tokens:
                        continue
                    try:
                        totals[key] = int(tokens[0]) * 1024  # values in kB
                    except ValueError:
                        continue
            total = totals.get("MemTotal")
            available = totals.get("MemAvailable")
            if total is not None and available is not None:
                return int(total - available)
        except Exception:  # pragma: no cover - best effort fallback
            pass

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss
        if os.name == "posix":
            try:
                sysname = os.uname().sysname.lower()
            except AttributeError:  # pragma: no cover
                sysname = ""
            if "darwin" not in sysname:
                rss *= 1024
        return int(rss)
    except Exception:  # pragma: no cover - worst case fallback
        return 0


def _cuda_memory_used_bytes(index: int) -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info(index)
        return int(total - free)
    except Exception:
        try:
            allocated = torch.cuda.memory_allocated(index)
            reserved = torch.cuda.memory_reserved(index) if hasattr(torch.cuda, "memory_reserved") else 0
            return int(max(allocated, reserved))
        except Exception:
            return None


def _xpu_memory_used_bytes(index: int) -> Optional[int]:
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not hasattr(xpu, "is_available") or not xpu.is_available():
        return None
    try:
        if hasattr(xpu, "mem_get_info"):
            free, total = xpu.mem_get_info(index)
            return int(total - free)
    except Exception:
        pass
    try:
        allocated = xpu.memory_allocated(index)
        reserved = xpu.memory_reserved(index) if hasattr(xpu, "memory_reserved") else 0
        return int(max(allocated, reserved))
    except Exception:
        return None


def _accelerator_memory_used_bytes(device_type: str, index: int) -> Optional[int]:
    if device_type in {"cuda", "rocm"}:
        return _cuda_memory_used_bytes(index)
    if device_type == "xpu":
        return _xpu_memory_used_bytes(index)
    return None


class Device:
    """
    Lightweight replacement for the external `device_smi.Device`.
    Provides a compatible interface (metrics/close) used for logging.
    """

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id

    def metrics(self, fast: bool = True):
        if self.device_id == "cpu":
            used_bytes = _cpu_memory_used_bytes()
            return SimpleNamespace(memory_used=float(used_bytes))

        if ":" in self.device_id:
            device_type, index_str = self.device_id.split(":", 1)
            try:
                index = int(index_str)
            except ValueError as exc:
                raise RuntimeError(f"Invalid device identifier `{self.device_id}`.") from exc
        else:
            device_type = self.device_id
            index = 0

        used_bytes = _accelerator_memory_used_bytes(device_type, index)
        if used_bytes is None:
            raise RuntimeError(f"Metrics not available for device `{self.device_id}`.")
        return SimpleNamespace(memory_used=float(used_bytes))

    def close(self) -> None:
        # External handle cleanup is not required for the lightweight implementation.
        return
