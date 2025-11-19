from __future__ import annotations

import os
from typing import Optional, Union

import torch
from torch import nn as nn

from ..models._const import CPU, CUDA_0

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


_BYTES_IN_GIB = 1024**3


def _bytes_to_gib(value: float | int) -> float:
    return float(value) / _BYTES_IN_GIB


def _get_cuda_index(device: torch.device) -> int:
    if device.index is not None:
        return device.index
    try:
        return torch.cuda.current_device()
    except RuntimeError:
        return 0


def _cpu_memory_used_bytes() -> int:
    if psutil is not None:
        mem = psutil.virtual_memory()
        return int(mem.total - mem.available)

    if os.name == "posix":
        meminfo_path = "/proc/meminfo"
        if os.path.exists(meminfo_path):
            totals: dict[str, int] = {}
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
                return total - available

    # Fallback: return process resident memory as an approximation.
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports ru_maxrss in bytes, Linux in kilobytes.
        rss = usage.ru_maxrss
        if os.name == "posix":
            try:
                sysname = os.uname().sysname.lower()
            except AttributeError:  # pragma: no cover - platform specific
                sysname = ""
            if "darwin" not in sysname:
                rss *= 1024
        return int(rss)
    except Exception:  # pragma: no cover - best effort fallback
        return 0


def _gpu_memory_used_bytes(device: torch.device) -> int:
    if not torch.cuda.is_available():
        return 0

    idx = _get_cuda_index(device)
    try:
        free, total = torch.cuda.mem_get_info(idx)
        return int(total - free)
    except AttributeError:
        pass  # older torch without mem_get_info
    except RuntimeError:
        # initialise context if needed and retry once
        try:
            torch.cuda.set_device(idx)
            free, total = torch.cuda.mem_get_info(idx)
            return int(total - free)
        except Exception:
            pass

    try:
        allocated = torch.cuda.memory_allocated(idx)
    except RuntimeError:
        return 0
    reserved = (
        torch.cuda.memory_reserved(idx) if hasattr(torch.cuda, "memory_reserved") else 0
    )
    return int(max(allocated, reserved))


# unit: GiB
def get_gpu_usage_memory():
    return _bytes_to_gib(_gpu_memory_used_bytes(CUDA_0))


# unit: GiB
def get_cpu_usage_memory():
    return _bytes_to_gib(_cpu_memory_used_bytes())


def get_cpu_concurrency() -> int:
    """
    Return the total number of logical CPU cores available.

    Mirrors the previous `Device('cpu').count * Device('cpu').cores` behaviour.
    """
    concurrency = os.cpu_count() or 1
    return max(1, concurrency)


def get_device(obj: torch.Tensor | nn.Module) -> torch.device:
    if isinstance(obj, torch.Tensor):
        return obj.device

    params = list(obj.parameters())
    buffers = list(obj.buffers())
    if len(params) > 0:
        return params[0].device
    elif len(buffers) > 0:
        return buffers[0].device
    else:
        return CPU


def get_device_new(
    obj: torch.Tensor | nn.Module,
    recursive: bool = False,
    assert_mode: bool = False,
    expected: Optional[Union[str, torch.device]] = None,
    check_index: bool = False,
) -> torch.device:
    """
    Return a representative device for a Tensor/Module and optionally assert uniformity.

    Args:
        obj: Tensor or nn.Module.
        recursive: If obj is an nn.Module, traverse submodules (parameters/buffers)
                   recursively (like module.parameters(recurse=True)).
        assert_mode: If True, perform assertions about device placement:
            - If `expected` is provided: assert that ALL params/buffers live on a device
              whose .type matches `expected`'s .type (and, if check_index, the same index).
            - If `expected` is None: assert that ALL params/buffers share a single uniform
              device type (and, if check_index, the same index).
        expected: A target device or device string (e.g., "cpu", "cuda", "cuda:1").
        check_index: If True, also require the same device index (e.g., all on cuda:0).

    Returns:
        torch.device: A representative device. Priority order:
            - Tensor: its own device
            - Module: the first parameter device, else first buffer device, else CPU
    """

    # --- Helper to normalize an "expected" device to (type, index) ---
    def _normalize_expected(exp: Optional[Union[str, torch.device]]):
        if exp is None:
            return None, None
        dev = torch.device(exp) if isinstance(exp, str) else exp
        return dev.type, dev.index

    # --- Collect devices present on the object ---
    if isinstance(obj, torch.Tensor):
        devices = [obj.device]
    elif isinstance(obj, nn.Module):
        # Pull parameters/buffers; recurse if requested
        params = list(obj.parameters(recurse=recursive))
        buffs = list(obj.buffers(recurse=recursive))
        devices = []
        if params:
            devices.extend(p.device for p in params)
        if buffs:
            devices.extend(b.device for b in buffs)
        if not devices:
            devices = [CPU]
    else:
        raise TypeError(f"get_device() expects Tensor or nn.Module, got {type(obj)}")

    # Representative device (keep legacy behavior)
    rep_device = devices[0]

    # --- Assertions (if requested) ---
    if assert_mode:
        exp_type, exp_index = _normalize_expected(expected)

        def _key(d: torch.device):
            return (d.type, d.index if check_index else None)

        if exp_type is not None:
            # Check against expected device TYPE (and optionally INDEX)
            mismatches = [
                d
                for d in devices
                if d.type != exp_type or (check_index and d.index != exp_index)
            ]
            if mismatches:
                # Build a concise error message with a few examples
                sample = ", ".join({f"{d.type}:{d.index}" for d in mismatches[:5]})
                target = f"{exp_type}" + (f":{exp_index}" if check_index else "")
                raise AssertionError(
                    f"Device assertion failed: expected all tensors on {target}, "
                    f"but found mismatches (e.g., {sample}). Total tensors checked: {len(devices)}."
                )
        else:
            # Ensure uniformity across all devices (by type, and optionally index)
            unique = {_key(d) for d in devices}
            if len(unique) > 1:
                # Summarize what we actually found
                summary = ", ".join(sorted(f"{t}:{i}" for (t, i) in unique))
                detail = ", ".join({f"{d.type}:{d.index}" for d in devices[:8]})
                msg = (
                    "Device assertion failed: tensors are on multiple devices. "
                    f"Found {{{summary}}}. Examples: {detail}."
                )
                if not check_index:
                    msg += " (Tip: set check_index=True to also require same device index.)"
                raise AssertionError(msg)

    return rep_device
