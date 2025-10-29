import contextlib
import time
from contextlib import contextmanager
from typing import Callable, List, Optional, Union

import torch
from packaging import version
from torch.cpu import StreamContext

from ..utils.logger import setup_logger
from ..utils.safe import GC
from . import gte_python_3_13_3, gte_python_3_14, has_gil_disabled, log_gil_requirements_for


_TORCH_VERSION = version.parse(torch.__version__)

# PyTorch 2.6.0 fixes many compilation errors affecting nano model builds.
TORCH_HAS_COMPILE = _TORCH_VERSION >= version.Version("2.6")
TORCH_GTE_28 = _TORCH_VERSION >= version.Version("2.8")
TORCH_GTE_210 = _TORCH_VERSION >= version.Version("2.10")

HAS_CUDA = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
HAS_XPU = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
HAS_MPS = bool(getattr(torch, "mps", None) and torch.mps.is_available())
HAS_NPU = bool(getattr(torch, "npu", None) and torch.npu.is_available())
HAS_MLX = False

CPU = torch.device("cpu")
META = torch.device("meta")

# Cache commonly reused stream so helper contexts can share it cheaply.
_STREAM_CACHE: Optional[Union[torch.cuda.Stream, "torch.xpu.Stream"]] = None

log = setup_logger()


def timed_gc_collect() -> int:
    """Run ``gc.collect`` and log the elapsed time along with reclaimed object count."""
    start = time.perf_counter()

    # Python 3.14 removed gen1 so there is only gen0 and gen2
    collected = GC.collect()

    duration = time.perf_counter() - start
    log.info(f"gc.collect() reclaimed {collected} objects in {duration:.3f}s")
    return collected

# reset dynamo cache on each model load since during ci loop model inference may exhuast cache
try:
    torch._dynamo.reset()
    # Increase the dynamo cache size limit, default of 8 is too low
    if torch._dynamo.config.cache_size_limit < 128:
        torch._dynamo.config.cache_size_limit = 128
except BaseException:
    # triton built from source maybe incompatible with _dynamo private api
    pass

# Track MLX support separately because it is decoupled from PyTorch.
mlx_core = None
try:
    import mlx.core as mlx_core
    HAS_MLX = True
except BaseException:
    mlx_core = None

BACKENDS_HAS_FP32_PRECISION = hasattr(torch.backends, "fp32_precision")


def _set_tf32_state(enabled: bool) -> None:
    if BACKENDS_HAS_FP32_PRECISION:
        mode = "tf32" if enabled else "ieee"
        torch.backends.fp32_precision = mode
        torch.backends.cuda.matmul.fp32_precision = mode
        torch.backends.cudnn.fp32_precision = mode
        torch.backends.cudnn.conv.fp32_precision = mode
        torch.backends.cudnn.rnn.fp32_precision = mode
        return

    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _snapshot_tf32_state():
    if BACKENDS_HAS_FP32_PRECISION:
        return (
            torch.backends.fp32_precision,
            torch.backends.cuda.matmul.fp32_precision,
            torch.backends.cudnn.fp32_precision,
            torch.backends.cudnn.conv.fp32_precision,
            torch.backends.cudnn.rnn.fp32_precision,
        )

    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )


def _restore_tf32_state(state) -> None:
    if BACKENDS_HAS_FP32_PRECISION:
        torch.backends.fp32_precision = state[0]
        torch.backends.cuda.matmul.fp32_precision = state[1]
        torch.backends.cudnn.fp32_precision = state[2]
        torch.backends.cudnn.conv.fp32_precision = state[3]
        torch.backends.cudnn.rnn.fp32_precision = state[4]
        return

    torch.backends.cuda.matmul.allow_tf32 = state[0]
    torch.backends.cudnn.allow_tf32 = state[1]


@contextmanager
def _tf32_state_guard(enabled: bool):
    if not HAS_CUDA:
        yield
        return

    previous_state = _snapshot_tf32_state()
    _set_tf32_state(enabled)
    try:
        yield
    finally:
        _restore_tf32_state(previous_state)


def torch_compile(
    module: Union[torch.nn.Module, Callable],
    backend: str = "inductor",
    mode: str = None,
    fullgraph: bool = False,
):
    """Invoke `torch.compile` when the runtime supports it, otherwise return the original module."""
    # Requires torch >= 2.8 for Python 3.13.3t (free-threaded) compatibility.
    if has_gil_disabled() and not gte_python_3_13_3():
        log_gil_requirements_for("Torch Compile")
        return module

    if gte_python_3_14() and not TORCH_GTE_210:
        log_gil_requirements_for("Torch Compile")
        return module

    if not TORCH_HAS_COMPILE:
        return module
    if HAS_MPS and not TORCH_GTE_28:
        if not torch._dynamo.config.suppress_errors:
            log.warn("To use compile() with MPS, you need to have torch version >= 2.8.0, "
                     "please upgrade it by `pip install -U torch torchaudio torchvision`")
            torch._dynamo.config.suppress_errors = True
        return module
    try:
        return torch.compile(module, backend=backend, mode=mode, fullgraph=fullgraph)
    except BaseException as e:
        log.warn.once(f"Failed to compile `{module}`, {e}")
        return module

def torch_new_stream(force_new: bool = False):
    """Return a cached accelerator stream, recreating it when `force_new` is True."""
    global _STREAM_CACHE

    if not force_new and _STREAM_CACHE is not None:
        return _STREAM_CACHE

    if HAS_CUDA:
        _STREAM_CACHE = torch.cuda.Stream()
    elif HAS_XPU:
        _STREAM_CACHE = torch.xpu.Stream()
    else:
        _STREAM_CACHE = None

    return _STREAM_CACHE


def torch_new_stream_ctx():
    """Context manager for the shared accelerator stream."""
    stream = torch_new_stream()

    if stream is None:
        return contextlib.nullcontext()

    if HAS_CUDA:
        return torch.cuda.stream(stream)
    if HAS_XPU:
        return torch.xpu.stream(stream)

    return contextlib.nullcontext()

def torch_sync(device: torch.device = None):
    """Synchronize accelerator queues.

    When no device is provided we synchronize every detected accelerator index so
    replication work staged on multiple GPUs/NPUs completes before issuing more
    kernels.
    """

    if device is None:
        synchronized_any = False

        for flag, backend in (
            (HAS_CUDA, getattr(torch, "cuda", None)),
            (HAS_XPU, getattr(torch, "xpu", None)),
            (HAS_NPU, getattr(torch, "npu", None)),
        ):
            if not flag or backend is None or not hasattr(backend, "device_count"):
                continue

            dev_count = backend.device_count()
            if not dev_count:
                continue

            synchronized_any = True
            for idx in range(dev_count):
                backend.synchronize(idx)

        if HAS_MPS and getattr(torch, "mps", None):
            torch.mps.synchronize()
            synchronized_any = True

        if not synchronized_any:
            torch.cpu.synchronize()
        return

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize(device=device)
    elif device.type == "cpu":
        torch.cpu.synchronize()

def torch_empty_cache(device: torch.device = None, gc: bool = True):
    """Clear per-backend allocator caches and optionally trigger Python GC."""
    if gc:
        timed_gc_collect()

    if device is None:
        for flag, backend, method in (
            (HAS_CUDA, getattr(torch, "cuda", None), "empty_cache"),
            (HAS_XPU, getattr(torch, "xpu", None), "empty_cache"),
            (HAS_MPS, getattr(torch, "mps", None), "empty_cache"),
        ):
            if flag and backend and hasattr(backend, method):
                getattr(backend, method)()

        if HAS_MLX and mlx_core is not None and hasattr(mlx_core, "clear_cache"):
            mlx_core.clear_cache()
        return

    # if device passed, only execute for device backend
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

        # mlx is detached from pytorch
        if HAS_MLX and mlx_core is not None and hasattr(mlx_core, "clear_cache"):
            mlx_core.clear_cache()

def auto_select_torch_device(index: int = 0):
    """Choose a torch.device for the requested index while handling out-of-range values."""
    assert index >= 0, f"device index should be a positive integer: actual = `{index}`"

    if HAS_CUDA:
        dev_count = torch.cuda.device_count()
        if index >= dev_count > 0:
            index = 0
        return torch.device(f"cuda:{index}")

    if HAS_XPU:
        dev_count = torch.xpu.device_count()
        if index >= dev_count > 0:
            index = 0
        return torch.device(f"xpu:{index}")

    if HAS_MPS:
        return torch.device("mps")  # MPS has no index concept.

    return CPU  # CPU has no index concept.

# some device types can have multiple gpus cuda/rocm + xpu
def torch_devices() -> List[torch.device]:
    if HAS_CUDA:
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    elif HAS_XPU:
        return [torch.device(f"xpu:{i}") for i in range(torch.xpu.device_count())]
    elif HAS_MPS:
        return [torch.device("mps")]
    else:
        return [CPU]

ALL_DEVICES = torch_devices()

if HAS_CUDA:
    ALL_STREAMS = [torch.cuda.Stream(device=device) for device in ALL_DEVICES]
elif HAS_XPU:
    ALL_STREAMS = [torch.xpu.Stream(device=device) for device in ALL_DEVICES]
else:
    ALL_STREAMS = [contextlib.nullcontext()]

DEVICE_0 = auto_select_torch_device(index=0)
# device_1 may be same as device_0 if there is only 1 visible/active device
DEVICE_1 = auto_select_torch_device(index=1)

DEVICE_0_STREAM = ALL_STREAMS[0]

def torch_stream_ctx(stream: Union[torch.cuda.Stream, "torch.xpu.Stream"]) -> StreamContext:
    """Enter the provided stream using the appropriate backend context manager."""
    if HAS_CUDA:
        return torch.cuda.stream(stream)
    if HAS_XPU:
        return torch.xpu.stream(stream)
    return contextlib.nullcontext()


# Backwards compatibility for older call sites.
torch_streamCtx = torch_stream_ctx


@contextmanager
def tf32_high_precision_guard():
    """Force IEEE precision by temporarily disabling TF32 kernels."""
    with _tf32_state_guard(False):
        yield


@contextmanager
def tf32_disable_guard():
    """Alias for `tf32_high_precision_guard` for compatibility."""
    with _tf32_state_guard(False):
        yield


@contextmanager
def tf32_enable_guard():
    """Temporarily enable TF32 kernels even if the global state disables them."""
    with _tf32_state_guard(True):
        yield
