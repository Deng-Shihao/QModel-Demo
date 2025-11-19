from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import transformers
from torch import device

from ..utils import BACKEND
from ..utils.rocm import IS_ROCM
from ..utils.torch import HAS_CUDA, HAS_MPS, HAS_XPU


CPU = device("cpu")
META = device("meta")
CUDA = device("cuda")
CUDA_0 = device("cuda:0")
XPU = device("xpu")
XPU_0 = device("xpu:0")
MPS = device("mps")
ROCM = CUDA_0  # rocm maps to fake cuda

__all__ = [
    "CALIBRATION_DATASET_CONCAT_CHAR",
    "CPU",
    "CUDA",
    "CUDA_0",
    "DEFAULT_MAX_SHARD_SIZE",
    "DEVICE",
    "EXPERT_INDEX_PLACEHOLDER",
    "META",
    "MPS",
    "PLATFORM",
    "ROCM",
    "SUPPORTS_MODULE_TYPES",
    "XPU",
    "XPU_0",
    "get_best_device",
    "normalize_device",
    "validate_cuda_support",
]

# Module type patterns shipped to quantization passes; order matters downstream.
SUPPORTS_MODULE_TYPES: tuple[type[nn.Module], ...] = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    transformers.Conv1D,
)

DEFAULT_MAX_SHARD_SIZE = "4GB"

# Shared accelerator priority; keeps `normalize_device` and `get_best_device` aligned.
_ACCELERATOR_DEVICE_PRIORITY: tuple[tuple[bool, torch.device], ...] = (
    (HAS_CUDA, CUDA_0),
    (HAS_XPU, XPU_0),
    (HAS_MPS, MPS),
)


class DEVICE(str, Enum):
    ALL = "all"  # All device
    CPU = "cpu"  # All CPU: Optimized for IPEX is CPU has AVX, AVX512, AMX, or XMX instructions
    CUDA = "cuda"  # Nvidia GPU: Optimized for Ampere+
    XPU = "xpu"  # Intel GPU: Datacenter Max + Arc
    MPS = "mps"  # MacOS GPU: Apple Silicon/Metal
    ROCM = "rocm"  # AMD GPU: ROCm maps to fake cuda

    @classmethod
    def _missing_(cls, value):
        """Handle case-insensitive ROCm aliases when initialising the enum."""
        if IS_ROCM and f"{value}".lower() == "rocm":
            return cls.ROCM
        return super()._missing_(value)

    @property
    def type(self) -> str:
        """Return the backend type compatible with torch.device semantics."""
        if self == DEVICE.ROCM:
            return "cuda"
        return self.value

    @property
    def index(self) -> int | None:
        """Default index used when materialising a torch.device from this enum."""
        if self in (DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU):
            return 0
        return None

    def to_torch_device(self) -> torch.device:
        """Convert the enum to a concrete torch.device, defaulting to index 0."""
        idx = self.index
        return torch.device(self.type if idx is None else f"{self.type}:{idx}")

    def to_device_map(self) -> dict[str, DEVICE]:
        """Create a `load_in_8bit`-style map understood by HF accelerate pipelines."""
        return {"": DEVICE.CUDA if self == DEVICE.ROCM else self}


# Mirror of `_ACCELERATOR_DEVICE_PRIORITY` expressed in enum form.
_ACCELERATOR_ENUM_PRIORITY: tuple[tuple[bool, DEVICE], ...] = (
    (HAS_CUDA, DEVICE.CUDA),
    (HAS_XPU, DEVICE.XPU),
    (HAS_MPS, DEVICE.MPS),
)


class PLATFORM(str, Enum):
    ALL = "all"  # All platform
    LINUX = "linux"  # linux
    WIN32 = "win32"  # windows
    DARWIN = "darwin"  # macos


_CUDA_MINIMUM_MAJOR = 6
_CUDA_CAPABILITY_ERROR = (
    "NanoModel cuda requires Pascal or later gpu with compute capability >= `6.0`."
)


def validate_cuda_support(raise_exception: bool = False) -> bool:
    """Verify CUDA availability and minimal compute capability support."""
    if not HAS_CUDA:
        return False

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return False

    # Query each GPU since heterogeneous systems may mix capability levels.
    has_required_capability = any(
        torch.cuda.get_device_capability(index)[0] >= _CUDA_MINIMUM_MAJOR
        for index in range(device_count)
    )
    if not has_required_capability:
        if raise_exception:
            raise EnvironmentError(_CUDA_CAPABILITY_ERROR)
        return False

    return True


def normalize_device(type_value: str | DEVICE | int | torch.device) -> DEVICE:
    """Normalise heterogeneous user input into a `DEVICE` enum."""
    if isinstance(type_value, DEVICE):
        return type_value

    if isinstance(type_value, int):
        # Pick the best accelerator available when users pass an index without a type.
        for has_hw, device_enum in _ACCELERATOR_ENUM_PRIORITY:
            if has_hw:
                return device_enum
        return DEVICE.CPU

    if isinstance(type_value, torch.device):
        type_value = type_value.type

    if not isinstance(type_value, str):
        raise ValueError(f"Invalid device type_value type: {type(type_value)}")

    # Remove explicit indices like `cuda:1` before enum conversion.
    normalized = type_value.split(":", 1)[0].strip()
    return DEVICE(normalized.lower())


def get_best_device(backend: BACKEND = BACKEND.AUTO) -> torch.device:
    """Select the most capable local device; `backend` is reserved for future routing."""
    # Traverse hardware priority list so that future accelerators can slot in easily.
    for has_hw, torch_device in _ACCELERATOR_DEVICE_PRIORITY:
        if has_hw:
            return torch_device

    return CPU


EXPERT_INDEX_PLACEHOLDER = "{expert_index}"

# Concatenation token used while constructing calibration payloads.
CALIBRATION_DATASET_CONCAT_CHAR = " "
