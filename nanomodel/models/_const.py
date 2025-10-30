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
ROCM = device("cuda:0")  # rocm maps to fake cuda

# Modules frequently quantized by NanoModel; order matters for downstream pattern matching.
SUPPORTS_MODULE_TYPES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    transformers.Conv1D,
)

DEFAULT_MAX_SHARD_SIZE = "4GB"


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

    has_required_capability = any(
        torch.cuda.get_device_capability(index)[0] >= _CUDA_MINIMUM_MAJOR for index in range(device_count)
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
        if HAS_CUDA:
            return DEVICE.CUDA
        if HAS_XPU:
            return DEVICE.XPU
        if HAS_MPS:
            return DEVICE.MPS
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
    if HAS_CUDA:
        return CUDA_0
    if HAS_XPU:
        return XPU_0
    if HAS_MPS:
        return MPS
    return CPU

EXPERT_INDEX_PLACEHOLDER = "{expert_index}"

# Concatenation token used while constructing calibration payloads.
CALIBRATION_DATASET_CONCAT_CHAR = " "
