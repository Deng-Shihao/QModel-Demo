"""NanoModel quantization toolkit."""
import os

from .utils.env import env_flag


DEBUG_ON = env_flag("DEBUG")

from .utils.linalg_warmup import run_torch_linalg_warmup
from .utils.threadx import DeviceThreadPool


DEVICE_THREAD_POOL = DeviceThreadPool(
    inference_mode=True,
    warmups={
        "cuda": run_torch_linalg_warmup,
        "xpu": run_torch_linalg_warmup,
        "mps": run_torch_linalg_warmup,
        "cpu": run_torch_linalg_warmup,
    },
    workers={
        "cuda:per": 4,
        "xpu:per": 1,
        "mps": 8,
        "cpu": min(12, max(1, (os.cpu_count() or 1) // 2)),
        "model_loader:cpu": 2,
    },
    empty_cache_every_n=512,
)

from .models import AutoNanoModel, get_best_device
from .quantization import BaseQuantizeConfig, QuantizeConfig
from .utils import BACKEND
from .version import __version__


if os.getenv('GPTQMODEL_USE_MODELSCOPE', 'False').lower() in ['true', '1']:
    try:
        from modelscope.utils.hf_util.patcher import patch_hub
        patch_hub()
    except Exception:
        raise ModuleNotFoundError("you have set GPTQMODEL_USE_MODELSCOPE env, but doesn't have modelscope? install it with `pip install modelscope`")

