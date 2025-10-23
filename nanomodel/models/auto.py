from __future__ import annotations

import os

from ..utils.logger import setup_logger

if not os.environ.get("PYTORCH_ALLOC_CONF", None):
    os.environ["PYTORCH_ALLOC_CONF"] = 'expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7'

if not os.environ.get("CUDA_DEVICE_ORDER", None):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

if 'CUDA_VISIBLE_DEVICES' in os.environ and 'ROCR_VISIBLE_DEVICES' in os.environ:
    del os.environ['ROCR_VISIBLE_DEVICES']


import os.path  # noqa: E402
import random  # noqa: E402
from os.path import isdir, join  # noqa: E402
from typing import Any, Dict, List, Optional, Type, Union  # noqa: E402

import numpy  # noqa: E402
import torch  # noqa: E402

from huggingface_hub import list_repo_files  # noqa: E402

from transformers import AutoConfig  # noqa: E402

from ..quantization import METHOD, QUANT_CONFIG_FILENAME  # noqa: E402
from ..utils import BACKEND  # noqa: E402
from .base import BaseNanoModel, QuantizeConfig  # noqa: E402

import sys  # noqa: E402
if sys.platform == "darwin":
    fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback_env is None or fallback_env.lower() not in {"1", "true", "yes", "on"}:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # log.info("ENV: Auto enabling PYTORCH_ENABLE_MPS_FALLBACK=1 to cover missing MPS aten ops.")
        print("ENV: Auto enabling PYTORCH_ENABLE_MPS_FALLBACK=1 to cover missing MPS aten ops.")


# Support Models
from .definitions.llama import LlamaNanoModel  # noqa: E402
from .definitions.qwen2 import Qwen2NanoModel  # noqa: E402
from .definitions.qwen3 import Qwen3NanoModel  # noqa: E402

# make quants and inference more determinisitc
torch.manual_seed(233)
random.seed(233)
numpy.random.seed(233)

MODEL_MAP = {
    "llama": LlamaNanoModel,
    "qwen2": Qwen2NanoModel, # Base on Llama
    "qwen3": Qwen3NanoModel # Base on Llama
}

SUPPORTED_MODELS = list(MODEL_MAP.keys())

def check_and_get_model_type(model_dir, trust_remote_code=False):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if config.model_type.lower() not in SUPPORTED_MODELS:
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type.lower()

class AutoNanoModel:
    def __init__(self):
        raise EnvironmentError(
            "NanoModel is not designed to be instantiated\n"
            "use `NanoModel.from_pretrained` to load pretrained model and prepare for quantization via `.quantize()`.\n"
            "use `NanoModel.from_quantized` to inference with post-quantized model."
        )

    @classmethod
    def load(
            cls,
            model_id_or_path: Optional[str],
            quantize_config: Optional[QuantizeConfig | Dict] = None,
            device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
            device: Optional[Union[str, torch.device]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            trust_remote_code: bool = False,
            debug: Optional[bool] = False,
            **kwargs,
    ):
        if isinstance(model_id_or_path, str):
            model_id_or_path = model_id_or_path.strip()

        # normalize config to cfg instance
        if isinstance(quantize_config, Dict):
            quantize_config = QuantizeConfig(**quantize_config)

        if isinstance(backend, str):
            backend = BACKEND(backend)

        is_model_quantized = False
        model_cfg = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
        if hasattr(model_cfg, "quantization_config") and "quant_format" in model_cfg.quantization_config:
            # only if the model is quantized or compatible with model should we set is_quantized to true
            if model_cfg.quantization_config["quant_format"].lower() in (METHOD.GPTQ, METHOD.AWQ):
                is_model_quantized = True
        else:
            for name in [QUANT_CONFIG_FILENAME, "quant_config.json"]:
                if isdir(model_id_or_path):  # Local
                    if os.path.exists(join(model_id_or_path, name)):
                        is_model_quantized = True
                        break

                else:  # Remote
                    files = list_repo_files(repo_id=model_id_or_path)
                    for f in files:
                        if f == name:
                            is_model_quantized = True
                            break

        if is_model_quantized:
            m = cls.from_quantized(
                model_id_or_path=model_id_or_path,
                device_map=device_map,
                device=device,
                backend=backend,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        else:
            m = cls.from_pretrained(
                model_id_or_path=model_id_or_path,
                quantize_config=quantize_config,
                device_map=device_map,
                device=device,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        # debug model structure
        # if debug:
        #     print_module_tree(m.model)

        return m


    @classmethod
    def from_pretrained(
            cls,
            model_id_or_path: str,
            quantize_config: QuantizeConfig,
            trust_remote_code: bool = False,
            **model_init_kwargs,
    ) -> BaseNanoModel:
        if hasattr(AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code),
                   "quantization_config"):
            print("Model is already quantized, will use `from_quantized` to load quantized model.\n"
                           "If you want to quantize the model, please pass un_quantized model path or id, and use "
                           "`from_pretrained` with `quantize_config`.""")
            # log.warn("Model is already quantized, will use `from_quantized` to load quantized model.\n"
            #                "If you want to quantize the model, please pass un_quantized model path or id, and use "
            #                "`from_pretrained` with `quantize_config`.")
            return cls.from_quantized(model_id_or_path, trust_remote_code=trust_remote_code)

        if quantize_config and quantize_config.dynamic:
            print("NanoModel's per-module `dynamic` quantization feature is fully supported in latest vLLM and SGLang but not yet available in hf transformers.")
            # log.warn(
            #     "NanoModel's per-module `dynamic` quantization feature is fully supported in latest vLLM and SGLang but not yet available in hf transformers.")

        model_type = check_and_get_model_type(model_id_or_path, trust_remote_code)
        return MODEL_MAP[model_type].from_pretrained(
            pretrained_model_id_or_path=model_id_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
            cls,
            model_id_or_path: Optional[str],
            device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
            device: Optional[Union[str, int]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            trust_remote_code: bool = False,
            **kwargs,
    ) -> BaseNanoModel:

        model_type = check_and_get_model_type(model_id_or_path, trust_remote_code)

        if isinstance(backend, str):
            backend = BACKEND(backend)

        return MODEL_MAP[model_type].from_quantized(
            model_id_or_path=model_id_or_path,
            device_map=device_map,
            device=device,
            backend=backend,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )