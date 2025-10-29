from __future__ import annotations

import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union

import numpy
import torch
from huggingface_hub import list_repo_files
from transformers import AutoConfig

from ..quantization import METHOD, QUANT_CONFIG_FILENAME
from ..utils import BACKEND
from ..utils.logger import setup_logger
from .base import BaseNanoModel, QuantizeConfig
from .definitions.llama import LlamaNanoModel
from .definitions.qwen2 import Qwen2NanoModel
from .definitions.qwen3 import Qwen3NanoModel

__all__ = ["AutoNanoModel"]

log = setup_logger()

_DEFAULT_SEED = 233
_QUANT_CONFIG_CANDIDATES = (QUANT_CONFIG_FILENAME, "quant_config.json")


def _ensure_runtime_env() -> None:
    """Populate environment defaults required for consistent runtime behavior."""
    if os.environ.get("PYTORCH_ALLOC_CONF") is None:
        os.environ["PYTORCH_ALLOC_CONF"] = (
            "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7"
        )

    if os.environ.get("CUDA_DEVICE_ORDER") is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if "CUDA_VISIBLE_DEVICES" in os.environ and "ROCR_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)


def _enable_mps_fallback_if_needed() -> None:
    """Enable MPS fallback on macOS to cover missing aten ops."""
    if sys.platform != "darwin":
        return

    fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback_env and fallback_env.lower() in {"1", "true", "yes", "on"}:
        return

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    log.info(
        "ENV: Auto enabling PYTORCH_ENABLE_MPS_FALLBACK=1 to cover missing MPS aten ops."
    )


def _seed_rng(seed: int = _DEFAULT_SEED) -> None:
    """Seed RNGs to keep quantization and inference deterministic."""
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def _resolve_backend(backend: Union[str, BACKEND]) -> BACKEND:
    """Normalize backend values to the enum."""
    return backend if isinstance(backend, BACKEND) else BACKEND(backend)


@lru_cache(maxsize=None)
def _list_remote_files(repo_id: str) -> set[str]:
    """Cache huggingface repo file listings to limit API calls."""
    return set(list_repo_files(repo_id=repo_id))


def _has_local_quant_config(model_path: Path) -> bool:
    return any((model_path / name).exists() for name in _QUANT_CONFIG_CANDIDATES)


def _has_remote_quant_config(model_id: str) -> bool:
    files = _list_remote_files(repo_id=model_id)
    return any(name in files for name in _QUANT_CONFIG_CANDIDATES)


def _is_model_quantized(model_id_or_path: str, *, config: AutoConfig) -> bool:
    """Determine whether the model has been quantized by NanoModel-compatible tooling."""
    quant_cfg = getattr(config, "quantization_config", None)
    if quant_cfg and isinstance(quant_cfg, dict):
        quant_format = quant_cfg.get("quant_format", "").lower()
        return quant_format in {METHOD.GPTQ, METHOD.AWQ}

    model_path = Path(model_id_or_path)
    if model_path.is_dir():
        return _has_local_quant_config(model_path)

    try:
        return _has_remote_quant_config(model_id_or_path)
    except Exception:  # pragma: no cover - networking edge cases
        log.debug(
            "Falling back to AutoConfig quantization detection for %s build; "
            "list_repo_files lookup failed.",
            model_id_or_path,
        )
        return False


def check_and_get_model_type(
    model_dir: str, trust_remote_code: bool = False, *, config: Optional[AutoConfig] = None
) -> str:
    """Validate model type support and return a normalized key."""
    cfg = config or AutoConfig.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code
    )
    model_type = cfg.model_type.lower()
    if model_type not in MODEL_DICT:
        raise TypeError(f"{cfg.model_type} isn't supported yet.")
    return model_type


_ensure_runtime_env()
_enable_mps_fallback_if_needed()
_seed_rng()

MODEL_DICT = {
    "llama": LlamaNanoModel,
    "qwen2": Qwen2NanoModel,
    "qwen3": Qwen3NanoModel,
}


class AutoNanoModel:
    """Factory helpers that pick the correct NanoModel implementation for a given checkpoint."""
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
        **kwargs,
    ):
        """Automatically choose between quantized and pre-quantized loading."""
        if not model_id_or_path:
            raise ValueError("`model_id_or_path` is required to load a model.")

        if isinstance(model_id_or_path, str):
            model_id_or_path = model_id_or_path.strip()
            if not model_id_or_path:
                raise ValueError("`model_id_or_path` cannot be an empty string.")

        # normalize config to cfg instance
        if isinstance(quantize_config, Dict):
            quantize_config = QuantizeConfig(**quantize_config)

        backend_enum = _resolve_backend(backend)

        model_cfg = AutoConfig.from_pretrained(
            model_id_or_path, trust_remote_code=trust_remote_code
        )
        is_model_quantized = _is_model_quantized(
            model_id_or_path=model_id_or_path,
            config=model_cfg,
        )

        if is_model_quantized:
            model = cls.from_quantized(
                model_id_or_path=model_id_or_path,
                device_map=device_map,
                device=device,
                backend=backend_enum,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        else:
            model = cls.from_pretrained(
                model_id_or_path=model_id_or_path,
                quantize_config=quantize_config,
                device_map=device_map,
                device=device,
                trust_remote_code=trust_remote_code,
                backend=backend_enum,
                **kwargs,
            )

        return model

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        quantize_config: Optional[QuantizeConfig] = None,
        trust_remote_code: bool = False,
        **model_init_kwargs,
    ) -> BaseNanoModel:
        """Load a pretrained model and prepare it for quantization."""
        backend_enum = _resolve_backend(model_init_kwargs.pop("backend", BACKEND.AUTO))

        cfg = AutoConfig.from_pretrained(
            model_id_or_path, trust_remote_code=trust_remote_code
        )

        if getattr(cfg, "quantization_config", None):
            log.warning(
                "As the model is already quantized, loading will be done via from_quantized.\n"
                "To apply quantization yourself, provide an unquantized model path or ID, and load it using from_pretrained with quantize_config."
            )
            fallback_kwargs = dict(model_init_kwargs)
            device_map = fallback_kwargs.pop("device_map", None)
            device = fallback_kwargs.pop("device", None)
            return cls.from_quantized(
                model_id_or_path=model_id_or_path,
                device_map=device_map,
                device=device,
                backend=backend_enum,
                trust_remote_code=trust_remote_code,
                **fallback_kwargs,
            )

        if quantize_config is None:
            raise ValueError("QuantizeConfig must be provided when loading from pretrained weights.")

        if quantize_config and quantize_config.dynamic:
            log.warning(
                "Full support for NanoModel’s per-module dynamic quantization is now included "
                "in the latest vLLM and SGLang, but hf transformers have not yet added this capability."
            )

        model_type = check_and_get_model_type(
            model_id_or_path, trust_remote_code, config=cfg
        )
        return MODEL_DICT[model_type].from_pretrained(
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
        """Load a quantized NanoModel checkpoint."""
        cfg = AutoConfig.from_pretrained(
            model_id_or_path, trust_remote_code=trust_remote_code
        )
        model_type = check_and_get_model_type(
            model_id_or_path, trust_remote_code, config=cfg
        )

        return MODEL_DICT[model_type].from_quantized(
            model_id_or_path=model_id_or_path,
            device_map=device_map,
            device=device,
            backend=_resolve_backend(backend),
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
