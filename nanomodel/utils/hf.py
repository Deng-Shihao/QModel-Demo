import json
import os
from typing import Any, Optional

import torch
from accelerate import init_empty_weights
from transformers import GenerationConfig, PreTrainedModel

from ..utils.logger import setup_logger


log = setup_logger()

GENERATION_SAMPLING_DEFAULTS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "typical_p": 1.0,
    "min_p": 0.0,
    "epsilon_cutoff": 0.0,
    "eta_cutoff": 0.0,
}


def _has_active_sampling_settings(source: Any) -> bool:
    for field, default in GENERATION_SAMPLING_DEFAULTS.items():
        value = source.get(field) if isinstance(source, dict) else getattr(source, field, None)
        if value is None:
            continue
        if isinstance(default, (int, float)) and isinstance(value, (int, float)) and value == default:
            continue
        return True
    return False


def _sanitize_generation_config(cfg: GenerationConfig) -> bool:
    changed = False
    if cfg is None:
        return changed

    # Keep sampling knobs intact but make sure do_sample lines up.
    if getattr(cfg, "do_sample", None) is False and _has_active_sampling_settings(cfg):
        cfg.do_sample = True
        changed = True
    return changed


def _load_sanitized_generation_config(path: str) -> Optional[GenerationConfig]:
    try:
        config_dict, kwargs = GenerationConfig.get_config_dict(path)
    except Exception:
        return None

    cfg = GenerationConfig.from_dict(dict(config_dict), **kwargs)
    if _sanitize_generation_config(cfg):
        log.info(
            "Model: Normalized `generation_config` during load to keep sampling settings consistent."
        )
    return cfg


# TODO FIXME! Pre-quantized use AutoModelForCausalLM.from_pretrained() but post-quantized use AutoModelForCausalLM.from_config()
def autofix_hf_model_config(model: PreTrainedModel, path: str = None):
    if model.can_generate():
        # sync config first
        if path:
            log.info(f"Model: Loaded `generation_config`: {model.generation_config}")
            try:
                cfg = _load_sanitized_generation_config(path)
                if cfg is None:
                    cfg = GenerationConfig.from_pretrained(
                        pretrained_model_name=path, do_sample=True
                    )
                    _sanitize_generation_config(cfg)
                if cfg != model.generation_config:
                    # migrated pad_token_id to config
                    if hasattr(model.generation_config, "pad_token_id"):
                        cfg.pad_token_id = model.generation_config.pad_token_id

                    model.generation_config = cfg
                    log.info(
                        "Model: Auto-fixed `generation_config` mismatch between model and `generation_config.json`."
                    )
                    log.info(
                        f"Model: Updated `generation_config`: {model.generation_config}"
                    )
                else:
                    pass
                    # logger.info(f"Model: loaded `generation_config` matching `generation_config.json`.")
            except Exception:
                log.info("Model: `generation_config.json` not found. Skipped checking.")

        # print(f"Before autofix_hf_model_config: {model.generation_config}")
        autofix_hf_generation_config(model.generation_config)
        # print(f"After autofix_hf_model_config: {model.generation_config}")


def autofix_hf_generation_config(cfg: GenerationConfig):
    _sanitize_generation_config(cfg)
    # HF has recently started to perform very strict validation model save which results in warnings on load()
    # to become exceptions on save().
    if cfg.do_sample is False:
        errors = 0
        if (
            hasattr(cfg, "temperature")
            and cfg.temperature is not None
            and cfg.temperature != 1.0
        ):
            errors += 1
        if hasattr(cfg, "top_p") and cfg.top_p is not None and cfg.top_p != 1.0:
            errors += 1
        if hasattr(cfg, "min_p") and cfg.min_p is not None:
            errors += 1
        if (
            hasattr(cfg, "typical_p")
            and cfg.typical_p is not None
            and cfg.typical_p != 1.0
        ):
            errors += 1
        # contrastive search uses top_k
        if (hasattr(cfg, "top_k") and cfg.top_k is not None and cfg.top_k != 50) and (
            hasattr(cfg, "penalty_alpha") and cfg.penalty_alpha is None
        ):
            errors += 1
        if (
            hasattr(cfg, "epsilon_cutoff")
            and cfg.epsilon_cutoff is not None
            and cfg.epsilon_cutoff != 0.0
        ):
            errors += 1
        if (
            hasattr(cfg, "eta_cutoff")
            and cfg.eta_cutoff is not None
            and cfg.eta_cutoff != 0.0
        ):
            errors += 1

        # fix wrong do_sample
        if errors > 0:
            cfg.do_sample = True
            log.info(
                "Model: Auto-Fixed `generation_config` by setting `do_sample=True`."
            )


def sanitize_generation_config_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return False

    changed = False
    if _has_active_sampling_settings(data) and data.get("do_sample") is False:
        data["do_sample"] = True
        changed = True

    if data.get("pad_token_id") is None:
        config_path = os.path.join(os.path.dirname(path), "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as cfg_fp:
                cfg_data = json.load(cfg_fp)
                pad_token_id = cfg_data.get("pad_token_id") or cfg_data.get(
                    "eos_token_id"
                )
        except (OSError, json.JSONDecodeError):
            pad_token_id = None

        if pad_token_id is not None:
            data["pad_token_id"] = pad_token_id
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

    return changed


# load hf model with empty tensors on meta device (zero tensor memory usage)
def build_shell_model(
    loader,
    config: Any,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    **model_init_kwargs,
):
    """
    Instantiate the HF architecture with all parameters and buffers on 'meta' (no CPU RAM).
    Preserves the full module topology (Linear/MLP/Attention/etc.).

    Args:
        model_id_or_path: Hugging Face model ID or local path.
        dtype: Target dtype for model parameters (replaces `torch_dtype`).
        trust_remote_code: Allow loading custom model classes.
    """
    init_kwargs = model_init_kwargs.copy()

    del init_kwargs["device_map"]
    del init_kwargs["_fast_init"]
    # All nn.Parameters and buffers are created

    # All nn.Parameters and buffers are created on 'meta' and initializers are skipped.
    pb = log.spinner(title="Model loading...", interval=0.1)
    try:
        with init_empty_weights(include_buffers=True):
            shell = loader.from_config(
                config, dtype=dtype, trust_remote_code=trust_remote_code, **init_kwargs
            )
    finally:
        pb.close()

    return shell
