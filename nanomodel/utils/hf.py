import json
from typing import Any, Optional

import torch
from accelerate import init_empty_weights
from transformers import GenerationConfig, PreTrainedModel

from ..utils.logger import setup_logger


log = setup_logger()

GENERATION_SAMPLING_FIELDS = ("temperature", "top_p")


def _sanitize_generation_config(cfg: GenerationConfig, *, drop_sampling_fields: bool = False) -> bool:
    changed = False
    if cfg is None:
        return changed

    if getattr(cfg, "do_sample", None) is not True:
        cfg.do_sample = True
        changed = True

    if drop_sampling_fields:
        for field in GENERATION_SAMPLING_FIELDS:
            if hasattr(cfg, field):
                if getattr(cfg, field) is not None:
                    changed = True
                setattr(cfg, field, None)
    return changed


def _load_sanitized_generation_config(path: str) -> Optional[GenerationConfig]:
    try:
        config_dict, kwargs = GenerationConfig.get_config_dict(path)
    except Exception:
        return None

    cleaned = dict(config_dict)
    changed = False
    if cleaned.get("do_sample") is not True:
        cleaned["do_sample"] = True
        changed = True

    cfg = GenerationConfig.from_dict(cleaned, **kwargs)
    if changed:
        log.info("Model: Normalized `generation_config.json` while loading (`do_sample=True`).")
    _sanitize_generation_config(cfg, drop_sampling_fields=False)
    return cfg


def _resolve_pad_token_id(model: PreTrainedModel) -> Optional[int]:
    tokenizer = getattr(model, "tokenizer", None)
    config = getattr(model, "config", None)
    candidates = (
        getattr(config, "pad_token_id", None),
        getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None,
        getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None,
        getattr(config, "eos_token_id", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


# TODO FIXME! Pre-quantized use AutoModelForCausalLM.from_pretrained() but post-quantized use AutoModelForCausalLM.from_config()
def autofix_hf_model_config(model: PreTrainedModel, path: str = None):
    if model.can_generate():
        # sync config first
        if path:
            log.info(f"Model: Loaded `generation_config`: {model.generation_config}")
            try:
                cfg = _load_sanitized_generation_config(path)
                if cfg is None:
                    cfg = GenerationConfig.from_pretrained(pretrained_model_name=path, do_sample=True)
                    _sanitize_generation_config(cfg, drop_sampling_fields=False)
                if cfg != model.generation_config:
                    # migrated pad_token_id to config
                    if hasattr(model.generation_config, "pad_token_id"):
                        cfg.pad_token_id = model.generation_config.pad_token_id

                    model.generation_config = cfg
                    log.info(
                        "Model: Auto-fixed `generation_config` mismatch between model and `generation_config.json`.")
                    log.info(f"Model: Updated `generation_config`: {model.generation_config}")
                else:
                    pass
                    # logger.info(f"Model: loaded `generation_config` matching `generation_config.json`.")
            except Exception:
                log.info("Model: `generation_config.json` not found. Skipped checking.")

        # print(f"Before autofix_hf_model_config: {model.generation_config}")
        autofix_hf_generation_config(model.generation_config)
        pad_token_id = _resolve_pad_token_id(model)
        if pad_token_id is not None and getattr(model.generation_config, "pad_token_id", None) != pad_token_id:
            model.generation_config.pad_token_id = pad_token_id
            log.info("Model: Synced `pad_token_id=%s` into `generation_config`.", pad_token_id)
        # print(f"After autofix_hf_model_config: {model.generation_config}")


def autofix_hf_generation_config(cfg: GenerationConfig):
    _sanitize_generation_config(cfg, drop_sampling_fields=False)
    # HF has recently started to perform very strict validation model save which results in warnings on load()
    # to become exceptions on save().
    if cfg.do_sample is False:
        errors = 0
        if hasattr(cfg, "temperature") and cfg.temperature is not None and cfg.temperature != 1.0:
            errors += 1
        if hasattr(cfg, "top_p") and cfg.top_p is not None and cfg.top_p != 1.0:
            errors += 1
        if hasattr(cfg, "min_p") and cfg.min_p is not None:
            errors += 1
        if hasattr(cfg, "typical_p") and cfg.typical_p is not None and cfg.typical_p != 1.0:
            errors += 1
        # contrastive search uses top_k
        if (hasattr(cfg, "top_k") and cfg.top_k is not None and cfg.top_k != 50) and (hasattr(cfg, "penalty_alpha") and cfg.penalty_alpha is None):
            errors += 1
        if hasattr(cfg, "epsilon_cutoff") and cfg.epsilon_cutoff is not None and cfg.epsilon_cutoff != 0.0:
            errors += 1
        if hasattr(cfg, "eta_cutoff") and cfg.eta_cutoff is not None and cfg.eta_cutoff != 0.0:
            errors += 1

        # fix wrong do_sample
        if errors > 0:
            cfg.do_sample = True
            log.info("Model: Auto-Fixed `generation_config` by setting `do_sample=True`.")


def sanitize_generation_config_file(path: str, cfg: Optional[GenerationConfig] = None) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return False

    changed = False

    if cfg is not None:
        required_fields = {
            "pad_token_id": getattr(cfg, "pad_token_id", None),
            "temperature": getattr(cfg, "temperature", None),
            "top_k": getattr(cfg, "top_k", None),
        }
        for field, value in required_fields.items():
            if value is None:
                continue
            if data.get(field) != value:
                data[field] = value
                changed = True

    if data.get("do_sample") is not True:
        data["do_sample"] = True
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
                config,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                **init_kwargs
            )
    finally:
        pb.close()

    return shell
