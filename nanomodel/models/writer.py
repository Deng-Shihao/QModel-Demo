from __future__ import annotations

import copy
import csv
import json

import logging
import os

import re
from os.path import isfile, join
from typing import Any, Dict, Optional, Union

import torch
import transformers
from transformers import AutoConfig, PreTrainedTokenizerFast, ProcessorMixin
from transformers.modeling_utils import no_init_weights
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils.generic import ContextManagers

from ..quantization.config import (
    META_FIELD_ACT_GROUP_AWARE,
    META_FIELD_DAMP_AUTO_INCREMENT,
    META_FIELD_DAMP_PERCENT,
    META_FIELD_MSE,
    META_FIELD_QUANTIZER,
    META_FIELD_STATIC_GROUPS,
    META_FIELD_TRUE_SEQUENTIAL,
    META_FIELD_URI,
    META_QUANTIZER_NANOMODEL,
    META_VALUE_URI,
)
from ..utils.backend import BACKEND
from ..utils.hf import sanitize_generation_config_file
from ..utils.logger import setup_logger
from ..utils.model import (
    copy_py_files,
    find_modules,
    get_model_files_size,
    get_state_dict_for_save,
    load_checkpoint_in_model_then_tie_weights,
    make_quant,
    streaming_state_dict_to_shards,
)
from ..utils.structure import alias_all_from_turtle_if_meta
from ..utils.torch import torch_empty_cache
from ..version import __version__
from ._const import DEFAULT_MAX_SHARD_SIZE

log = setup_logger()

PROCESS_LOG_NAME = "process"
PROCESS_LOG_LAYER = "layer"
PROCESS_LOG_MODULE = "module"
QUANT_LOG_LOSS = "loss"
QUANT_LOG_NSAMPLES = "samples"
QUANT_LOG_DAMP = "damp"
PROCESS_LOG_TIME = "time"
PROCESS_LOG_FWD_TIME = "fwd_time"
PROCESS_USED_MEMORY = "(v)ram"



def ModelWriter(cls):
    def save_pretrained(
        self,
        save_dir: str,
        **kwargs,
    ):
        log.warn(
            "You are using save_pretrained, which will re-direct to save_quantized."
        )
        self.save_quantized(save_dir=save_dir, **kwargs)

    cls.save_pretrained = save_pretrained

    def save_quantized(
        self,
        save_dir: str,
        safetensors_metadata: Optional[Dict[str, str]] = None,
        max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
        meta_quantizer: Optional[str] = None,
    ):
        """save quantized model and configs to local disk"""
        os.makedirs(save_dir, exist_ok=True)

        if self.quant_log:
            quant_log_path = os.path.join(save_dir, "quant_log.csv")
            try:
                with open(
                    quant_log_path, mode="w", newline="", encoding="utf-8"
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            PROCESS_LOG_LAYER,
                            PROCESS_LOG_MODULE,
                            QUANT_LOG_LOSS,
                            QUANT_LOG_NSAMPLES,
                            QUANT_LOG_DAMP,
                            PROCESS_LOG_TIME,
                        ]
                    )
                    writer.writerows(
                        [
                            [
                                entry.get(PROCESS_LOG_LAYER),
                                entry.get(PROCESS_LOG_MODULE),
                                entry.get(QUANT_LOG_LOSS),
                                entry.get(QUANT_LOG_NSAMPLES),
                                entry.get(QUANT_LOG_DAMP),
                                entry.get(PROCESS_LOG_TIME),
                            ]
                            for entry in self.quant_log
                        ]
                    )
            except OSError as exc:
                log.warning(
                    "Unable to persist quantization log to %s: %s", quant_log_path, exc
                )

        pre_quantized_size_mb = get_model_files_size(self.model_local_path)
        pre_quantized_size_gb = pre_quantized_size_mb / 1024

        quantizers = [f"{META_QUANTIZER_NANOMODEL}:{__version__}"]
        if meta_quantizer:
            if len(meta_quantizer.split(":")) == 2:
                quantizers.append(meta_quantizer.replace(" ", ""))
            else:
                log.warn(
                    f"meta_quantizer: '{meta_quantizer}' format is invalid, expected: 'quantizer_name:version'"
                )

        # write gptqmodel tooling fingerprint to config
        self.quantize_config.meta_set_versionable(
            key=META_FIELD_QUANTIZER,
            value=quantizers,
        )

        self.quantize_config.meta_set(
            key=META_FIELD_URI,
            value=META_VALUE_URI,
        )

        # meta: write config fields to meta if they doe not participate in inference
        self.quantize_config.meta_set(
            key=META_FIELD_DAMP_PERCENT,
            value=self.quantize_config.damp_percent,
        )

        self.quantize_config.meta_set(
            key=META_FIELD_DAMP_AUTO_INCREMENT,
            value=self.quantize_config.damp_auto_increment,
        )

        self.quantize_config.meta_set(
            key=META_FIELD_STATIC_GROUPS,
            value=self.quantize_config.static_groups,
        )

        self.quantize_config.meta_set(
            key=META_FIELD_TRUE_SEQUENTIAL,
            value=self.quantize_config.true_sequential,
        )

        self.quantize_config.meta_set(
            key=META_FIELD_MSE,
            value=self.quantize_config.mse,
        )

        self.quantize_config.meta_set(
            key=META_FIELD_ACT_GROUP_AWARE,
            value=self.quantize_config.act_group_aware,
        )

        # The config, quantize_config and model may be edited in place in save_quantized.
        config = copy.deepcopy(self.model.config)
        quantize_config = copy.deepcopy(self.quantize_config)

        if not self.quantized:
            raise ValueError(
                "Save aborted as model is not quantized. Please call `quantize()` first.",
            )

        if self.load_quantized_model:
            self.model = self.get_model_with_quantize(
                qcfg=quantize_config,
                model_id_or_path=self.model_local_path,
            )

        # --- start config save block ---
        # Save quantized config
        config.quantization_config = quantize_config.to_dict()
        self.model.config = config

        # Save model config, including generation_config
        # Use empty state_dict hack to bypass saving weights
        self.model.save_pretrained(save_dir, state_dict={}, is_main_process=True)

        gen_config_path = os.path.join(save_dir, "generation_config.json")
        if sanitize_generation_config_file(gen_config_path):
            log.info("Model: Sanitized `generation_config.json` before packaging.")

        # Save `quantize_config.json`
        quantize_config.save_pretrained(save_dir)

        def _log_saved_config_snapshot(path: str) -> None:
            if not log.isEnabledFor(logging.DEBUG):
                return
            try:
                files = sorted(os.listdir(path))
            except OSError as exc:
                log.debug("Unable to inspect save directory %s: %s", path, exc)
                return

            log.debug(
                "Packaged artifacts in %s: %s", path, ", ".join(files) or "<empty>"
            )

            for file_name in ("generation_config.json", "config.json"):
                full_path = os.path.join(path, file_name)
                if not os.path.isfile(full_path):
                    log.debug("Expected file `%s` not found under %s", file_name, path)
                    continue
                try:
                    with open(full_path, "r", encoding="utf-8") as config_file:
                        config_data = json.load(config_file)
                except (OSError, json.JSONDecodeError) as exc:
                    log.debug("Unable to parse `%s`: %s", full_path, exc)
                    continue
                log.debug(
                    "Snapshot of `%s`: %s",
                    file_name,
                    json.dumps(config_data, indent=2, ensure_ascii=False),
                )

        _log_saved_config_snapshot(save_dir)

        # Save processor related config files. For example: preprocessor_config.json, chat_template.json
        if hasattr(self, "processor") and isinstance(self.processor, ProcessorMixin):
            self.processor.save_pretrained(save_dir)
        # --- end config save block ---

        # Due to shell/turtle state, we need to sync the modules from turtle to shell
        if not self.load_quantized_model:
            alias_all_from_turtle_if_meta(
                shell_model=self.model, turtle_model=self.turtle_model
            )

        offload_root = (
            self.quantize_config.offload_to_disk_path
            if getattr(self.quantize_config, "offload_to_disk", False)
            else None
        )
        state_dict = get_state_dict_for_save(self.model, offload_root=offload_root)

        model_base_name = "model"
        model_save_name = model_base_name + ".safetensors"

        if not self.qlinear_kernel.SUPPORTS_SHARDS and max_shard_size is not None:
            log.warn("Sharding is not supported for this quant. Disabling sharding.")
            max_shard_size = None

        def _parse_max_shard_size(value: Optional[Union[int, str]]) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, int):
                return value
            match = re.fullmatch(r"\s*(\d+)([KMGTP]?B?)\s*", value, re.IGNORECASE)
            if not match:
                raise ValueError(f"Invalid max_shard_size value: {value}")
            base = int(match.group(1))
            suffix = match.group(2).upper()
            multiplier = 1
            if suffix.startswith("K"):
                multiplier = 1024
            elif suffix.startswith("M"):
                multiplier = 1024**2
            elif suffix.startswith("G"):
                multiplier = 1024**3
            elif suffix.startswith("T"):
                multiplier = 1024**4
            elif suffix.startswith("P"):
                multiplier = 1024**5
            return base * multiplier

        def _normalize_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, str]:
            if meta is None:
                return {}
            if not isinstance(meta, dict):
                raise TypeError("safetensors_metadata must be a dictionary.")
            normalized: Dict[str, str] = {}
            for key, value in meta.items():
                try:
                    new_key = str(key)
                    new_value = str(value)
                except Exception as exc:
                    raise TypeError(
                        f"safetensors_metadata: both keys and values must be strings and conversion failed for ({key}, {value}): {exc}"
                    )
                if new_key in normalized:
                    log.warn(
                        f"Duplicate metadata key '{new_key}' after conversion to string; overwriting previous value."
                    )
                normalized[new_key] = new_value
            return normalized

        max_shard_size_bytes = _parse_max_shard_size(max_shard_size)
        metadata_dict = _normalize_metadata(safetensors_metadata)
        metadata_dict["format"] = "pt"

        expected_files, tensor_to_filename, total_size_bytes = (
            streaming_state_dict_to_shards(
                state_dict,
                save_dir=save_dir,
                model_base_name=model_base_name,
                single_file_name=model_save_name,
                metadata=metadata_dict,
                max_shard_size=max_shard_size_bytes,
            )
        )

        pattern = re.compile(
            rf"{re.escape(model_base_name)}-\d{{5}}-of-\d{{5}}\.safetensors"
        )
        for filename in os.listdir(save_dir):
            full_filename = join(save_dir, filename)
            if not isfile(full_filename):
                continue
            if filename == model_save_name and filename not in expected_files:
                os.remove(full_filename)
                continue
            if pattern.fullmatch(filename) and filename not in expected_files:
                os.remove(full_filename)

        total_size_mb = total_size_bytes / (1024 * 1024)

        if len(expected_files) > 1:
            index = {
                "metadata": {"total_size": total_size_bytes},
                "weight_map": tensor_to_filename,
            }
            index_save_name = model_save_name + ".index.json"
            index_save_path = join(save_dir, index_save_name)
            with open(index_save_path, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
        else:
            index_save_path = join(save_dir, model_save_name + ".index.json")
            if os.path.exists(index_save_path):
                os.remove(index_save_path)

        state_dict.clear()

        # If the saved model is a loaded quantized model, do not calculate the size diff.
        if not self.load_quantized_model:
            total_size_gb = total_size_mb / 1024
            size_diff_mb = pre_quantized_size_mb - total_size_mb
            size_diff_gb = size_diff_mb / 1024
            percent_diff = (size_diff_mb / pre_quantized_size_mb) * 100
            log.info(
                f"Pre-Quantized model size: {pre_quantized_size_mb:.2f}MB, {pre_quantized_size_gb:.2f}GB"
            )
            log.info(
                f"Quantized model size: {total_size_mb:.2f}MB, {total_size_gb:.2f}GB"
            )
            log.info(
                f"Size difference: {size_diff_mb:.2f}MB, {size_diff_gb:.2f}GB - {percent_diff:.2f}%"
            )

        # need to copy .py files for model/tokenizers not yet merged to HF transformers
        if self.trust_remote_code:
            copy_py_files(save_dir, model_id_or_path=self.model_local_path)

        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)

            # fixed this issue: https://github.com/huggingface/transformers/issues/35832
            saved_tokenizer_config = get_tokenizer_config(save_dir)
            config_tokenizer_class = saved_tokenizer_config.get("tokenizer_class")
            is_fast_tokenizer = bool(getattr(self.tokenizer, "is_fast", False))
            if not is_fast_tokenizer:
                backend_tokenizer = getattr(self.tokenizer, "tokenizer", None)
                is_fast_tokenizer = isinstance(
                    backend_tokenizer, PreTrainedTokenizerFast
                )
            # if the tokenizer is fast, but the tokenizer_config.json does not have Fast suffix, add "Fast" suffix
            if (
                config_tokenizer_class
                and not config_tokenizer_class.endswith("Fast")
                and is_fast_tokenizer
            ):
                saved_tokenizer_config["tokenizer_class"] = (
                    config_tokenizer_class + "Fast"
                )
                with open(
                    os.path.join(save_dir, "tokenizer_config.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(saved_tokenizer_config, f, indent=2, ensure_ascii=False)

    cls.save_quantized = save_quantized

    def get_model_with_quantize(self, qcfg, model_id_or_path):
        config = AutoConfig.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
        )

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        transformers.modeling_utils._init_weights = False
        init_contexts = [no_init_weights()]
        with ContextManagers(init_contexts):
            model = cls.loader.from_config(config, dtype=torch.float16)

            modules = find_modules(model)
            ignore_modules = [self.lm_head] + self.get_base_modules(model)

            for name in list(modules.keys()):
                # allow loading of quantized lm_head
                if qcfg.lm_head and name == self.lm_head:
                    continue

                if any(
                    name.startswith(ignore_module) for ignore_module in ignore_modules
                ) or all(
                    not name.endswith(ignore_module)
                    for sublist in self.simple_layer_modules(config, qcfg)
                    for ignore_module in sublist
                ):
                    # log non-lm-head quantizerd modules only
                    if name is not self.lm_head:
                        log.info(f"The layer {name} is not quantized.")
                    del modules[name]

            make_quant(
                model,
                qcfg=qcfg,
                quant_result=modules,
                backend=BACKEND.AUTO,
                lm_head_name=cls.lm_head,
                pack=True,
            )

        load_checkpoint_in_model_then_tie_weights(
            model,
            dtype=torch.float16,
            # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
            checkpoint=self.checkpoint_file_name,
            # device_map=device_map,
            # offload_state_dict=True,
            # offload_buffers=True,
        )
        torch_empty_cache()
        return model

    cls.get_model_with_quantize = get_model_with_quantize

    return cls
