from __future__ import annotations

import copy
import csv
import json
import os
from os.path import isfile, join
from typing import Any, Dict, Optional, Union

import pcre as re
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoConfig, PreTrainedTokenizerFast, ProcessorMixin
from transformers.modeling_utils import no_init_weights
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils.generic import ContextManagers

from tqdm import tqdm
import logging

from ..quantization.config import (
    FORMAT,
    META_FIELD_ACT_GROUP_AWARE,
    META_FIELD_DAMP_AUTO_INCREMENT,
    META_FIELD_DAMP_PERCENT,
    META_FIELD_MSE,
    META_FIELD_QUANTIZER,
    META_FIELD_STATIC_GROUPS,
    META_FIELD_TRUE_SEQUENTIAL,
    META_FIELD_URI,
    META_FIELD_V2_ALPHA,
    META_FIELD_V2_ENABLED,
    META_QUANTIZER_NANOMODEL,
    META_VALUE_URI,
    METHOD,
    MIN_VERSION_WITH_V2,
)
from ..utils.backend import BACKEND
from ..utils.hf import sanitize_generation_config_file
from ..utils.model import (
    convert_gptq_v2_to_v1_format,
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

# ------------------------------
# Logging setup
# ------------------------------
logger = logging.getLogger("ModelWriter")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

PROCESS_LOG_NAME = "process"
PROCESS_LOG_LAYER = "layer"
PROCESS_LOG_MODULE = "module"
QUANT_LOG_LOSS = "loss"
QUANT_LOG_NSAMPLES = "samples"
QUANT_LOG_DAMP = "damp"
PROCESS_LOG_TIME = "time"
PROCESS_LOG_FWD_TIME = "fwd_time"
PROCESS_USED_MEMORY = "(v)ram"

EORA_DEFAULT_FILE = "eora.safetensors"


def ModelWriter(cls):
    def save_pretrained(self, save_dir: str, **kwargs):
        logger.warning("You are using save_pretrained, which will re-direct to save_quantized.")
        self.save_quantized(save_dir=save_dir, **kwargs)

    cls.save_pretrained = save_pretrained

    def save_quantized(
        self,
        save_dir: str,
        safetensors_metadata: Optional[Dict[str, str]] = None,
        max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
        meta_quantizer: Optional[str] = None,
        eora_path: Optional[str] = None,
    ):
        """save quantized model and configs to local disk"""
        os.makedirs(save_dir, exist_ok=True)

        # progress indicator
        logger.info("Saving quantized model...")
        steps = ["write quant log", "update metadata", "save configs", "save weights", "finalize"]
        for step in tqdm(steps, desc="Model Save Steps"):
            if step == "write quant log" and self.quant_log:
                with open(os.path.join(save_dir, "quant_log.csv"), mode="w", newline="") as file:
                    w = csv.writer(file)
                    w.writerow(
                        [PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES, QUANT_LOG_DAMP, PROCESS_LOG_TIME]
                    )
                    w.writerows(
                        [
                            [
                                entry.get(PROCESS_LOG_LAYER),
                                entry.get(PROCESS_LOG_MODULE),
                                entry.get(QUANT_LOG_LOSS),
                                entry.get(QUANT_LOG_DAMP),
                                entry.get(PROCESS_LOG_TIME),
                            ]
                            for entry in self.quant_log
                        ]
                    )

            elif step == "update metadata":
                pre_quantized_size_mb = get_model_files_size(self.model_local_path)
                pre_quantized_size_gb = pre_quantized_size_mb / 1024
                quantizers = [f"{META_QUANTIZER_NANOMODEL}:{__version__}"]
                if meta_quantizer:
                    if len(meta_quantizer.split(":")) == 2:
                        quantizers.append(meta_quantizer.replace(" ", ""))
                    else:
                        logger.warning(f"meta_quantizer: '{meta_quantizer}' format is invalid, expected: 'quantizer_name:version'")
                self.quantize_config.meta_set_versionable(key=META_FIELD_QUANTIZER, value=quantizers)
                self.quantize_config.meta_set(key=META_FIELD_URI, value=META_VALUE_URI)

            elif step == "save configs":
                config = copy.deepcopy(self.model.config)
                quantize_config = copy.deepcopy(self.quantize_config)
                if not self.quantized:
                    raise ValueError("Save aborted as model is not quantized. Please call `quantize()` first.")
                if quantize_config.format == FORMAT.GPTQ_V2:
                    logger.warning(
                        f"Using 'format = {FORMAT.GPTQ_V2}': the serialized model is only supported by NanoModel version >= {MIN_VERSION_WITH_V2}."
                    )

                if self.load_quantized_model:
                    self.model = self.get_model_with_quantize(
                        qcfg=quantize_config, model_id_or_path=self.model_local_path
                    )

                config.quantization_config = quantize_config.to_dict()
                self.model.config = config
                self.model.save_pretrained(save_dir, state_dict={}, is_main_process=True)
                gen_config_path = os.path.join(save_dir, "generation_config.json")
                if sanitize_generation_config_file(gen_config_path):
                    logger.info("Model: Sanitized `generation_config.json` before packaging.")
                quantize_config.save_pretrained(save_dir)

            elif step == "save weights":
                offload_root = (
                    self.quantize_config.offload_to_disk_path
                    if getattr(self.quantize_config, "offload_to_disk", False)
                    else None
                )
                state_dict = get_state_dict_for_save(self.model, offload_root=offload_root)
                model_base_name = "model"
                model_save_name = model_base_name + ".safetensors"
                max_shard_size_bytes = 1024**3 if max_shard_size is None else None
                metadata_dict = {"format": "pt"}

                expected_files, tensor_to_filename, total_size_bytes = streaming_state_dict_to_shards(
                    state_dict,
                    save_dir=save_dir,
                    model_base_name=model_base_name,
                    single_file_name=model_save_name,
                    metadata=metadata_dict,
                    max_shard_size=max_shard_size_bytes,
                )

                total_size_mb = total_size_bytes / (1024 * 1024)
                total_size_gb = total_size_mb / 1024
                logger.info(f"Quantized model size: {total_size_mb:.2f}MB ({total_size_gb:.2f}GB)")

            elif step == "finalize":
                if self.tokenizer:
                    self.tokenizer.save_pretrained(save_dir)
                    saved_tokenizer_config = get_tokenizer_config(save_dir)
                    cls_name = saved_tokenizer_config.get("tokenizer_class")
                    if cls_name and not cls_name.endswith("Fast") and isinstance(self.tokenizer.tokenizer, PreTrainedTokenizerFast):
                        saved_tokenizer_config["tokenizer_class"] = cls_name + "Fast"
                        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                            json.dump(saved_tokenizer_config, f, indent=2, ensure_ascii=False)

        logger.info("Quantized model saved successfully.")

    cls.save_quantized = save_quantized
    return cls
