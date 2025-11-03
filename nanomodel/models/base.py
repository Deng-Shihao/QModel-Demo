from __future__ import annotations

import copy
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch._dynamo
import torch.nn as nn


from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    modeling_utils,
)


try:  # Optional dependency for huggingface datasets support
    from datasets import Dataset as HFDataset
    from datasets import IterableDataset as HFIterableDataset
except Exception:  # pragma: no cover - datasets may not be installed
    HFDataset = None
    HFIterableDataset = None

from .. import DEVICE_THREAD_POOL
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.lookahead import configure_default_lookahead
from ..nn_modules.qlinear.torch import TorchQuantLinear

from ..quantization import QuantizeConfig
from ..quantization.config import KERNEL, METHOD, QUANTIZE_BLACK_LIST, dynamic_get
from ..quantization.rotation.rotation import fuse_layer_norms, rotate_model

from ..utils.backend import BACKEND
from ..utils.data import collate_data
from ..utils.device import get_device
from ..utils.hf import autofix_hf_model_config
from ..utils.importer import select_quant_linear
from ..utils.logger import QuantizationRegionTimer, setup_logger
from ..utils.model import MODALITY, find_modules, get_module_by_name_prefix, move_to
from ..utils.offload import offload_to_disk
from ..utils.structure import alias_from_turtle_for_submodule
from ..utils.torch import TORCH_HAS_COMPILE, torch_compile
from ._const import (
    CALIBRATION_DATASET_CONCAT_CHAR,
    CPU,
    DEFAULT_MAX_SHARD_SIZE,
    DEVICE,
    EXPERT_INDEX_PLACEHOLDER,
    META,
)

from .loader import ModelLoader
from .writer import ModelWriter

if TYPE_CHECKING:
    try:
        from datasets import Dataset as HFDatasetType
        from datasets import IterableDataset as HFIterableDatasetType
    except Exception:  # pragma: no cover - optional dependency
        HFDatasetType = HFIterableDatasetType = object


class _ClassPropertyDescriptor:
    def __init__(self, fget, fset=None):
        self.fget = fget

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        return self.fget.__get__(instance, owner)()


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)


def generate_node_for_awq_scaling(
    inp, prev_op, module_kwargs, nodes_size, subset, module2inspect
):
    n = {
        "prev_op": prev_op,
        "layers": subset,
        "inp": inp,
    }
    if nodes_size == 0:
        # Only the first node needs kwargs
        n["kwargs"] = module_kwargs

    if module2inspect is not None:
        n["module2inspect"] = module2inspect

    return n, None


def check_support_param_buffer_assignment(*args, **kwargs):
    return False


def apply_module_tree_override(module_tree, override):
    """
    Recursively find the corresponding key of override in module_tree and override it.
    """
    if isinstance(module_tree, dict) and isinstance(override, dict):
        for k, v in override.items():
            if (
                k in module_tree
                and isinstance(module_tree[k], (dict, list))
                and isinstance(v, (dict, list))
            ):
                module_tree[k] = apply_module_tree_override(module_tree[k], v)
            else:
                module_tree[k] = v
    elif isinstance(module_tree, list) and isinstance(override, list):
        for o in override:
            if isinstance(o, dict):
                for b in module_tree:
                    if isinstance(b, dict):
                        apply_module_tree_override(b, o)
    return module_tree


NOT_QUANTIZE_FLAG = ":!"

# Fix cpu memory leak.
# See https://github.com/huggingface/transformers/issues/34366
modeling_utils.check_support_param_buffer_assignment = (
    check_support_param_buffer_assignment
)

log = setup_logger()


class BaseNanoModel(nn.Module):
    """Shared helpers for preparing calibration data and managing quantization lifecycle."""

    lm_head: str = "lm_head"

    module_tree: List[str] = None
    module_tree_overrides: dict[METHOD, List[str]] = None

    layer_modules_strict = True

    pre_lm_head_norm_module: str = None

    awq_scale_optimize_shape_dependent_modules: List[str] = None

    require_trust_remote_code = None
    require_pkgs_version: Optional[List[str]] = None
    require_dtype: Optional[str | torch.dtype] = None
    require_fast_init: bool = True

    # some models require Processor? For example, Qwen2VLImageProcessor.
    require_load_processor = False

    # allow dynamic expert n-count layer extraction
    # so moe model defs do not need to write out 64 layers if expert size is 64 (Qwen2Moe)
    # usage: set to property in model.config that holds this int value: total number of experts

    dynamic_expert_index: Optional[str] = None

    # some models require a different model loader, such as mllama which uses AutoModelForPreTraining
    loader = AutoModelForCausalLM

    require_monkeypatch = False

    support_batch_quantize = True

    info: Dict[str, str] = {}

    supports_act_order = [True, False]

    modality: List[MODALITY] = [MODALITY.TEXT]

    quant_override_files: Dict[str, Union[str | Dict[str, Any]]] = {}

    server = None

    ATTENTION_MASKS_DTYPE = torch.bool  # default to bool

    ATTENTION_MASKS_REQUIRED_FOR_INPUT: bool = False

    INPUT_EMBEDDING_EXTRA_ARGS = None

    def __init__(
        self,
        model: PreTrainedModel,
        quantized: bool,
        quantize_config: QuantizeConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        qlinear_kernel: Optional[Type[nn.Module]] = None,
        load_quantized_model: bool = False,
        trust_remote_code: bool = False,
        model_local_path: Optional[str] = None,
        turtle_model: Optional[PreTrainedModel] = None,
    ):
        """
        Initializes the BaseNanoModel.

        Args:
            model (PreTrainedModel): The Hugging Face model to be wrapped.
            quantized (bool): Whether the model is already quantized.
            quantize_config (QuantizeConfig): The configuration for quantization.
            tokenizer (Optional[PreTrainedTokenizerBase]): The tokenizer associated with the model.
            qlinear_kernel (Optional[Type[nn.Module]]): The kernel to use for quantized linear layers.
            load_quantized_model (bool): Whether we are loading a pre-quantized model.
            trust_remote_code (bool): Whether to trust remote code when loading models.
            model_local_path (Optional[str]): The local path to the model, if available.
            turtle_model (Optional[PreTrainedModel]): A meta-device model used to reduce CPU RAM
                usage during the quantization process.
        """
        super().__init__()

        # If a quantization method-specific module tree override exists, apply it.
        quant_method = quantize_config.quant_method
        if (
            self.module_tree_overrides is not None
            and self.module_tree_overrides.get(quant_method) is not None
        ):
            log.info(f"Module Tree: overridden by METHOD.{quant_method.upper()}")
            type(self).module_tree = apply_module_tree_override(
                self.module_tree, self.module_tree_overrides[quant_method]
            )

        # Record configuration early so model lifecycle hooks can rely on them.
        self.quantized = quantized
        self.load_quantized_model = load_quantized_model
        self.qlinear_kernel = qlinear_kernel
        self.trust_remote_code = trust_remote_code
        self.model_local_path = model_local_path
        self.quantize_config = quantize_config
        self.compiled = False  # Will be set to True after torch.compile is successful.

        # Timers and memory management for quantization.
        self.quant_region_timer = QuantizationRegionTimer(logger=log)
        self._turtle_reload_threshold_bytes = self._resolve_turtle_reload_threshold()
        self._turtle_reload_accum_bytes = 0
        self._turtle_materialized_ids: Set[int] = set()
        self._turtle_lock = threading.RLock()

        # State for quantization process.
        self.processor: Optional[ProcessorMixin] = None
        self.quant_log = []  # Stores per-layer quantization stats.

        # Load and configure the model.
        self.model = self.after_model_load(
            model, load_quantized_model=load_quantized_model
        )
        self.turtle_model = turtle_model
        self._assign_tokenizer(tokenizer)

        # Auto-fix common model config errors.
        if isinstance(self.model, PreTrainedModel):
            autofix_hf_model_config(self.model, path=model_local_path)

        # Load processor if required by the model.
        if self.require_load_processor:
            self.processor = AutoProcessor.from_pretrained(model_local_path)

        # Apply model-specific monkey patches if necessary.
        if self.require_monkeypatch:
            self.monkey_patch()

        # Configure lookahead optimization for quantized modules if applicable.
        self._auto_configure_lookahead()

    def _assign_tokenizer(self, tokenizer: Optional[PreTrainedTokenizerBase]) -> None:
        """Attach the tokenizer to both this wrapper and the underlying Hugging Face model."""
        if tokenizer is not None and not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`."
            )

        self.tokenizer = tokenizer
        # Ensure downstream helpers can rely on `model.tokenizer` consistently.
        self.model.tokenizer = tokenizer

    @classmethod
    def extract_layers_node(cls) -> List[str]:
        """
        Extracts the layers node path from the module_tree structure.

        It concatenates module names with '.' up to (but not including) the first "#" marker.
        Example:
            Given module_tree = ["model", "layers", "#", {...}]
            Returns: ["model.layers"]
        """
        prefix_parts = []
        for node in cls.module_tree:
            if node == "#":
                break
            if isinstance(node, str):
                prefix_parts.append(node)
            else:
                # Stop if a non-string element is found before '#'
                break
        return [".".join(prefix_parts)] if prefix_parts else []

    @classmethod
    def build_moe_modules_if_need(
        cls, model_config, layer_modules, is_awq_quantize: bool = False
    ):
        """
        Expands module names for Mixture-of-Experts (MoE) models if dynamic expert
        counting is enabled.
        """
        if model_config is None or cls.dynamic_expert_index is None:
            return layer_modules

        num_experts = cls.get_num_experts(model_config)
        moe_expanded_modules = []

        for names in layer_modules:
            # A block of module names is considered for expert expansion if all names
            # contain the expert placeholder.
            is_expert_block = all(EXPERT_INDEX_PLACEHOLDER in n for n in names)

            if not is_expert_block:
                moe_expanded_modules.append(names)
                continue

            current_block = []
            if is_awq_quantize:
                # For AWQ, group by expert index first.
                # e.g., ['mlp.experts.0.gate_proj', 'mlp.experts.0.up_proj', 'mlp.experts.1.gate_proj', ...]
                for index in range(num_experts):
                    for n in names:
                        current_block.append(
                            n.replace(EXPERT_INDEX_PLACEHOLDER, str(index))
                        )
            else:
                # For other methods, group by module type first.
                # e.g., ['mlp.experts.0.gate_proj', 'mlp.experts.1.gate_proj', 'mlp.experts.0.up_proj', ...]
                for n in names:
                    for index in range(num_experts):
                        current_block.append(
                            n.replace(EXPERT_INDEX_PLACEHOLDER, str(index))
                        )
            moe_expanded_modules.append(current_block)

        return moe_expanded_modules

    @classmethod
    def get_num_experts(cls, model_config) -> int:
        """Retrieves the number of experts from the model configuration."""
        if hasattr(model_config, "text_config"):
            # Handle nested configuration objects
            num_experts = getattr(model_config.text_config, cls.dynamic_expert_index)
        elif hasattr(model_config, "thinker_config"):
            num_experts = getattr(
                model_config.thinker_config.text_config, cls.dynamic_expert_index
            )
        else:
            num_experts = getattr(model_config, cls.dynamic_expert_index)
        return num_experts

    @classmethod
    def filter_not_quantize_module(cls, layer_modules, quantize_config):
        """Filters out modules that are marked as non-quantizable."""
        # Remove modules containing the non-quantize flag
        filtered_layer_modules = [
            [name for name in block if NOT_QUANTIZE_FLAG not in name]
            for block in layer_modules
        ]
        # Remove any empty blocks that result from filtering
        filtered_layer_modules = [block for block in filtered_layer_modules if block]

        # Further filter based on dynamic quantization configuration
        if getattr(quantize_config, "dynamic", None):
            new_layer_modules = []
            for modules in filtered_layer_modules:
                filtered = [
                    m
                    for m in modules
                    if dynamic_get(quantize_config.dynamic, module_name=m) is not False
                ]
                if filtered:
                    new_layer_modules.append(filtered)
            return new_layer_modules

        return filtered_layer_modules

    @classmethod
    def simple_layer_modules(
        cls, model_config, quantize_config, is_awq_quantize: bool = False
    ) -> List[List[str]]:
        """
        Builds a simplified list of layer modules for quantization.

        This list is created by building the full module list, expanding for MoE if needed,
        and then filtering out non-quantizable modules. This is the primary method
        used during the quantization process.
        """
        layer_modules = cls.build_layer_modules(cls.module_tree)
        layer_modules = cls.build_moe_modules_if_need(
            model_config, layer_modules, is_awq_quantize
        )
        layer_modules = cls.filter_not_quantize_module(layer_modules, quantize_config)
        return layer_modules

    @classmethod
    def full_layer_modules(
        cls, model_config=None, is_awq_quantize: bool = False
    ) -> List[List[str]]:
        """
        Builds the complete list of layer modules, including non-quantizable ones.

        This is useful for operations that need to inspect the entire model structure,
        such as AWQ scaling.
        """
        full = cls.build_layer_modules(cls.module_tree)
        full = cls.build_moe_modules_if_need(model_config, full, is_awq_quantize)
        return full

    def _materialize_calibration_examples(
        self,
        calibration_dataset: Union[
            List[Dict[str, Union[List[int], torch.LongTensor]]],
            List[str],
            List[List[int]],
            "HFDatasetType",
            "HFIterableDatasetType",
        ],
    ) -> List[Any]:
        """Convert the provided calibration dataset into a concrete list of examples."""
        hf_dataset_types: Tuple[type, ...] = tuple(
            dataset_type
            for dataset_type in (HFDataset, HFIterableDataset)
            if dataset_type is not None
        )

        if isinstance(calibration_dataset, str):
            raise ValueError(
                "Quantize: calibration dataset must be iterable, not a single string."
            )

        if hf_dataset_types and isinstance(calibration_dataset, hf_dataset_types):
            return list(calibration_dataset)

        try:
            return list(calibration_dataset)
        except TypeError as exc:
            raise ValueError(
                "Quantize: calibration dataset must be iterable and materializable."
            ) from exc

    def _resolve_sequence_length_limit(self) -> tuple[Optional[int], Optional[str]]:
        """Inspect the HF config for the smallest declared maximum sequence length."""
        model_config = getattr(self.model, "config", None)
        if model_config is None:
            return None, None

        max_positions: Optional[int] = None
        max_positions_source: Optional[str] = None

        def _maybe_resolve_length(value, source_name):
            nonlocal max_positions, max_positions_source
            try:
                if value is None:
                    return False
                limit = int(value)
            except Exception:
                return False
            if limit <= 0:
                return False
            if max_positions is None or limit < max_positions:
                max_positions = limit
                max_positions_source = source_name
            return True

        primary_names = ("max_position_embeddings",)
        fallback_names = (
            "max_sequence_length",
            "max_seq_len",
            "n_positions",
            "seq_length",
        )

        for attr_name in primary_names:
            if _maybe_resolve_length(getattr(model_config, attr_name, None), attr_name):
                break
        if max_positions is None:
            for attr_name in fallback_names:
                if _maybe_resolve_length(
                    getattr(model_config, attr_name, None), attr_name
                ):
                    break

        return max_positions, max_positions_source

    def prepare_dataset(
        self,
        calibration_dataset: Union[
            List[Dict[str, Union[List[int], torch.LongTensor]]],
            List[str],
            List[List[int]],
            "HFDatasetType",
            "HFIterableDatasetType",
        ],
        calibration_dataset_concat_size: Optional[int] = None,
        calibration_dataset_sort: Optional[str] = None,
        batch_size: int = 1,
        calibration_data_min_length: int = 10,
    ):
        """Tokenize/normalize calibration samples and collate them into batches."""
        # 1. Materialize and validate the initial dataset.
        raw_examples = self._materialize_calibration_examples(calibration_dataset)
        if not raw_examples:
            raise ValueError("Quantize: calibration dataset is empty.")

        # 2. Process and tokenize each example based on its format (text, dict, etc.).
        processed_examples = self._process_and_tokenize_examples(raw_examples)

        # 3. Trim sequences to the model's max length and filter out short examples.
        trimmed_examples = self._trim_and_filter_examples(
            processed_examples, min_length=calibration_data_min_length
        )

        # 4. Concatenate examples into larger sequences if a concat size is provided.
        if calibration_dataset_concat_size:
            final_examples = self._concatenate_examples(
                trimmed_examples, concat_size=calibration_dataset_concat_size
            )
        else:
            final_examples = trimmed_examples

        # 5. Sort the dataset according to the specified mode (asc, desc, shuffle).
        sorted_examples = self._sort_examples(
            final_examples, sort_mode=calibration_dataset_sort
        )

        # 6. Collate the final dataset into batches.
        batched_dataset = self._batch_examples(sorted_examples, batch_size=batch_size)

        return batched_dataset

    def _process_and_tokenize_examples(
        self, raw_examples: List[Any]
    ) -> List[Dict[str, torch.Tensor]]:
        """Process and tokenize a list of raw calibration examples."""
        processed_examples = []
        for idx, example in enumerate(raw_examples):
            if isinstance(example, Mapping):
                # Handle dictionary-like examples (e.g., from datasets library)
                example_dict = dict(example)
                if "messages" in example_dict:
                    processed_examples.append(
                        self._tokenize_messages_value(example_dict["messages"], idx)
                    )
                elif "text" in example_dict:
                    processed_examples.append(
                        self._tokenize_text_value(example_dict["text"], idx)
                    )
                elif "input_ids" in example_dict:
                    processed_examples.append(
                        self._pack_ids(
                            example_dict["input_ids"],
                            example_dict.get("attention_mask"),
                            idx,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported calibration dict at index {idx}: keys={list(example_dict.keys())}"
                    )
            elif isinstance(example, str):
                # Handle raw string examples
                processed_examples.append(self._tokenize_text_value(example, idx))
            elif isinstance(example, (list, tuple)):
                # Handle list-of-integers examples
                if all(isinstance(x, int) for x in example):
                    processed_examples.append(self._pack_ids(list(example), None, idx))
                else:
                    raise ValueError(
                        f"List-based calibration example at index {idx} must contain only integers."
                    )
            elif torch.is_tensor(example):
                # Handle tensor examples
                processed_examples.append(self._pack_ids(example, None, idx))
            else:
                raise ValueError(
                    f"Unsupported calibration example type {type(example)} at index {idx}."
                )
        return processed_examples

    def _trim_and_filter_examples(
        self, tokenized_examples: List[Dict[str, torch.Tensor]], min_length: int
    ) -> List[Dict[str, List[List[int]]]]:
        """Trim examples to max sequence length and filter out those that are too short."""
        max_positions, max_positions_source = self._resolve_sequence_length_limit()
        trimmed_row_count = 0
        longest_trimmed_row = 0

        new_calibration_dataset = []
        too_short_count = 0

        for example in tokenized_examples:
            # Convert tensors to nested lists for manipulation
            input_ids = self._convert_tensor_to_list(example["input_ids"])
            attention_mask = self._convert_tensor_to_list(example["attention_mask"])

            # Trim sequences that exceed the model's maximum length
            if max_positions is not None:
                trimmed = False
                trimmed_input_ids, trimmed_attention_mask = [], []
                for row_ids, row_mask in zip(input_ids, attention_mask):
                    if len(row_ids) > max_positions:
                        trimmed = True
                        trimmed_row_count += 1
                        longest_trimmed_row = max(longest_trimmed_row, len(row_ids))
                        trimmed_input_ids.append(row_ids[:max_positions])
                        trimmed_attention_mask.append(row_mask[:max_positions])
                    else:
                        trimmed_input_ids.append(row_ids)
                        trimmed_attention_mask.append(row_mask)
                if trimmed:
                    input_ids, attention_mask = (
                        trimmed_input_ids,
                        trimmed_attention_mask,
                    )

            # Filter out examples that are shorter than the minimum length
            if len(input_ids[0]) <= min_length:
                too_short_count += 1
                continue

            new_calibration_dataset.append(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )

        if too_short_count > 0:
            log.warning(
                f"Quantize: {too_short_count} inputs with length <= {min_length} were removed."
            )
        if trimmed_row_count > 0:
            log.info(
                f"Quantize: Trimmed {trimmed_row_count} calibration rows to {max_positions} tokens (source: {max_positions_source}, longest: {longest_trimmed_row})."
            )

        return new_calibration_dataset

    def _concatenate_examples(
        self, examples: List[Dict[str, List[List[int]]]], concat_size: int
    ) -> List[Dict[str, List[List[int]]]]:
        """Concatenate multiple examples into single, larger sequences of a fixed size."""
        self._require_tokenizer("`calibration_dataset_concat_size` is specified")

        concatenated_data = []
        input_ids_buffer, attention_mask_buffer = [], []
        current_length = 0

        # Tokenize the separator character
        new_line_tokens = self.tokenizer(
            CALIBRATION_DATASET_CONCAT_CHAR, return_tensors="pt"
        )
        new_line_ids = self._convert_tensor_to_list(new_line_tokens["input_ids"])[0]
        new_line_mask = self._convert_tensor_to_list(new_line_tokens["attention_mask"])[
            0
        ]
        new_line_len = len(new_line_ids)

        for example in examples:
            ids, mask = example["input_ids"][0], example["attention_mask"][0]

            # If adding the next example exceeds the concat size, process the buffer
            if current_length + len(ids) + new_line_len >= concat_size:
                if current_length > 0:
                    # Fill remaining space in the buffer
                    remaining = concat_size - current_length
                    if remaining > new_line_len:
                        input_ids_buffer.extend(new_line_ids)
                        input_ids_buffer.extend(ids[: remaining - new_line_len])
                        attention_mask_buffer.extend(new_line_mask)
                        attention_mask_buffer.extend(mask[: remaining - new_line_len])

                    concatenated_data.append(
                        {
                            "input_ids": [input_ids_buffer],
                            "attention_mask": [attention_mask_buffer],
                        }
                    )

                # Start a new buffer with the current example
                input_ids_buffer = ids[:concat_size]
                attention_mask_buffer = mask[:concat_size]
                current_length = len(input_ids_buffer)
            else:
                # Add the example to the current buffer
                if current_length > 0:
                    input_ids_buffer.extend(new_line_ids)
                    attention_mask_buffer.extend(new_line_mask)
                    current_length += new_line_len

                input_ids_buffer.extend(ids)
                attention_mask_buffer.extend(mask)
                current_length += len(ids)

        # Add the last buffer if it contains data
        if input_ids_buffer:
            padding_len = concat_size - len(input_ids_buffer)
            if padding_len > 0:
                input_ids_buffer.extend([self.tokenizer.pad_token_id] * padding_len)
                attention_mask_buffer.extend([0] * padding_len)
            concatenated_data.append(
                {
                    "input_ids": [input_ids_buffer],
                    "attention_mask": [attention_mask_buffer],
                }
            )

        return concatenated_data

    def _sort_examples(
        self, examples: List[Dict[str, List[List[int]]]], sort_mode: Optional[str]
    ) -> List[Dict[str, List[List[int]]]]:
        """Sort or shuffle the calibration dataset."""
        sort_mode = (sort_mode or "").lower()
        if sort_mode in {"asc", "desc"}:
            log.info(f"Calibration: Sorting by length in {sort_mode}ending order.")
            return sorted(
                examples,
                key=lambda item: len(item["input_ids"][0]),
                reverse=sort_mode == "desc",
            )
        elif sort_mode == "shuffle":
            log.info("Calibration: Shuffling dataset randomly.")
            shuffled = examples[:]
            random.shuffle(shuffled)
            return shuffled
        else:
            log.info("Calibration: Using native dataset order.")
            return examples

    def _batch_examples(
        self, examples: List[Dict[str, Any]], batch_size: int
    ) -> List[Dict[str, torch.Tensor]]:
        """Collate a list of examples into batches."""
        if self.support_batch_quantize:
            # Collate data into batches with padding
            batched_dataset = [
                collate_data(
                    examples[start : start + batch_size], self.tokenizer.pad_token_id
                )
                for start in range(0, len(examples), batch_size)
            ]

            # Log token statistics
            total_padded = sum(
                (batch["attention_mask"] == 0).sum().item() for batch in batched_dataset
            )
            total_non_padded = sum(
                (batch["attention_mask"] == 1).sum().item() for batch in batched_dataset
            )
            log.info(
                f"Calibration: Total tokens: {total_non_padded + total_padded} ({total_non_padded} non-padded, {total_padded} padded)."
            )

            return batched_dataset
        else:
            # If batching is not supported, each example is its own "batch"
            return [
                {"input_ids": torch.tensor(block["input_ids"], dtype=torch.long)}
                for block in examples
            ]

    def _require_tokenizer(self, reason: str) -> None:
        """Check for the existence of a tokenizer, raising an error if it's missing."""
        if self.tokenizer is None:
            raise ValueError(f"A tokenizer must be provided when {reason}.")

    def _to_2d_long_tensor(self, value: Any, name: str, idx: int) -> torch.Tensor:
        """Convert a value to a 2D LongTensor, raising detailed errors on failure."""
        try:
            tensor = torch.as_tensor(value, dtype=torch.long)
        except Exception as exc:
            raise ValueError(
                f"Failed to convert `{name}` to a tensor for calibration item {idx}."
            ) from exc

        if tensor.ndim == 0:
            raise ValueError(
                f"`{name}` for item {idx} must be 1D or 2D, but got a scalar."
            )
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(
                f"`{name}` for item {idx} must be 1D or 2D, but got rank {tensor.ndim}."
            )
        return tensor

    def _pack_ids(
        self, ids_value: Any, mask_value: Any, idx: int
    ) -> Dict[str, torch.Tensor]:
        """Pack input_ids and an optional attention_mask into a standardized dictionary."""
        ids_tensor = self._to_2d_long_tensor(ids_value, "input_ids", idx)

        if mask_value is None:
            mask_tensor = torch.ones_like(ids_tensor)
        else:
            mask_tensor = self._to_2d_long_tensor(mask_value, "attention_mask", idx)
            if mask_tensor.shape != ids_tensor.shape:
                # Attempt to reshape if the number of elements is the same
                if mask_tensor.numel() == ids_tensor.numel():
                    mask_tensor = mask_tensor.reshape(ids_tensor.shape)
                else:
                    raise ValueError(
                        f"Shape mismatch for item {idx}: input_ids is {ids_tensor.shape}, attention_mask is {mask_tensor.shape}."
                    )

        return {
            "input_ids": ids_tensor.detach(),
            "attention_mask": mask_tensor.detach(),
        }

    def _tokenize_text_value(
        self, text_value: Any, idx: int
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a raw text string."""
        self._require_tokenizer("calibration data contains raw text")
        tokenized = self.tokenizer(
            text_value, add_special_tokens=True, return_tensors="pt"
        )
        return self._pack_ids(
            tokenized["input_ids"], tokenized.get("attention_mask"), idx
        )

    def _tokenize_messages_value(
        self, messages_value: Any, idx: int
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a chat-style messages list using the tokenizer's template."""
        self._require_tokenizer("calibration data uses the `messages` feature")
        apply_fn = getattr(self.tokenizer, "apply_template", None)
        if apply_fn is None:
            raise ValueError(
                "Tokenizer must have `apply_template` to handle `messages` calibration data."
            )

        # `apply_template` can have different signatures
        try:
            templated = apply_fn(messages_value, tokenize=False)
        except TypeError:
            templated = apply_fn(messages_value)

        if templated is None:
            raise ValueError(
                f"tokenizer.apply_template returned None for calibration item {idx}."
            )

        # The result of apply_template can be a dict, list of ints, a tensor, or a string
        if hasattr(templated, "get") and "input_ids" in templated:
            return self._pack_ids(
                templated["input_ids"], templated.get("attention_mask"), idx
            )
        if isinstance(templated, str):
            return self._tokenize_text_value(templated, idx)
        if torch.is_tensor(templated) or (
            isinstance(templated, (list, tuple))
            and templated
            and isinstance(templated[0], int)
        ):
            return self._pack_ids(templated, None, idx)

        raise ValueError(
            f"tokenizer.apply_template returned an unsupported type {type(templated)} for item {idx}."
        )

    def _convert_tensor_to_list(
        self, tensor: Union[torch.Tensor, List]
    ) -> List[List[int]]:
        """Ensure the input is a nested list of integers."""
        if isinstance(tensor, torch.Tensor):
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return tensor.long().cpu().numpy().tolist()
        # Already a list
        if isinstance(tensor[0], list):
            return tensor
        return [tensor]

    def quantize(
        self,
        calibration: Union[
            List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]
        ],
        calibration_concat_size: Optional[int] = None,
        calibration_sort: Optional[str] = "desc",
        batch_size: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        backend: Optional[BACKEND] = BACKEND.AUTO,
        calibration_data_min_length: int = 10,
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Performs quantization on the model using the provided calibration dataset.

        Args:
            calibration: The dataset to use for calibration.
            calibration_concat_size: If specified, concatenates calibration samples into sequences of this size.
            calibration_sort: The sorting method for the calibration dataset ('asc', 'desc', 'shuffle').
            batch_size: The batch size to use for quantization.
            tokenizer: An optional tokenizer. If not provided, the model's existing tokenizer is used.
            backend: The computation backend to use (e.g., 'auto', 'torch').
            calibration_data_min_length: Minimum sequence length for a calibration sample to be included.

        Returns:
            A dictionary containing the quantization log.
        """
        if self.quantized:
            raise EnvironmentError(
                "quantize() called on a model that is already quantized."
            )

        if self.quantize_config.quant_method in QUANTIZE_BLACK_LIST:
            raise ValueError(
                f"Unsupported quant method: {self.quantize_config.quant_method}"
            )

        # Reset timers and memory management state.
        self.quant_region_timer.reset()
        self._turtle_reload_accum_bytes = 0
        self._turtle_materialized_ids = set()

        # Handle model-specific constraints.
        if not self.support_batch_quantize:
            log.warning(
                "Model does not support batch quantization; batch_size is forced to 1."
            )
            batch_size = 1

        # Validate quantization configuration.
        self._validate_quantize_config()

        # Determine the computation backend.
        preferred_backend = backend
        if not preferred_backend or preferred_backend == BACKEND.AUTO:
            preferred_backend = BACKEND.TORCH

        # Select the appropriate quantized linear kernel.
        self.qlinear_kernel = select_quant_linear(
            bits=self.quantize_config.bits,
            group_size=self.quantize_config.group_size,
            act_order=self.quantize_config.act_order,
            sym=self.quantize_config.sym,
            pack=True,
            dynamic=self.quantize_config.dynamic,
            device=self.quantize_config.device,
            pack_dtype=self.quantize_config.pack_dtype,
            multi_select=False,
            backend=preferred_backend,
            kernel=self.quantize_config.kernel,
            quant_method=self.quantize_config.quant_method,
        )

        # Update tokenizer if a new one is provided.
        if tokenizer is not None:
            self._assign_tokenizer(tokenizer)

        # Apply model rotation if configured.
        if self.quantize_config.rotation:
            self._apply_model_rotation()

        # Prepare arguments for quantization processors.
        processor_args = {
            "tokenizer": self.tokenizer,
            "qcfg": self.quantize_config,
            "calibration": calibration,
            "prepare_dataset_func": self.prepare_dataset,
            "calibration_concat_size": calibration_concat_size,
            "calibration_sort": calibration_sort,
            "batch_size": batch_size,
        }

        # Initialize the quantization processors based on the chosen method.
        from ..processors.module_processor import ModuleProcessor
        from ..processors.tensorparallel_weight_processor import (
            TensorParallelWeightProcessor,
        )

        if self.quantize_config.quant_method == METHOD.AWQ:
            from ..processors.awq_processor import AWQProcessor

            os.environ["AWQ_BATCH_SIZE"] = str(batch_size)
            awq_args = {
                "gptq_model": self,
                "model": self.model,
                "batch_size": batch_size,
                **processor_args,
            }
            processors = [
                TensorParallelWeightProcessor(**processor_args),
                AWQProcessor(**awq_args),
            ]
        else:  # Default to GPTQ
            from ..processors.gptq_processor import GPTQProcessor

            processors = [
                TensorParallelWeightProcessor(**processor_args),
                GPTQProcessor(**processor_args),
            ]

        # Run the quantization process.
        module_processor = ModuleProcessor(self, processors=processors)
        result = module_processor.loop(
            backend=backend,
            fail_safe=self.quantize_config.fail_safe,
        )

        self.quant_region_timer.flush()
        return result

    def _validate_quantize_config(self):
        """Performs validation and applies auto-fixes to the quantization config."""
        if self.quantize_config.kernel == KERNEL.MARLIN:
            raise ValueError(
                "FORMAT.MARLIN is deprecated for quantization. Please use FORMAT.GPTQ, which will "
                "automatically use the Marlin kernel for accelerated inference where supported."
            )

        if self.quantize_config.quant_method == METHOD.AWQ:
            if self.quantize_config.kernel == KERNEL.GEMV_FAST:
                log.info(
                    "AWQ with GEMV_FAST kernel requires pack_dtype=torch.int16. Auto-fixing."
                )
                self.quantize_config.pack_dtype = torch.int16
            elif self.quantize_config.kernel == KERNEL.MARLIN:
                log.info(
                    "AWQ with Marlin kernel requires zero_point=False. Auto-fixing."
                )
                self.quantize_config.zero_point = False

    def _apply_model_rotation(self):
        """Applies layer rotation to the model if configured."""
        from nanomodel.models.definitions.llama import LlamaNanoModel
        from nanomodel.models.definitions.qwen2 import Qwen2NanoModel

        if not isinstance(self, (LlamaNanoModel, Qwen2NanoModel)):
            raise ValueError(
                f"Rotation is only supported for Llama and Qwen2 models, not {self.__class__.__name__}."
            )

        if self.model.config.tie_word_embeddings:
            log.info("Rotation requires untied word embeddings. Untying weights.")
            self.model.config.tie_word_embeddings = False
            lm_head, _ = get_module_by_name_prefix(self.model, self.lm_head)
            lm_head.weight = nn.Parameter(lm_head.weight.data.clone())

        module_name_args = {
            "layers_node": self.extract_layers_node(),
            "lm_head_name": self.lm_head,
        }

        # Fuse layer norms before rotation
        self.model = fuse_layer_norms(
            model=self.model,
            pre_lm_head_norm_module_name=self.pre_lm_head_norm_module,
            **module_name_args,
        )

        # Apply rotation, using CPU for MPS devices as float64 is not supported.
        rotation_device = (
            self.quantize_config.device
            if self.quantize_config.device != DEVICE.MPS
            else DEVICE.CPU
        )
        self.model, _ = rotate_model(
            model=self.model,
            rotate_mode=self.quantize_config.rotation,
            device=rotation_device,
            **module_name_args,
        )

    def to(self, device: Union[str, torch.device]):
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
            return self
        else:
            raise f"{self.model.__class__.__name__} does not support the to() method"

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, inputs=None, **kwargs):
        with torch.inference_mode():
            # fix hf generate not applying correct pad token
            pad_token_id = kwargs.get("pad_token_id", None)
            if pad_token_id is None and self.tokenizer:
                kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            if isinstance(inputs, str) or (
                isinstance(inputs, list) and all(isinstance(x, str) for x in inputs)
            ):
                if self.tokenizer is None:
                    raise ValueError(
                        "You passed in an `input` to `generate()` of type `str` but model is missing `model.tokenizer`. Please set `model.tokenizer = my_tokenizer`."
                    )
                inputs = self.tokenizer(
                    inputs, return_tensors="pt", padding=True, padding_side="left"
                ).to(self.model.device)
                return self.model.generate(**inputs, **kwargs)

            return self.model.generate(inputs=inputs, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    # placeholder, noop, and alert users to correct static api
    def push_to_hub(
        self,
        repo_id: str,
        quantized_path: str,  # saved local directory path
        private: bool = False,
        exists_ok: bool = False,  # set to true if repo already exists
        token: Optional[str] = None,
    ):
        log.error(
            "`push_to_hub()` api cannot be used on the model instance. Please use `NANOMODEL.push_to_hub()` static api instead."
        )

    def save(
        self,
        save_dir: str,
        safetensors_metadata: Optional[Dict[str, str]] = None,
        max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
        meta_quantizer: Optional[str] = None,
        **kwargs,
    ):
        timer = getattr(self, "quant_region_timer", None)
        start_time = time.perf_counter() if timer else None

        try:
            if self.quantized:
                # Safetensors is unable to save tied weights, so we untie them here. Reference: https://github.com/huggingface/safetensors/issues/202
                # untie_weights(self.model)

                self.save_quantized(
                    save_dir=save_dir,
                    safetensors_metadata=safetensors_metadata,
                    max_shard_size=max_shard_size,
                    meta_quantizer=meta_quantizer,
                )

                # overwrite quant_override_files
                for name, value in self.quant_override_files.items():
                    json_path = os.path.join(save_dir, name)
                    with open(json_path, "w", encoding="utf-8") as f:
                        if isinstance(value, str):
                            f.write(value)
                        else:
                            f.write(json.dumps(value))
            else:
                self.save_pretrained(save_dir=save_dir, **kwargs)
        finally:
            if timer is not None and start_time is not None:
                try:
                    target = os.path.abspath(save_dir)
                except (TypeError, ValueError, OSError):
                    target = str(save_dir)
                timer.record(
                    "model_save",
                    time.perf_counter() - start_time,
                    source=target,
                )
                timer.flush()

    def kernels(self) -> List[Type[BaseQuantLinear]]:
        """Returns a list of unique qlinear kernel types currently loaded in the model."""
        if not isinstance(self.model, nn.Module):
            return []

        loaded_kernels = {
            v.__class__
            for v in find_modules(self.model, layers=[BaseQuantLinear]).values()
        }
        return list(loaded_kernels)

    def _auto_configure_lookahead(self) -> None:
        """Automatically configures lookahead optimization for TorchQuantLinear modules if enabled."""
        if not isinstance(self.model, nn.Module):
            return

        quant_modules = [
            m for m in self.model.modules() if isinstance(m, TorchQuantLinear)
        ]
        if not quant_modules or not any(
            getattr(m, "_lookahead_enabled", False) for m in quant_modules
        ):
            return

        configure_default_lookahead(self.model)

    def optimize(
        self, backend: str = "inductor", mode: str = None, fullgraph: bool = False
    ):
        """
        Optimizes the quantized model using torch.compile.

        Args:
            backend (str): The backend to use for compilation (e.g., "inductor").
            mode (str): The compilation mode (e.g., "reduce-overhead").
            fullgraph (bool): Whether to attempt to compile the entire model into a single graph.
        """
        if not self.quantized:
            log.warning("Model is not quantized; skipping compilation.")
            return self

        if TORCH_HAS_COMPILE:
            log.warning(
                "torch.compile is not available. Please upgrade to PyTorch 2.6.0 or newer."
            )
            return self

        # First, optimize individual qlinear modules.
        log.info(f"Compiling qlinear modules with backend: `{backend}`, mode: `{mode}`")
        modules = find_modules(self.model, layers=[BaseQuantLinear])
        for name, module in modules.items():
            module.optimize(fullgraph=False, backend=backend, mode=mode)

        # Then, compile the entire model.
        log.info(f"Compiling the full model with backend: `{backend}`, mode: `{mode}`")
        self.model = torch_compile(
            self.model, fullgraph=fullgraph, backend=backend, mode=mode
        )
        self.compiled = True

        return self

    def serve(self, host: str = "0.0.0.0", port: int = 80, async_mode: bool = False):
        """Starts an OpenAI-compatible API server for the model."""
        from ..utils.openai_server import OpenAiServer

        self.server = OpenAiServer(model=self)
        self.server.start(host=host, port=port, async_mode=async_mode)

    def serve_shutdown(self):
        """Shuts down the running API server."""
        if self.server is not None:
            self.server.shutdown()

    def serve_wait_until_ready(self, timeout: int = 30, check_interval: float = 0.1):
        """Waits until the API server is ready to accept requests."""
        if self.server is not None:
            self.server.wait_until_ready(timeout=timeout, check_interval=check_interval)

    def before_model_load(self, load_quantized_model):
        """Hook executed before the model is loaded."""
        pass

    def after_model_load(self, model, load_quantized_model):
        """Hook executed after the model is loaded."""
        return model

    def pre_quantize_generate_hook_start(self):
        """Hook executed at the start of the pre-quantization generation phase."""
        pass

    def pre_quantize_generate_hook_end(self):
        """Hook executed at the end of the pre-quantization generation phase."""
        if self.quantize_config.offload_to_disk:
            offload_to_disk(
                model=self.model,
                module=self.get_base_modules(model=self.model),
                disk_path=self.quantize_config.offload_to_disk_path,
            )

    def lm_head_pre_quantize_generate_hook(
        self, inputs: List[List[torch.tensor]]
    ) -> List[List[torch.tensor]]:
        """Hook to process inputs just before the LM head during pre-quantization."""
        if self.pre_lm_head_norm_module:
            norm, _ = get_module_by_name_prefix(
                self.model, [self.pre_lm_head_norm_module]
            )
            norm = self.pre_quantize(norm)

            for element in inputs:
                for i in range(len(element)):
                    element[i] = norm(element[i])

            self.post_quantize(norm)
        return inputs

    def pre_quantize(self, module: nn.Module) -> nn.Module:
        """Prepares a module for quantization, materializing it from the turtle model if needed."""
        if get_device(module) == META:
            return self.shell_module_materialize(
                target_submodule=module,
                device=self.quantize_config.device,
            )
        elif get_device(module) == CPU and self.quantize_config.device != CPU:
            return move_to(module, device=self.quantize_config.device)
        else:
            return module

    def post_quantize(self, module: nn.Module) -> nn.Module:
        """Cleans up a module after quantization, typically by moving it to CPU."""
        return move_to(module, device=CPU)

    def move_embed(self, device: str):
        """Moves embedding modules to the specified device."""
        for embed_module_name in self.get_base_modules(self.model):
            embed_module, _ = get_module_by_name_prefix(self.model, embed_module_name)
            if embed_module is not None:
                self.shell_module_materialize(
                    target_submodule=embed_module,
                    device=device,
                )

    def awq_skip_modules_for_scaling(self) -> bool:
        """Hook to determine if certain modules should be skipped for AWQ scaling."""
        pass

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        """
        Identifies and groups modules within a layer for AWQ (Activation-aware Weight Quantization) scaling.

        This method iterates through the model's layers as defined by `full_layer_modules`
        and creates a list of "nodes". Each node represents a group of linear layers (a "subset")
        that should be scaled together, along with their preceding operation (e.g., a LayerNorm)
        and the input features required for scaling.

        Args:
            module: The current layer module (e.g., a single Transformer block).
            input_feat: A dictionary mapping module names to their captured input features.
            module_kwargs: Keyword arguments to be passed to the scaling function.

        Returns:
            A list of nodes, where each node is a dictionary containing information for scaling.
        """
        nodes = []
        last_module = None  # The most recent non-quantized module, typically a normalization layer.
        last_module_name = None
        last_module_root = (
            None  # The root of the last module (e.g., 'self_attn' or 'mlp').
        )

        num_experts = (
            self.get_num_experts(self.model.config)
            if self.dynamic_expert_index
            else None
        )

        def strip_not_quantize_flag(module_name):
            return module_name.split(NOT_QUANTIZE_FLAG)[0]

        # Iterate over blocks of modules defined in the model's architecture.
        for block in self.full_layer_modules(self.model.config, is_awq_quantize=True):
            # A block can be a list of module names, e.g., ['self_attn.q_proj', 'self_attn.k_proj']
            # or a norm layer, e.g., ['post_attention_layernorm:!']

            # If all modules in the block are marked as non-quantizable, it's likely a norm layer.
            if all(NOT_QUANTIZE_FLAG in name for name in block):
                last_module_name = strip_not_quantize_flag(block[-1])
                last_module, _ = get_module_by_name_prefix(module, last_module_name)
                continue

            # Handle MoE (Mixture of Experts) blocks where one norm precedes multiple expert layers.
            if (
                num_experts is not None
                and len(block) == num_experts
                and last_module is not None
            ):
                target_suffix = last_module_name.split(".")[-1]
                for name in block:
                    # Find the corresponding preceding operation for each expert.
                    prev_op_name = ".".join(name.split(".")[:-1] + [target_suffix])
                    prev_op, _ = get_module_by_name_prefix(module, prev_op_name)
                    assert prev_op is not None, (
                        f"Could not find prev_op: {prev_op_name}"
                    )

                    m, _ = get_module_by_name_prefix(module, name)
                    n, _ = generate_node_for_awq_scaling(
                        inp=input_feat[name],
                        prev_op=prev_op,
                        module_kwargs=module_kwargs,
                        nodes_size=len(nodes),
                        subset=[m],
                        module2inspect=None,
                    )
                    nodes.append(n)
            else:
                # Handle regular, non-MoE blocks.
                subset = []
                skip_block = False
                for name in block:
                    if NOT_QUANTIZE_FLAG in name:
                        continue

                    # AWQ-specific logic to skip certain modules like 'mlp.gate'
                    if name == "mlp.gate":
                        skip_block = True
                        break

                    m, _ = get_module_by_name_prefix(module, name)

                    # Skip attention output projection if its shape mismatches the previous norm (e.g., in GQA).
                    if (
                        self.awq_scale_optimize_shape_dependent_modules
                        and name in self.awq_scale_optimize_shape_dependent_modules
                        and isinstance(last_module, nn.Linear)
                        and last_module.weight.shape != m.weight.shape
                    ):
                        skip_block = True
                        break
                    subset.append(m)

                if skip_block or not subset:
                    continue

                assert last_module is not None, (
                    "prev_op (last_module) not found for a quantizable block."
                )

                # Determine the root module to inspect for inputs (e.g., 'self_attn').
                root = block[0].split(".")[0]
                module2inspect = None
                if root != last_module_root:
                    last_module_root = root
                    module2inspect, _ = get_module_by_name_prefix(module, root)

                # For MoE, the input might come from the root of the expert block.
                if num_experts and len(block) == 2 * num_experts and module2inspect:
                    inp = input_feat[last_module_root]
                else:
                    inp = input_feat[block[0]]

                n, _ = generate_node_for_awq_scaling(
                    inp=inp,
                    prev_op=last_module,
                    module_kwargs=module_kwargs,
                    nodes_size=len(nodes),
                    subset=subset,
                    module2inspect=module2inspect,
                )
                nodes.append(n)

            # Update the last seen module to the final one in the current block.
            last_module_name = strip_not_quantize_flag(block[-1])
            last_module, _ = get_module_by_name_prefix(module, last_module_name)

        return nodes

    def _clone_model_init_kwargs(self, source: PreTrainedModel) -> Dict[str, Any]:
        kwargs = getattr(source, "_model_init_kwargs", {}) or {}
        if isinstance(kwargs, dict):
            return dict(kwargs)
        return copy.deepcopy(kwargs)

    def _resolve_turtle_reload_threshold(self) -> int:
        if not getattr(self.quantize_config, "offload_to_disk", False):
            return 0

        default_bytes = 512 * 1024**3  # 512MB
        raw = os.getenv("NANOMODEL_RELOAD_THRESHOLD")
        if raw is None or raw.strip() == "":
            return default_bytes

        value = raw.strip().lower()
        if value in {"0", "off", "disable", "disabled", "none"}:
            return 0

        units = {
            "b": 1,
            "kb": 1024,
            "mb": 1024**2,
            "gb": 1024**3,
            "tb": 1024**4,
        }

        match = re.match(r"^([0-9]*\.?[0-9]+)\s*([a-z]*)$", value)
        if match is None:
            log.warn(
                "NANOMODEL_RELOAD_THRESHOLD value `%s` is invalid; defaulting to 512MB.",
                raw,
            )
            return default_bytes

        amount = float(match.group(1))
        unit = match.group(2) or "b"
        multiplier = units.get(unit, None)
        if multiplier is None:
            log.warn(
                "NANOMODEL_RELOAD_THRESHOLD unit `%s` is unsupported; defaulting to bytes.",
                unit,
            )
            multiplier = 1

        threshold = int(amount * multiplier)
        if threshold < 0:
            threshold = 0
        return threshold

    def _estimate_module_bytes(self, module: nn.Module) -> int:
        if module is None:
            return 0

        total = 0
        seen: Set[int] = set()
        tensors = list(module.parameters(recurse=True)) + list(
            module.buffers(recurse=True)
        )
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type == "meta":
                continue
            try:
                ptr = tensor.data_ptr()
            except (RuntimeError, AssertionError):
                ptr = None
            if ptr is not None:
                if ptr in seen:
                    continue
                seen.add(ptr)
            total += tensor.numel() * tensor.element_size()
        return total

    def _maybe_auto_reload_after_alias(
        self,
        module: nn.Module,
        target_submodule: nn.Module,
    ) -> None:
        """
        After materializing a module, check if the memory threshold has been exceeded
        and trigger a reload of the turtle model if necessary.
        """
        if self.turtle_model is None or self._turtle_reload_threshold_bytes <= 0:
            return

        module_id = id(module)
        if module_id in self._turtle_materialized_ids:
            return

        bytes_added = self._estimate_module_bytes(module)
        self._turtle_materialized_ids.add(module_id)

        if bytes_added <= 0:
            return

        self._turtle_reload_accum_bytes += bytes_added

        if self._turtle_reload_accum_bytes >= self._turtle_reload_threshold_bytes:
            label = (
                getattr(target_submodule, "full_name", None)
                or getattr(target_submodule, "name", None)
                or getattr(module, "full_name", None)
                or module.__class__.__name__
            )
            log.info(
                f"Memory threshold reached. Reloading turtle model to free RAM (triggered by: {label})."
            )
            self.reload_turtle_model(source=f"auto:{label}")

    def reload_turtle_model(self, *, source: Optional[str] = None) -> None:
        """
        Reloads the 'turtle model' (the meta-device model) from disk to free up
        CPU memory consumed by materialized modules.
        """
        if not self.quantize_config.offload_to_disk:
            return

        timer = getattr(self, "quant_region_timer", None)
        timing_ctx = (
            timer.measure("model_reload", source=source) if timer else nullcontext()
        )

        with timing_ctx:

            def _do_reload():
                with self._turtle_lock:
                    if self.turtle_model is None or self.model_local_path is None:
                        return

                    # Clone init kwargs and config before deleting the old model
                    reload_kwargs = self._clone_model_init_kwargs(self.turtle_model)
                    config = self.turtle_model.config
                    del self.turtle_model

                    # Reload the model from the pretrained path with low memory usage
                    new_model = self.loader.from_pretrained(
                        self.model_local_path,
                        config=config,
                        low_cpu_mem_usage=True,
                        **reload_kwargs,
                    )
                    new_model._model_init_kwargs = reload_kwargs
                    new_model.eval()
                    self.turtle_model = new_model
                    self._turtle_reload_accum_bytes = 0

            reload_spinner = log.spinner(
                title="Turtle model reloading...", interval=0.1
            )
            try:
                # Run reload in a dedicated thread to avoid blocking
                DEVICE_THREAD_POOL.submit("model_loader:cpu", _do_reload).result()
            finally:
                reload_spinner.close()

    def shell_module_materialize(
        self,
        target_submodule: torch.nn.Module,
        device: torch.device,
        non_blocking: bool = False,
    ) -> torch.nn.Module:
        """
        Materializes a submodule from the 'shell' (the main model) by copying weights
        from the 'turtle' (the meta-device model) and moving it to the target device.
        """
        with self._turtle_lock:
            if self.turtle_model is None:
                # If no turtle model, just move the existing submodule to the device
                if get_device(target_submodule) != device:
                    target_submodule.to(device)
                return target_submodule

            # Create an alias, which copies parameters from turtle to shell
            module = alias_from_turtle_for_submodule(
                target_model=self.model,
                turtle_model=self.turtle_model,
                target_submodule=target_submodule,
                device=device,
            )

        # Check if reloading the turtle model is necessary after this operation
        self._maybe_auto_reload_after_alias(module, target_submodule)
        return module

    @classmethod
    def build_layer_modules(cls, tree: List[Union[str, Dict]]) -> List[List[str]]:
        """
        Parses the `module_tree` to build a structured list of module names for each layer.

        The tree format is a list containing module names and a dictionary that defines
        the structure of quantizable layers.
        e.g., ["model", "layers", "#", { "parent_module": ("child[:!][:grp]", ...), ... }]

        Rules:
          - ':!': Marks a module as non-quantizable (e.g., a norm layer).
          - ':<digit>': Assigns a module to a group. Modules with the same group ID are processed together.
          - Nested dicts: Define complex structures, like in MoE models.
          - EXPERT_INDEX_PLACEHOLDER: A placeholder replaced with expert indices for MoE layers.

        Returns:
          A list of lists, where each inner list is a block of module names to be processed together.
        """
        mapping = next((item for item in tree if isinstance(item, dict)), None)
        if mapping is None:
            raise ValueError("Mapping configuration not found in the module tree.")

        out_blocks = []

        def process_entries(
            parent: str, entries: Union[tuple, list, dict], parent_group_offset: int = 0
        ) -> defaultdict[int, list]:
            """Recursively process module entries to handle groups and nested structures."""
            groups = defaultdict(list)

            if isinstance(entries, (tuple, list)):
                # Base case: a list/tuple of module names
                for ent in entries:
                    parts = ent.split(":")
                    child = parts[0]
                    flags = parts[1:]
                    has_bang = "!" in flags
                    grp = (
                        next((int(p) for p in flags if p.isdigit()), 0)
                        + parent_group_offset
                    )

                    full_path = (
                        f"{parent}.{child}" if parent and parent != child else child
                    )
                    groups[grp].append((full_path, has_bang))

            elif isinstance(entries, dict):
                # Recursive case: a dictionary defining a nested structure (like MoE)
                max_current_group = 0
                for sub_entries in entries.values():
                    if isinstance(sub_entries, (tuple, list)):
                        for ent in sub_entries:
                            grp = next(
                                (int(p) for p in ent.split(":")[1:] if p.isdigit()), 0
                            )
                            max_current_group = max(max_current_group, grp)

                current_offset = parent_group_offset
                for sub_parent, sub_entries in entries.items():
                    if sub_parent == "#":
                        # MoE expert placeholder
                        template_parent = f"{parent}.{EXPERT_INDEX_PLACEHOLDER}"
                        expert_offset = (
                            current_offset + max_current_group + 100
                        )  # Use a large offset to avoid group collisions
                        sub_groups = process_entries(
                            template_parent, sub_entries, expert_offset
                        )
                    else:
                        # Regular nested module
                        full_sub_parent = (
                            f"{parent}.{sub_parent}"
                            if parent and sub_parent
                            else parent or sub_parent
                        )
                        sub_groups = process_entries(
                            full_sub_parent, sub_entries, current_offset
                        )

                    for grp, items in sub_groups.items():
                        groups[grp].extend(items)
                    if sub_groups:
                        current_offset = max(sub_groups.keys()) + 1

            return groups

        for parent, entries in mapping.items():
            groups = process_entries(parent, entries)
            # Sort by group ID and create the final blocks of module names
            for g in sorted(groups.keys()):
                block = [
                    path + (NOT_QUANTIZE_FLAG if has_bang else "")
                    for path, has_bang in groups[g]
                ]
                out_blocks.append(block)

        return out_blocks

    @classmethod
    def get_base_modules(cls, model):
        """
        Return list of base modules directly under 'model' but not 'model.layers'.
        """
        # Find the index of "#"
        tree = cls.module_tree
        try:
            sharp_idx = tree.index("#")
        except ValueError:
            raise ValueError("module_tree must contain '#' to separate hierarchy")

        assert sharp_idx > 0, "failed to get_base_modules"
        # root_path = ["model"] or ["model", "language_model"]
        root_path = tree[: sharp_idx - 1]

        out = []
        # Traverse each layer in root_path
        for i in range(len(root_path)):
            path = root_path[: i + 1]
            base = model
            exclude = tree[len(path)]

            for node in path:
                base = getattr(base, node)

            for name, _ in base.named_children():
                if name != exclude:
                    out.append(".".join(path + [name]))

        # print(f"Base Modules: {out}")
        return out

    def generate_layers_modules_tree_simple(self, node):
        """
        Recursively walk a nested list/dict structure and:
          1. Drop dict entries where *all* values are ':!' flagged.
          2. Remove ':!' and ':<digit>' markers from strings.
        """

        # If it's a list, recurse into each element
        if isinstance(node, list):
            return [self.generate_layers_modules_tree_simple(x) for x in node]

        # If it's a dict, process each key -> value
        if isinstance(node, dict):
            new_dict = {}
            for k, v in node.items():
                # Expand tuple-of-strings blocks (special handling)
                if isinstance(v, (tuple, list)) and all(isinstance(x, str) for x in v):
                    # Rule 1: check if ALL entries are :!
                    if all(any(p == "!" for p in x.split(":")[1:]) for x in v):
                        continue  # skip this parent entirely

                    # Rule 2: strip :! and :digit markers
                    cleaned = tuple(x.split(":")[0] for x in v)
                    new_dict[k] = cleaned
                else:
                    # Recurse deeper
                    new_dict[k] = self.generate_layers_modules_tree_simple(v)
            return new_dict

        # If it's a plain string (unlikely here), strip markers
        if isinstance(node, str):
            return node.split(":")[0]

        # For other types, return as-is
        return node

    def tied_word_embedding(self) -> bool:
        return getattr(self.model.config, "tie_word_embeddings", False)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception as exc:  # torch Modules raise AttributeError here
            model = self.__dict__.get("model")
            if model is None:
                model = (
                    self._modules.get("model") if hasattr(self, "_modules") else None
                )
            if model is not None and item != "model":
                return getattr(model, item)
            raise exc


__all__ = ["BaseNanoModel"]

BaseNanoModel = ModelLoader(ModelWriter(BaseNanoModel))
