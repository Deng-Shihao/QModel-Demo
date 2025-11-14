from __future__ import annotations

import copy
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from contextlib import nullcontext
from itertools import count
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type, Union

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
from ..utils.calibration import prepare_calibration_dataset
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
CAPTURE_ONLY_FLAG = ":?"
NON_QUANTIZE_FLAGS = (NOT_QUANTIZE_FLAG, CAPTURE_ONLY_FLAG)

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

    # prefixes that identify expert modules in MoE architectures
    moe_expert_module_name_prefixes: List[str] = [".expert", ".experts"]

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
        cls,
        model_config,
        quantize_config,
        is_awq_quantize: bool = False,
        include_capture_only: bool = False,
    ) -> List[List[str]]:
        """
        Builds a simplified list of layer modules for quantization.

        This list is created by building the full module list, expanding for MoE if needed,
        and then filtering out non-quantizable modules. This is the primary method
        used during the quantization process.
        """
        layer_modules = cls.build_layer_modules(
            cls.module_tree, include_capture_only=include_capture_only
        )
        layer_modules = cls.build_moe_modules_if_need(
            model_config, layer_modules, is_awq_quantize
        )
        layer_modules = cls.filter_not_quantize_module(layer_modules, quantize_config)
        return layer_modules

    @classmethod
    def full_layer_modules(
        cls,
        model_config=None,
        is_awq_quantize: bool = False,
        include_capture_only: bool = False,
    ) -> List[List[str]]:
        """
        Builds the complete list of layer modules, including non-quantizable ones.

        This is useful for operations that need to inspect the entire model structure,
        such as AWQ scaling.
        """
        full = cls.build_layer_modules(
            cls.module_tree, include_capture_only=include_capture_only
        )
        full = cls.build_moe_modules_if_need(model_config, full, is_awq_quantize)
        return full

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
        calibration_concat_separator: Optional[str] = CALIBRATION_DATASET_CONCAT_CHAR,
    ):
        """
        Tokenize/normalize calibration samples and collate them into batches.

        The heavy lifting is delegated to `prepare_calibration_dataset` so the logic stays
        in sync with GPTQModel's reference implementation.
        """
        return prepare_calibration_dataset(
            model=self,
            calibration_dataset=calibration_dataset,
            calibration_dataset_concat_size=calibration_dataset_concat_size,
            calibration_dataset_sort=calibration_dataset_sort,
            batch_size=batch_size,
            calibration_data_min_length=calibration_data_min_length,
            calibration_concat_separator=calibration_concat_separator,
            logger=log,
        )


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
        calibration_concat_separator: Optional[str] = CALIBRATION_DATASET_CONCAT_CHAR,
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
            calibration_concat_separator: Optional separator token/string when concatenating samples.

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
            "calibration_data_min_length": calibration_data_min_length,
            "calibration_concat_separator": calibration_concat_separator,
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
        Mirror GPTQModel's AWQ scaling node construction logic, supporting capture-only
        markers, expert-aware grouping, and better diagnostics.
        """
        nodes = []
        last_module = None  # most recent norm object
        last_module_name = None
        last_module_root = None  # e.g., "self_attn" or "mlp"

        if self.model.config is not None and self.dynamic_expert_index is not None:
            self.get_num_experts(self.model.config)

        def strip_non_quantize_flags(module_name: str) -> str:
            for flag in NON_QUANTIZE_FLAGS:
                if flag in module_name:
                    module_name = module_name.replace(flag, "")
            return module_name

        def _select_feature_name(names: List[str]) -> Optional[str]:
            """Return the first quantized child that has captured activations."""
            for raw in names:
                stripped = strip_non_quantize_flags(raw)
                if stripped in input_feat:
                    return stripped
            return strip_non_quantize_flags(names[0]) if names else None

        def _try_update_last_module(candidate_name: str) -> bool:
            nonlocal last_module, last_module_name, last_module_root

            resolved_module, _ = get_module_by_name_prefix(module, candidate_name)
            if resolved_module is None:
                log.debug(
                    "awq_get_modules_for_scaling: last-module candidate `%s` missing; retaining previous `%s`",
                    candidate_name,
                    last_module_name,
                )
                return False

            last_module = resolved_module
            last_module_name = candidate_name
            if "." in candidate_name:
                last_module_root = candidate_name.split(".", 1)[0]
            return True

        full_layer_modules = self.full_layer_modules(
            self.model.config,
            is_awq_quantize=True,
            include_capture_only=True,
        )
        for i, block in enumerate(full_layer_modules):
            not_quantized = all(
                any(flag in name for flag in NON_QUANTIZE_FLAGS) for name in block
            )
            if not_quantized:
                if (
                    i > 0
                    and all(
                        any(flag in name for flag in NON_QUANTIZE_FLAGS)
                        for name in full_layer_modules[i - 1]
                    )
                ):
                    continue

                candidate_name = strip_non_quantize_flags(block[-1])
                _try_update_last_module(candidate_name)
                continue

            is_moe_block = any(
                any(k in name for k in self.moe_expert_module_name_prefixes)
                for name in block
            )
            is_moe_down_block = is_moe_block and any("down" in name for name in block)
            is_moe_gate_up_block = is_moe_block and any("gate" in name for name in block) and any(
                "up" in name for name in block
            )
            if is_moe_down_block and last_module is not None and last_module_name is not None:
                target_suffix = last_module_name.split(".")[-1]
                for name in block:
                    prev_op_name = ".".join(name.split(".")[:-1] + [target_suffix])
                    prev_op, _ = get_module_by_name_prefix(module, prev_op_name)
                    if prev_op is None or strip_non_quantize_flags(name) not in input_feat:
                        log.debug(
                            "awq_get_modules_for_scaling: skipping expert `%s` due to missing prev_op or features",
                            name,
                        )
                        continue

                    m, _ = get_module_by_name_prefix(module, name)
                    if m is None:
                        log.debug(
                            "awq_get_modules_for_scaling: skipping missing expert module `%s`",
                            name,
                        )
                        continue
                    subset = [m]
                    n, root = generate_node_for_awq_scaling(
                        inp=input_feat[strip_non_quantize_flags(name)],
                        prev_op=prev_op,
                        module_kwargs=module_kwargs,
                        nodes_size=len(nodes),
                        subset=subset,
                        module2inspect=None,
                    )
                    if root is not None and last_module_root != root:
                        last_module_root = root

                    nodes.append(n)
            else:
                subset = []
                skip = False
                for name in block:
                    if all(flag not in name for flag in NON_QUANTIZE_FLAGS):
                        m, _ = get_module_by_name_prefix(module, name)

                        if (
                            self.awq_scale_optimize_shape_dependent_modules is not None
                            and name in self.awq_scale_optimize_shape_dependent_modules
                            and isinstance(last_module, nn.Linear)
                            and last_module.weight.shape != m.weight.shape
                        ):
                            skip = True

                        if m is None:
                            log.debug(
                                "awq_get_modules_for_scaling: skipping missing module `%s`",
                                name,
                            )
                            skip = True
                            break
                        subset.append(m)

                if skip or not subset:
                    continue

                prev_op = last_module
                if prev_op is None:
                    log.debug(
                        "awq_get_modules_for_scaling: skipping block %s due to missing previous module",
                        block,
                    )
                    continue

                feature_name = _select_feature_name(block) or strip_non_quantize_flags(block[0])
                root_split = feature_name.split(".")
                module2inspect = None
                if len(root_split) >= 2:
                    root = root_split[0]
                    if root != last_module_root:
                        last_module_root = root
                        module2inspect, _ = get_module_by_name_prefix(module, root)

                if is_moe_gate_up_block and module2inspect is not None:
                    if last_module_root not in input_feat:
                        log.debug(
                            "awq_get_modules_for_scaling: missing input feature for `%s` while processing experts block (layer block size=%s)",
                            last_module_root,
                            len(block),
                        )
                    inp = input_feat.get(last_module_root, input_feat.get(_select_feature_name(block)))
                else:
                    inp = input_feat.get(feature_name)

                if inp is None:
                    log.debug(
                        "awq_get_modules_for_scaling: skipping block %s due to missing input features",
                        block,
                    )
                    continue

                n, root = generate_node_for_awq_scaling(
                    inp=inp,
                    prev_op=prev_op,
                    module_kwargs=module_kwargs,
                    nodes_size=len(nodes),
                    subset=subset,
                    module2inspect=module2inspect,
                )

                nodes.append(n)

            if is_moe_gate_up_block:
                gate_up_proj_indices = [
                    idx
                    for idx, name in enumerate(block)
                    if any(k in name for k in self.moe_expert_module_name_prefixes)
                    and ("gate" in name or "up" in name)
                ]

                assert gate_up_proj_indices, "No expert gate_proj/up_proj found in block."
                last_up_proj_index = gate_up_proj_indices[-1]

                candidate_name = strip_non_quantize_flags(block[last_up_proj_index])
                assert "gate" in candidate_name or "up" in candidate_name
            else:
                candidate_name = strip_non_quantize_flags(block[-1])
            _try_update_last_module(candidate_name)

        import torch

        def format_nodes(nodes):
            out = []
            for n in nodes:
                entry = {}
                for k, v in n.items():
                    if isinstance(v, torch.Tensor):
                        entry[k] = f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype})"
                    elif isinstance(v, dict):
                        entry[k] = [
                            f"Key: {kk}, Value: Tensor(shape={tuple(x.shape)}, dtype={x.dtype}); "
                            if isinstance(x, torch.Tensor)
                            else type(x).__name__
                            for kk, x in v.items()
                        ]
                    else:
                        entry[k] = v
                out.append(entry)
            return out

        # print("DEBUG AWQ NODES:", format_nodes(nodes))
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
    def build_layer_modules(
        cls, tree: List[Union[str, Dict]], include_capture_only: bool = False
    ) -> List[List[str]]:
        """
        Parse the `module_tree` definition to build execution-ordered layer blocks.

        The structure mirrors GPTQModel's builder and supports additional annotations:
          - ':!' keeps the module for activation capture but skips quantization.
          - ':?' marks capture-only nodes; only included when include_capture_only=True.
          - ':<digit>' assigns modules to a shared processing group.
          - Nested dicts describe MoE hierarchies and expert aliases.
        """

        mapping = next((item for item in tree if isinstance(item, dict)), None)
        if mapping is None:
            raise ValueError("Mapping configuration not found in the module tree.")

        out_blocks: List[List[str]] = []
        alias_groups: Dict[tuple[Optional[str], int], List[tuple[str, bool, bool]]] = {}
        alias_meta: Dict[tuple[Optional[str], int], Dict[str, int]] = {}
        alias_seq = count()
        group_seq = count()

        def _parse_token(token: str) -> tuple[str, List[str]]:
            parts = token.split(":")
            name = parts[0]
            flags = [p for p in parts[1:] if p]
            return name, flags

        def _group_from_flags(flags: List[str]) -> int:
            for flag in flags:
                if flag.isdigit():
                    return int(flag)
            return 0

        def _has_numeric_flag(flags: List[str]) -> bool:
            return any(flag.isdigit() for flag in flags)

        def _get_scope(parent_name: str) -> Optional[str]:
            if not parent_name:
                return None
            return parent_name.split(".", 1)[0]

        def process_entries(
            parent_token: str,
            entries: Union[tuple, list, dict],
            parent_group_offset: int = 0,
            scope_key: Optional[str] = None,
        ) -> defaultdict[int, List[tuple]]:
            """Recursively process module entries to handle groups and nested structures."""
            groups: defaultdict[int, List[tuple]] = defaultdict(list)

            parent_name, parent_flags = _parse_token(parent_token)
            parent_rel_group = _group_from_flags(parent_flags)
            parent_group = parent_group_offset + parent_rel_group
            parent_has_bang = "!" in parent_flags
            parent_capture_only = "?" in parent_flags
            parent_has_numeric = _has_numeric_flag(parent_flags)

            scope = scope_key if scope_key is not None else _get_scope(parent_name)
            parent_alias_scope = scope if parent_has_numeric else parent_name

            def _make_entry(
                full_path: str,
                has_bang: bool,
                capture_only: bool,
                *,
                alias_base: int,
                alias_rel: int,
                alias_scope: Optional[str],
            ) -> tuple:
                return (full_path, has_bang, capture_only, alias_scope, (alias_base, alias_rel))

            child_group_offset = parent_group_offset
            add_parent = parent_has_bang or (parent_capture_only and include_capture_only)
            if add_parent:
                alias_base = parent_rel_group if parent_has_numeric else parent_group
                parent_entry_scope = (
                    f"{parent_alias_scope}.__parent__" if parent_alias_scope is not None else None
                )
                groups[parent_group].append(
                    _make_entry(
                        parent_name,
                        parent_has_bang,
                        parent_capture_only,
                        alias_base=alias_base,
                        alias_rel=0,
                        alias_scope=parent_entry_scope,
                    )
                )
                child_group_offset = max(child_group_offset, parent_group + 1)

            if isinstance(entries, (tuple, list)):
                for ent in entries:
                    child_name, child_flags = _parse_token(ent)

                    has_bang = "!" in child_flags
                    capture_only = "?" in child_flags
                    child_rel_group = _group_from_flags(child_flags)
                    grp = child_group_offset + child_rel_group

                    if parent_name.endswith(f".{child_name}") or parent_name == child_name:
                        full_path = parent_name
                    elif parent_name:
                        full_path = f"{parent_name}.{child_name}"
                    else:
                        full_path = child_name

                    alias_scope = parent_alias_scope if parent_has_numeric else scope
                    alias_base = child_rel_group if _has_numeric_flag(child_flags) else grp
                    entry = _make_entry(
                        full_path,
                        has_bang,
                        capture_only,
                        alias_base=alias_base,
                        alias_rel=0,
                        alias_scope=alias_scope,
                    )
                    groups[grp].append(entry)

            elif isinstance(entries, dict):
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
                        template_parent = f"{parent_name}.{EXPERT_INDEX_PLACEHOLDER}"
                        template_parent_token = f"{template_parent}:{parent_rel_group}"
                        alias_scope = scope or parent_name
                        alias_base = parent_rel_group if parent_has_numeric else parent_group
                        expert_offset = current_offset + max_current_group + 100
                        if isinstance(sub_entries, (tuple, list)):
                            groups[expert_offset].append(
                                _make_entry(
                                    template_parent,
                                    False,
                                    False,
                                    alias_base=alias_base,
                                    alias_rel=0,
                                    alias_scope=alias_scope,
                                )
                            )
                        else:
                            sub_groups = process_entries(
                                template_parent_token, sub_entries, expert_offset, scope
                            )
                            for grp, items in sub_groups.items():
                                groups[grp].extend(items)
                    else:
                        if sub_parent == "":
                            full_sub_parent = parent_name
                        else:
                            full_sub_parent = (
                                f"{parent_name}.{sub_parent}" if parent_name else sub_parent
                            )
                        sub_groups = process_entries(
                            full_sub_parent, sub_entries, current_offset, scope
                        )
                        for grp, items in sub_groups.items():
                            groups[grp].extend(items)
                        if sub_groups:
                            current_offset = max(sub_groups.keys()) + 1

            return groups

        def _register_alias(order_idx: int, item: tuple):
            full_path, has_bang, capture_only, scope, alias_parts = item
            if capture_only and not include_capture_only:
                return
            alias_scope = scope
            alias_base, alias_rel = alias_parts
            alias_index = alias_base + alias_rel
            key = (alias_scope, alias_index)
            meta = alias_meta.get(key)
            if meta is None:
                alias_meta[key] = {"order": order_idx, "seq": next(alias_seq)}
                alias_groups[key] = [(full_path, has_bang, capture_only)]
            else:
                meta["order"] = min(meta["order"], order_idx)
                alias_groups[key].append((full_path, has_bang, capture_only))

        for parent, entries in mapping.items():
            groups = process_entries(parent, entries)

            for g in sorted(groups):
                order_idx = next(group_seq)
                items = groups[g]
                for item in items:
                    if len(item) == 3:
                        full_path, has_bang, capture_only = item
                        scope = full_path
                        alias_parts = (g, 0)
                        _register_alias(order_idx, (full_path, has_bang, capture_only, scope, alias_parts))
                    else:
                        _register_alias(order_idx, item)

        for key in sorted(
            alias_groups.keys(), key=lambda k: (alias_meta[k]["order"], alias_meta[k]["seq"])
        ):
            block = []
            for full_path, has_bang, capture_only in alias_groups[key]:
                name = full_path
                if has_bang:
                    name += NOT_QUANTIZE_FLAG
                if capture_only and include_capture_only:
                    name += CAPTURE_ONLY_FLAG
                block.append(name)
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
                if isinstance(v, (tuple, list)) and all(isinstance(x, str) for x in v):
                    if all(any(p in {"!", "?"} for p in x.split(":")[1:]) for x in v):
                        continue
                    cleaned = tuple(x.split(":")[0] for x in v)
                    new_dict[k] = cleaned
                else:
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
