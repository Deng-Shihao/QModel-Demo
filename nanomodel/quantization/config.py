import json
import re
import uuid
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ..utils.logger import setup_logger


log = setup_logger()

DEFAULT_DAMP_PERCENT = 0.05
DEFAULT_DAMP_AUTO_INCREMENT = 0.01
VALID_PACK_DTYPE_NAMES = {
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
}

BITS_FIELD_CODE = "bits"
GROUP_SIZE_FIELD_CODE = "group_size"
FORMAT_FIELD_CODE = "kernel"
FORMAT_FIELD_CHECKPOINT = "checkpoint_format"
FORMAT_FIELD_COMPAT_MARLIN = "is_marlin_format"
QUANT_METHOD_FIELD = "quant_method"
PACK_DTYPE_FIELD = "pack_dtype"
QUANT_CONFIG_FILENAME = "quantize_config.json"
QUANT_CONFIG_FILENAME_COMPAT = [
    QUANT_CONFIG_FILENAME,
    "quant_config.json",
    "config.json",
]

META_FIELD = "meta"
# quantizer is the tool that did the quantization
META_FIELD_QUANTIZER = "quantizer"

META_QUANTIZER_NANOMODEL = "nanomodel"

META_FIELD_URI = "uri"
META_VALUE_URI = "DEMO"

META_FIELD_DAMP_PERCENT = "damp_percent"
META_FIELD_DAMP_AUTO_INCREMENT = "damp_auto_increment"

META_FIELD_STATIC_GROUPS = "static_groups"
META_FIELD_TRUE_SEQUENTIAL = "true_sequential"

META_FIELD_MSE = "mse"
META_FIELD_ACT_GROUP_AWARE = "act_group_aware"

# saved kernels
class KERNEL(str, Enum):
    # GPTQ
    GPTQ = "gptq"
    MARLIN = "marlin"

    # AWQ
    GEMM = "gemm"
    GEMV = "gemv"
    GEMV_FAST = "gemv_fast"


# quant methods
class METHOD(str, Enum):
    GPTQ = "gptq"
    AWQ = "awq"


QUANT_METHOD_FORMAT_MAPPING = {
    METHOD.GPTQ: {
        KERNEL.GPTQ,
        KERNEL.MARLIN,
    },
    METHOD.AWQ: {
        KERNEL.GEMM,
        KERNEL.GEMV,
        KERNEL.GEMV_FAST,
        KERNEL.MARLIN,
    },
}

# inference only methods should go here
QUANTIZE_BLACK_LIST = {}

# compat
QUANT_CONFIG_ARG_SYNONYMS = {
    "w_bit": BITS_FIELD_CODE,
    # QQQ compat
    "wbits": BITS_FIELD_CODE,
    "q_group_size": GROUP_SIZE_FIELD_CODE,
    # AWQ compat
    "version": FORMAT_FIELD_CODE,
    # map format field (checkpoint_format) to class/code (format)
    FORMAT_FIELD_CHECKPOINT: FORMAT_FIELD_CODE,
}


def dict_scale_dtype_to_str(d: Dict[str, Any]) -> None:
    """
    Checks whether the passed dictionary and its nested dicts have a *scale_dtype* key and if it's not None,
    converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
    string, which can then be stored in the json format.
    """
    if d.get("scale_dtype", None) is not None and not isinstance(d["scale_dtype"], str):
        d["scale_dtype"] = str(d["scale_dtype"]).split(".")[1]
    for value in d.values():
        if isinstance(value, dict):
            dict_scale_dtype_to_str(value)


def dynamic_get(
    dynamic: Optional[Dict[str, Dict[str, Union[int, bool]]]],
    module_name: str,
    key: Optional[str] = None,
    default: Union[int, bool, float, None] = None,
    sub_key: Optional[str] = None,
) -> Union[Dict[str, Union[int, bool, float]], int, bool, float, None]:
    """
    Resolve a dynamic override for a module by matching regex patterns in the configuration.

    Patterns prefixed with "+:" behave as inclusive overrides, while "-:" disables matches.
    """
    if not dynamic:
        return default

    for pattern, overrides in dynamic.items():
        normalized_pattern = (
            pattern[2:] if pattern.startswith(("-:", "+:")) else pattern
        )

        if not re.match(normalized_pattern, module_name):
            continue

        if pattern.startswith("-:"):
            return False

        if key is None:
            return overrides

        # subkey example: LoRA override format: `{ "adapter": { "rank": 512 } }`
        if sub_key:
            sub_value = overrides.get(key)
            if isinstance(sub_value, dict):
                return sub_value.get(sub_key, default)
            log.info(
                "QuantConfig: Dynamic `sub_key`: `%s` failed extraction from `sub_value`: `%s`",
                sub_key,
                sub_value,
            )
            continue

        return overrides.get(key, default)

    return default


@dataclass
class QuantizeConfig:
    """Configuration object describing how a checkpoint is (or should be) quantized."""

    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})

    # allow dynamic bitsize per layer, if None or some layer not set, use bits
    dynamic: Optional[Dict[str, Dict[str, Union[int, bool]]]] = field(default=None)

    # 128 offer good balance between inference speed, vram usage (bpw), and quality
    # use 32 for highest quality with slower inference and higher vram usage
    group_size: int = field(default=128)

    # increase damp if NaN is encountered during `.quantize()` and/or increase calib dataset size
    damp_percent: float = field(default=None)
    damp_auto_increment: float = field(default=None)

    act_order: Optional[bool] = field(default=None)
    act_group_aware: Optional[bool] = field(default=None)  # gar
    static_groups: bool = field(default=False)
    sym: bool = field(default=True)
    true_sequential: bool = field(default=True)

    lm_head: bool = field(default=False)

    quant_method: METHOD = field(default=METHOD.GPTQ)
    kernel: KERNEL = field(default=KERNEL.GPTQ)

    # quantization_order: str = "activate",
    # quantization_scale: str = "mse", # or absmax
    # is_distributed: bool = False,
    # tied_gptq_handle: Optional["GPTQ"] = None

    # mean square error calculation: may reduce error loss for some models
    mse: float = field(default=0.0)

    # properties that do not directly contributes to quantization or quant inference should be placed in meta
    # i.e. quantizer tool (producer) + version, timestamp, entity who made the quant, etc
    meta: Optional[Dict] = field(default=None)

    # normalized to DEVICE after passing to load()
    device: Optional[Union[str, torch.device]] = field(default=None)

    # gptq was originally designed to pack quantized weights inside INT32 dtypes
    # allowing using different dtypes used for packing quantized weights
    # affects [`qweights`, `qzeros`]
    pack_dtype: Optional[Union[str, torch.dtype]] = field(default=torch.int32)

    # packing implementation hinpt (`original` = legacy CPU pack, `gpu` enables CUDA pack, `cpu` forces block CPU pack).
    pack_impl: str = field(default="cpu")

    # quantization only:
    # controls cpu memory saving by offloading layers/modules to disk in the slow quantization process
    # default to true as the benefit of ~73.5% cpu memory saving is tremendous
    offload_to_disk: bool = field(
        default=True,
        metadata={
            "help": "Offload completed module memory to disk during quantization loop"
        },
    )
    offload_to_disk_path: str = field(
        default=None,
        metadata={
            "help": "Offload disk path. Only applicable if Offload to disk is enabled"
        },
    )

    rotation: Optional[str] = field(
        default=None, metadata={"choices": ["hadamard", "random"]}
    )

    # deprecated: only used for compat
    is_marlin_format: bool = False

    # use mock quantization to quantize module so the gptq process can continue and not fail
    fail_safe: bool = field(default=False)

    # v2 only:
    v2: bool = field(default=False)
    v2_alpha: float = field(default=0.25)
    v2_memory_device: str = field(default="auto")

    # awq only:
    zero_point: bool = field(default=True)

    # gptq only:
    # skip all heavy computations for testing model loading
    mock_quantization: bool = field(
        default=False,
        metadata={"help": "Skip heavy computations for fast model loading validation"},
    )

    # Hessian accumulation controls (GPTQ only)
    hessian_chunk_size: Optional[int] = field(
        default=None, metadata={"help": "Maximum rows per Hessian chunk"}
    )
    hessian_chunk_bytes: Optional[int] = field(
        default=None,
        metadata={"help": "Memory budget (in bytes) for Hessian chunk staging"},
    )
    hessian_use_bfloat16_staging: bool = field(
        default=False,
        metadata={"help": "Stage Hessian chunks in bfloat16 when supported"},
    )

    def __post_init__(self):
        bits_field = next(field for field in fields(self) if field.name == "bits")
        bits_choices = list(bits_field.metadata.get("choices", []))

        self.pack_dtype = self._normalize_pack_dtype(self.pack_dtype)

        # validate quant method and ensure kernel compatibility
        valid_formats = QUANT_METHOD_FORMAT_MAPPING.get(self.quant_method, None)
        if valid_formats is None:
            raise ValueError(
                f"QuantizeConfig: Unsupported `quant_method`: {self.quant_method}"
            )

        # apply defaults tuned for respective methods
        if self.damp_percent is None:
            self.damp_percent = DEFAULT_DAMP_PERCENT
        if self.damp_auto_increment is None:
            self.damp_auto_increment = DEFAULT_DAMP_AUTO_INCREMENT

        # AWQ checkpoints saved before kernel metadata existed default to GEMM
        if self.quant_method == METHOD.AWQ and self.kernel not in {
            KERNEL.MARLIN,
            KERNEL.GEMV,
            KERNEL.GEMV_FAST,
            KERNEL.GEMM,
        }:
            log.info(
                "QuantizeConfig: Auto fix `format` to `%s` for AWQ checkpoints.",
                KERNEL.GEMM,
            )
            self.kernel = KERNEL.GEMM

        if self.kernel not in valid_formats:
            raise ValueError(
                f"QuantizeConfig: checkpoint `format` used is {self.kernel}, and the quantization method is {self.quant_method}. "
            )

        if bits_choices and self.bits not in bits_choices:
            raise ValueError(
                f"QuantizeConfig: `bits` must be in the set of `{bits_choices}`."
            )

        if self.dynamic is not None:
            self.dynamic = self._normalize_dynamic_overrides(self.dynamic, bits_choices)

        self._validate_group_size(self.group_size)
        self._validate_damp_params()
        self._validate_hessian_config()

        # resolve activation ordering compatibility and defaults
        act_order_user_value = self.act_order
        act_group_aware_user_value = self.act_group_aware

        if act_order_user_value is None:
            # GPTQ defaults to higher quality ordering disabled, others retain legacy default
            self.act_order = False if self.quant_method == METHOD.GPTQ else True
        elif isinstance(act_order_user_value, bool):
            self.act_order = act_order_user_value
        else:
            self.act_order = bool(act_order_user_value)

        if act_group_aware_user_value is None:
            # auto-enable for GPTQ unless user explicitly disables it
            self.act_group_aware = self.quant_method == METHOD.GPTQ
        elif isinstance(act_group_aware_user_value, bool):
            self.act_group_aware = act_group_aware_user_value
        else:
            self.act_group_aware = bool(act_group_aware_user_value)

        self._resolve_activation_ordering(
            act_order_user_value, act_group_aware_user_value
        )

        # validate hybrid act order
        if self.act_group_aware and self.act_order:
            raise ValueError(
                "QuantizeConfig:: `act_group_aware` == `True` requires `act_order` == `False`."
            )

        # validate meta
        self.meta = self._ensure_meta_dict(self.meta)
        self.offload_to_disk_path = self._ensure_offload_path(
            self.offload_to_disk, self.offload_to_disk_path
        )

    def _resolve_activation_ordering(
        self,
        act_order_user_value: Optional[bool],
        act_group_aware_user_value: Optional[bool],
    ) -> None:
        """Normalize defaults and enforce compatibility between act_order and act_group_aware."""

        act_order_enabled_by_user = (
            bool(act_order_user_value) if act_order_user_value is not None else False
        )
        act_group_aware_enabled_by_user = (
            bool(act_group_aware_user_value)
            if act_group_aware_user_value is not None
            else False
        )

        if (
            act_order_enabled_by_user
            and act_group_aware_user_value is not None
            and act_group_aware_enabled_by_user
        ):
            raise ValueError(
                "QuantizeConfig:: `act_group_aware` == `True` requires `act_order` == `False` when both are explicitly set."
            )

        if (
            act_order_enabled_by_user
            and act_group_aware_user_value is None
            and self.act_group_aware
        ):
            log.warning(
                "QuantizeConfig: `act_order=True` automatically disables `act_group_aware`. "
                "Set `act_group_aware=False` explicitly to silence this warning."
            )
            self.act_group_aware = False

    @staticmethod
    def _normalize_pack_dtype(
        pack_dtype: Optional[Union[str, torch.dtype]],
    ) -> torch.dtype:
        """Ensure pack_dtype resolves to a supported torch dtype."""
        if pack_dtype is None:
            return torch.int32

        if isinstance(pack_dtype, torch.dtype):
            if pack_dtype not in VALID_PACK_DTYPE_NAMES.values():
                raise ValueError(
                    f"QuantizeConfig: Unsupported `pack_dtype`: {pack_dtype}"
                )
            return pack_dtype

        if isinstance(pack_dtype, str):
            dtype_key = pack_dtype.lower()
            if dtype_key not in VALID_PACK_DTYPE_NAMES:
                raise ValueError(
                    f"QuantizeConfig: Unsupported `pack_dtype`: {pack_dtype}"
                )
            return VALID_PACK_DTYPE_NAMES[dtype_key]

        raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {pack_dtype}")

    @staticmethod
    def _to_kernel(value: Union[str, KERNEL]) -> KERNEL:
        """Convert user-provided kernel values into the canonical enum."""
        if isinstance(value, KERNEL):
            return value
        try:
            return KERNEL(str(value).lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: Unknown quantization format: `{value}`."
            ) from exc

    @staticmethod
    def _to_method(value: Union[str, METHOD]) -> METHOD:
        """Convert user-provided method values into the canonical enum."""
        if isinstance(value, METHOD):
            return value
        try:
            return METHOD(str(value).lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: Unknown quantization method: `{value}`."
            ) from exc

    @staticmethod
    def _normalize_dynamic_overrides(
        overrides: Dict[str, Dict[str, Union[int, bool, float]]],
        bits_choices: List[int],
    ) -> Dict[str, Dict[str, Union[int, bool, float]]]:
        """Re-order and validate dynamic layer overrides."""
        ordered_keys = [key for key in overrides if key.startswith("-")]
        ordered_keys.extend(key for key in overrides if not key.startswith("-"))

        normalized: Dict[str, Dict[str, Union[int, bool, float]]] = {}
        for key in ordered_keys:
            layer_config = overrides[key]
            if not isinstance(layer_config, dict):
                raise ValueError(
                    f"QuantizeConfig: Dynamic override for `{key}` must be a dictionary."
                )
            validated_layer_config: Dict[str, Union[int, bool, float]] = {}
            for override_key, value in layer_config.items():
                if (
                    override_key == "bits"
                    and bits_choices
                    and value not in bits_choices
                ):
                    raise ValueError(
                        f"QuantizeConfig: Layer `{key}` only supports quantization of `{bits_choices}` bits."
                    )
                if override_key == "group_size" and value != -1 and value <= 0:
                    raise ValueError(
                        "QuantizeConfig: `group_size` must be one of `[-1, 16, 32, 64, 128, 256, 512, 1024]`."
                    )
                validated_layer_config[override_key] = value
            normalized[key] = validated_layer_config
        return normalized

    @staticmethod
    def _validate_group_size(group_size: int) -> None:
        """Ensure the group size is positive or disabled via -1."""
        if group_size != -1 and group_size <= 0:
            raise ValueError(
                "QuantizeConfig: `group_size` must be one of `[-1, 16, 32, 64, 128, 256, 512, 1024]`."
            )

    def _validate_damp_params(self) -> None:
        """Validate damping hyper-parameters."""
        if not (0 < self.damp_percent < 1):
            raise ValueError("QuantizeConfig: `damp_percent` must between 0 and 1.")
        if self.damp_auto_increment < 0:
            raise ValueError(
                "QuantizeConfig:: `damp_auto_increment` must greater than 0."
            )

    def _validate_hessian_config(self) -> None:
        """Validate Hessian staging hints used during GPTQ quantization."""
        if self.hessian_chunk_size is not None:
            if not isinstance(self.hessian_chunk_size, int):
                raise ValueError(
                    "QuantizeConfig: `hessian_chunk_size` must be an integer or None."
                )
            if self.hessian_chunk_size <= 0:
                raise ValueError(
                    "QuantizeConfig: `hessian_chunk_size` must be a positive integer."
                )

        if self.hessian_chunk_bytes is not None:
            if not isinstance(self.hessian_chunk_bytes, int):
                raise ValueError(
                    "QuantizeConfig: `hessian_chunk_bytes` must be an integer or None."
                )
            if self.hessian_chunk_bytes <= 0:
                raise ValueError(
                    "QuantizeConfig: `hessian_chunk_bytes` must be a positive integer amount of bytes."
                )

    @staticmethod
    def _ensure_meta_dict(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a metadata dictionary with string keys, creating one if needed."""
        if meta is None:
            return {}
        if not isinstance(meta, dict):
            raise ValueError("QuantizeConfig: `meta` must be a dictionary")
        for key in meta:
            if not isinstance(key, str):
                raise ValueError("QuantizeConfig: `meta` keys must be strings")
        return dict(meta)

    @staticmethod
    def _ensure_offload_path(
        offload_to_disk: bool, offload_path: Optional[str]
    ) -> Optional[str]:
        """Assign a deterministic offload path when offloading is enabled."""
        if not offload_to_disk:
            return offload_path
        if offload_path:
            return str(offload_path)
        path_key = f"{uuid.uuid4().hex}-{uuid.uuid4().hex}"
        resolved = f"./nanomodel_offload/{path_key}/"
        log.info("QuantizeConfig: offload_to_disk_path auto set to `%s`", resolved)
        return resolved

    def meta_set(self, key: str, value: Any):
        """Attach a metadata entry to the configuration."""
        self.meta[key] = value

    def meta_get(self, key: str) -> Any:
        """Fetch a metadata value if present."""
        return self.meta.get(key)

    def dynamic_get(
        self,
        layer_name: str,
        key: str = None,
        default: Union[int, bool, float] = None,
        sub_key: str = None,
    ) -> Union[Dict, int, bool, float]:
        return dynamic_get(self.dynamic, layer_name, key, default, sub_key)

    # versionable is a meta.property that pairs value with version i.e "value:1.0.0"
    def meta_set_versionable(self, key: str, value: List[str]):
        """Store a list of version-tagged metadata strings."""
        self.meta_set(key, value)

    # versionable is a meta.property that pairs value with version i.e "value:1.0.0"
    def meta_get_versionable(self, key: str) -> List[Tuple[str, str]]:
        """Return metadata pairs of (value, version) if available."""
        values = self.meta_get(key)
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]
        result = []
        for val in values:
            parts = val.split(":")
            if len(parts) >= 2:
                result.append((parts[0].lower(), parts[1].lower()))
        return result

    def save_pretrained(self, save_dir: str, **kwargs):
        """Persist the quantization configuration alongside a checkpoint directory."""
        save_path = Path(save_dir) / QUANT_CONFIG_FILENAME
        with open(save_path, "w", encoding="utf-8") as f:
            d = self.to_dict()
            json_str = json.dumps(d, indent=2)
            log.info(f"Saved Quantize Config: \n{json_str}")
            f.write(json_str)

    @classmethod
    # normalize quant config for compat and also performs validation
    def from_quant_config(cls, quantize_cfg, kernel: str = None):
        """Create a configuration object from a saved quantize_config dict."""
        valid_formats = {KERNEL.GPTQ, KERNEL.MARLIN}
        format_auto_inferred = False
        # compat: format can be passed in via from_quantized() if field missing from json
        kernel_enum = None
        if kernel:
            kernel_enum = cls._to_kernel(kernel)
            if kernel_enum not in valid_formats:
                raise ValueError(
                    f"QuantizeConfig: Unknown quantization checkpoint format: {kernel}."
                )
            if quantize_cfg.get(FORMAT_FIELD_CHECKPOINT):
                raise ValueError(
                    "QuantizeConfig: Conflicting quantization format passed in manually and also exists in model config."
                )
        # compat: warn if checkpoint_format is missing
        elif quantize_cfg.get(FORMAT_FIELD_CHECKPOINT) is None:
            format_auto_inferred = True

        field_names = [field.name for field in fields(cls)]

        # FIXME convert awg quantize_config to gptq quantize_config
        normalized = {
            QUANT_METHOD_FIELD: METHOD.GPTQ,
            FORMAT_FIELD_CODE: kernel_enum if kernel_enum else KERNEL.GPTQ,
        }
        for key, val in quantize_cfg.items():
            key_normalized = key.lower()

            # remap keys according to compat map
            if (
                key_normalized in QUANT_CONFIG_ARG_SYNONYMS
                and QUANT_CONFIG_ARG_SYNONYMS[key_normalized] in field_names
            ):
                key_normalized = QUANT_CONFIG_ARG_SYNONYMS[key_normalized]

            if key_normalized in {FORMAT_FIELD_CHECKPOINT, FORMAT_FIELD_CODE}:
                normalized[FORMAT_FIELD_CODE] = cls._to_kernel(val)
            elif key_normalized == QUANT_METHOD_FIELD:
                lowered_val = str(val).lower()
                if lowered_val == KERNEL.MARLIN.value:
                    normalized[FORMAT_FIELD_CODE] = KERNEL.MARLIN
                else:
                    normalized[QUANT_METHOD_FIELD] = cls._to_method(val)
            elif key_normalized in field_names:
                normalized[key_normalized] = val
            else:
                log.info(
                    "QuantizeConfig: Ignoring unknown parameter in the quantization configuration: %s.",
                    key,
                )

        if format_auto_inferred:
            log.info(
                "QuantizeConfig: `%s` is missing from the quantization configuration and is automatically inferred to %s",
                FORMAT_FIELD_CHECKPOINT,
                normalized[FORMAT_FIELD_CODE],
            )
        if "sym" not in normalized:
            log.warning(
                "QuantizeConfig: config does not contain `sym` (symmetric quantization). This may result in silent errors. Defaulting to `sym=True`."
            )

        return cls(**normalized)

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        """Load quantization parameters from a checkpoint directory."""
        kernel = kwargs.pop("kernel", None)

        transformers_config = False
        resolved_config_file = None
        for quantize_config_filename in QUANT_CONFIG_FILENAME_COMPAT:
            candidate_path = Path(save_dir) / quantize_config_filename
            if candidate_path.exists():
                resolved_config_file = candidate_path
                if quantize_config_filename == "config.json":
                    transformers_config = True
                break

        if resolved_config_file is None:
            raise ValueError(
                "QuantizeConfig: No quantize_config.json, quant_config.json or config.json file was found in the model repository."
            )

        with open(resolved_config_file, "r", encoding="utf-8") as f:
            args_from_json = json.load(f)

            if transformers_config:
                args_from_json = args_from_json["quantization_config"]

            return cls.from_quant_config(args_from_json, kernel)

    def to_dict(self):
        """Serialize the configuration to a JSON-friendly dictionary."""
        out = {
            "bits": self.bits,
            "dynamic": self.dynamic,
            "group_size": self.group_size,
            "act_order": self.act_order,
            "sym": self.sym,
            "lm_head": self.lm_head,
            QUANT_METHOD_FIELD: self.quant_method,
            FORMAT_FIELD_CHECKPOINT: self.kernel,
            # torch.dtype convert to string
            PACK_DTYPE_FIELD: str(self.pack_dtype).split(".")[-1],
            META_FIELD: self.meta,
        }

        if getattr(self, "pack_impl", "original") != "original":
            out["pack_impl"] = self.pack_impl

        # TODO FIXME: upstream gpt-qmodel config for awq recognition to transformers/sglang/vllm
        if self.quant_method == METHOD.AWQ:
            out["zero_point"] = self.zero_point
            # awq compat with vllm/sglang/transformers loaders
            out["version"] = self.kernel

        # simplify: clean keys where the value is None or empty [list, dict]
        out = {k: v for k, v in out.items() if v is not None and (v not in [None, {}])}

        dict_scale_dtype_to_str(out)
        return out

    # TODO FIX ME, g_idx int32 per infeature but infeature count is per module
    def calculate_bits_per_weight(self):
        """Log an approximate bits-per-weight figure for the current configuration."""
        if self.group_size != -1:
            # naive bits is
            # mlp.down_proj.g_idx: I32
            # mlp.down_proj.qweight: I32
            # mlp.down_proj.qzeros: I32
            # mlp.down_proj.scales: F16
            per_group_bits = (
                self.group_size * self.bits
            )  # qweight: packed by group_size
            per_group_bits += 16  # scales fp16: one per group
            per_group_bits += self.bits  # qzeros: one per group
            # FIX ME: g_idx is I32, one per infeature
            per_group_bits += (
                4  # ESTIMATE for g_idx int32: one per features/group_size item
            )
            bpw = per_group_bits / self.group_size

            # normally g_idx (int32 allocated one per in_feature) is allocated in device memory
            # but each module may have different infeatures we don't have enouch ctx here, use estimated `0.1` for now
            bpw += 0.1
        else:
            # there is only one scale int32 + one qzero int32 per entire module so overall it contributes to close to 0 bpw
            bpw = self.bits
        log.info(
            "Estimated Quantization BPW (bits per weight): %s bpw, based on [bits: %s, group_size: %s]",
            bpw,
            self.bits,
            self.group_size,
        )


# deprecated: will be removed in future update
@dataclass
class BaseQuantizeConfig(QuantizeConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log.warning(
            "QuantizeConfig: BaseQuantizeConfig is re-named and pending deprecation. Please use `QuantizeConfig` instead."
        )
