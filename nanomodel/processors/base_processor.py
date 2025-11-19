from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from .. import DEVICE_THREAD_POOL
from ..processors.input_cache import InputCache
from ..processors.named_module import NamedModule
from ..models import BaseNanoModel
from ..models.writer import (
    PROCESS_LOG_FWD_TIME,
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_NAME,
    PROCESS_LOG_TIME,
    PROCESS_USED_MEMORY,
    QUANT_LOG_DAMP,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
)
from ..quantization.config import QuantizeConfig
from ..utils.monitor import Device
from ..utils.logger import setup_logger
from ..utils.torch import CPU, DEVICE_0, DEVICE_1

log = setup_logger()

# global level lock
PROCESSOR_GLOBAL_LOCK = threading.Lock()

MODULE_FEATURE_COLUMN = "feat: in, out"
DTYPE_SIZE_COLUMN = "dtype: size"

DEFAULT_LOG_COLUMNS: List[str] = [
    PROCESS_LOG_NAME,
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    MODULE_FEATURE_COLUMN,
    DTYPE_SIZE_COLUMN,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
    QUANT_LOG_DAMP,
    PROCESS_LOG_TIME,
    PROCESS_LOG_FWD_TIME,
    PROCESS_USED_MEMORY,
    "dynamic",
]


# LoopProcessor is a singleton(), not per module instance
class BaseProcessor:
    """Shared lifecycle hooks for quantization processors."""

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration,
        prepare_dataset_func: Optional[Callable] = None,
        calibration_concat_size: Optional[int] = None,
        calibration_sort: Optional[str] = None,
        batch_size: int = 1,
        require_fwd: bool = True,
        fwd_after_process: bool = True,
        fwd_all_modules_in_single_pass: bool = False,
    ):
        # Serialise per-processor state transitions.
        self.lock = threading.Lock()

        # Result is total collection of all module results mapped by module.full_name.
        self._results: Dict[str, Any] = {}
        self._results_lock = threading.Lock()

        self.tokenizer = tokenizer
        self.qcfg = qcfg
        self.qcfg_dynamic = None  # cloned and dynamic filtered

        self.require_fwd = require_fwd  # default True
        self.fwd_after_process = fwd_after_process  # default True
        self.fwd_all_modules_in_single_pass = (
            fwd_all_modules_in_single_pass  # default False
        )

        # Capture layer inputs and metadata shared across processors.
        self.inputs_cache: InputCache = InputCache([], [], [], [])
        self.tasks: Dict[str, Any] = {}

        self.pb = None
        self.fwd_time: Optional[float] = None
        self.layer_count: Optional[int] = None

        self.gpu_memorys: List[float] = []
        self.cpu_memorys: List[float] = []
        self.durations: List[float] = []
        self.module_names: List[str] = []

        # Logging buffers populated by subclasses during processing.
        self.log: List[Dict[str, Any]] = []
        self.log_call_count = 0
        self._log_column_labels: List[str] = []
        self._log_columns = None
        self._log_header_interval = 20
        current_time = datetime.now().strftime("%m_%d_%Y_%Hh_%Mm_%Ss")
        self.log_tmp_log_file_name = (
            f"{self.name()}_log_{uuid.uuid4().hex}_time_{current_time}.log"
        )
        self._device_smi_handles = self._init_device_smi_handles()
        self._cpu_device_smi = self._init_cpu_device_handle()
        self._device_metric_failures: Set[str] = set()

        (
            self.calibration_dataset,
            self.num_batches,
        ) = self._prepare_calibration_dataset(
            calibration=calibration,
            prepare_dataset_func=prepare_dataset_func,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            batch_size=batch_size,
        )

        # Track the current calibration batch index on a per-thread basis so
        # processors can retrieve deterministic ordering information (e.g.
        # GPTQ's Hessian updates) even when forwards run on multiple threads.
        self._batch_tls = threading.local()

    def _prepare_calibration_dataset(
        self,
        calibration: Optional[Sequence[Dict[str, Any]]],
        prepare_dataset_func: Optional[Callable[..., Iterable[Dict[str, Any]]]],
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Validate, preprocess, and materialise the calibration dataset."""
        if calibration is None:
            return [], 0

        if len(calibration) == 0:
            raise ValueError("Calibration dataset must not be empty.")

        min_dataset_size = 256
        if len(calibration) < min_dataset_size:
            log.warn(
                f"Calibration dataset size should be more than {min_dataset_size}. "
                f"Current: {len(calibration)}."
            )

        if prepare_dataset_func is None:
            raise ValueError(
                "prepare_dataset_func must be provided when calibration data is supplied."
            )

        processed = prepare_dataset_func(
            calibration_dataset=calibration,
            calibration_dataset_concat_size=calibration_concat_size,
            calibration_dataset_sort=calibration_sort,
            batch_size=batch_size,
        )

        dataset = list(processed)
        if not dataset:
            raise ValueError("Processed calibration dataset is empty.")

        self._warn_on_short_sequences(dataset)
        return dataset, len(dataset)

    def _warn_on_short_sequences(self, dataset: Sequence[Dict[str, Any]]) -> None:
        """Emit guidance when calibration samples appear too small."""
        min_avg_length = 256
        avg_length = self._average_input_length(dataset)

        if avg_length < min_avg_length:
            log.warn(
                "The average length of input_ids of calibration_dataset should be greater than "
                f"{min_avg_length}: actual avg: {avg_length}."
            )

    def _average_input_length(self, dataset: Sequence[Dict[str, Any]]) -> float:
        """Compute the average token length for the provided calibration rows."""
        total = 0
        for row in dataset:
            total += self._normalize_input_length(row["input_ids"])
        return total / len(dataset)

    def _normalize_input_length(self, input_ids: Any) -> int:
        """Return the trailing dimension of supported tensor/list inputs."""
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() <= 2:
                return int(input_ids.shape[-1])
            raise ValueError(
                "Expected a 1-dimensional tensor or 2-dimensional tensor for 'input_ids', "
                f"but got a tensor with {input_ids.dim()} dimensions."
            )
        return len(input_ids)

    def _set_current_batch_index(self, batch_index: Optional[int]) -> None:
        if batch_index is None:
            if hasattr(self._batch_tls, "index"):
                delattr(self._batch_tls, "index")
        else:
            self._batch_tls.index = int(batch_index)

    def current_batch_index(self) -> Optional[int]:
        return getattr(self._batch_tls, "index", None)

    def _async_log_writer(self, stat: Dict[str, Any]) -> None:
        """Persist structured log rows asynchronously."""
        with open(self.log_tmp_log_file_name, "a") as f:
            json.dump(stat, f, indent=4)
            f.write("\n")

    def log_save_async(self, stat: Dict[str, Any]) -> None:
        # Serialize writes on the CPU-bound worker to avoid interleaved JSON output.
        DEVICE_THREAD_POOL.submit_serial(CPU, self._async_log_writer, stat)

    def log_new_row(self, stat: Dict[str, Any]) -> None:
        """Stream log rows to the structured table and async sink."""
        with self.lock:
            self.log_call_count += 1
            columns_rebuilt = self._ensure_log_columns(stat)

            if self._log_columns is None:
                return

            if columns_rebuilt or self.log_call_count % self._log_header_interval == 1:
                self._log_columns.info.header()

            row_values = [
                self._format_log_value(column, stat.get(column, ""))
                for column in self._log_column_labels
            ]
            self._log_columns.info(*row_values)

        self.log_save_async(stat)

    def loss_color(self, loss_value: float) -> str:
        """Colour-code loss values for quick terminal scanning."""
        if loss_value <= 0.1:
            return "\033[92m"  # Green
        elif loss_value <= 1:
            return "\033[96m"  # Cyan
        elif loss_value <= 5:
            return "\033[93m"  # Yellow
        elif loss_value <= 20:
            return "\033[33m"  # Orange
        else:
            return "\033[91m"  # Red

    def _ensure_log_columns(self, stat: Dict[str, Any]) -> bool:
        """Track new keys so the live log header stays in sync."""
        desired_labels = list(DEFAULT_LOG_COLUMNS)
        for key in stat.keys():
            if key not in desired_labels:
                desired_labels.append(key)

        if self._log_columns is not None and desired_labels == self._log_column_labels:
            return False

        self._log_column_labels = desired_labels
        return True

    def _format_log_value(self, key: str, value: Any) -> str:
        """Apply per-column formatting before emitting a log row."""
        text = "" if value is None else str(value)

        if key == QUANT_LOG_LOSS and text:
            try:
                color_code = self.loss_color(float(text))
            except (TypeError, ValueError):
                return text
            reset = "\033[0m"
            return f"{color_code}{text}{reset}"

        return text

    def module_feature_summary(self, module: NamedModule) -> str:
        """Return a compact view of common module feature dimensions."""
        in_features = module.state.get("in_features")
        out_features = module.state.get("out_features")

        if isinstance(in_features, int) and isinstance(out_features, int):
            return f"{in_features}, {out_features}"
        return ""

    def module_dtype_size_summary(self, module: NamedModule) -> str:
        """Summarise parameter dtype and total storage footprint for logging."""
        weight = getattr(module.module, "weight", None)
        dtype = getattr(weight, "dtype", None)
        total_bytes = 0

        if isinstance(weight, torch.Tensor):
            total_bytes += weight.numel() * weight.element_size()
        else:
            dtype = dtype or getattr(module, "module_dtype", None)
            in_features = module.state.get("in_features")
            out_features = module.state.get("out_features")
            if (
                dtype is not None
                and isinstance(in_features, int)
                and isinstance(out_features, int)
            ):
                element_size = torch.empty((), dtype=dtype).element_size()
                total_bytes += in_features * out_features * element_size

        bias = getattr(module.module, "bias", None)
        if isinstance(bias, torch.Tensor):
            total_bytes += bias.numel() * bias.element_size()

        # account for persistent tensors captured in module.state (e.g., q_scales, adapters)
        total_bytes += self._state_tensor_bytes(module)

        dtype = dtype or getattr(module, "module_dtype", None)
        dtype_label = self._format_dtype(dtype)
        size_mb = total_bytes / (1024 * 1024)
        return f"{dtype_label}: {size_mb:.1f}MB"

    def _state_tensor_bytes(self, module: NamedModule) -> int:
        """Estimate bytes held by tensors cached on the NamedModule."""
        seen: Set[int] = set()
        total = 0
        for key, value in module.state.items():
            if key in {"in_features", "out_features"}:
                continue
            total += self._collect_tensor_bytes(value, seen)
        return total

    def _collect_tensor_bytes(self, obj: Any, seen: Set[int]) -> int:
        if isinstance(obj, torch.Tensor):
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)
            return obj.numel() * obj.element_size()

        if isinstance(obj, (list, tuple, set)):
            return sum(self._collect_tensor_bytes(item, seen) for item in obj)

        if isinstance(obj, dict):
            return sum(self._collect_tensor_bytes(item, seen) for item in obj.values())

        # handle known adapter containers without traversing entire nn.Module graphs
        if hasattr(obj, "lora_A") and hasattr(obj, "lora_B"):
            return self._collect_tensor_bytes(
                obj.lora_A, seen
            ) + self._collect_tensor_bytes(obj.lora_B, seen)

        return 0

    def _format_dtype(self, dtype: Optional[torch.dtype]) -> str:
        if dtype is None:
            return "n/a"

        dtype_str = str(dtype)
        if dtype_str.startswith("torch."):
            dtype_str = dtype_str.split(".", 1)[1]

        dtype_alias = {
            "bfloat16": "bf16",
            "float16": "f16",
            "float32": "f32",
        }

        return dtype_alias.get(dtype_str, dtype_str)

    def _init_device_smi_handles(self) -> Dict[str, Device]:
        """Create lightweight handles for device memory queries."""
        handles: Dict[str, Device] = {}

        for device_id in self._discover_accelerator_devices():
            try:
                handles[device_id] = Device(device_id)
            except Exception as exc:  # pragma: no cover - defensive, external tool
                log.debug(f"Device-SMI initialisation failed for `{device_id}`: {exc}")

        return handles

    def _init_cpu_device_handle(self) -> Optional[Device]:
        """Create a CPU handle if the monitoring backend supports it."""
        try:
            return Device("cpu")
        except Exception as exc:  # pragma: no cover - defensive, external tool
            log.debug(f"Device-SMI CPU initialisation failed: {exc}")
            return None

    def _discover_accelerator_devices(self) -> List[str]:
        """Return device identifiers for available accelerator backends."""
        devices: List[str] = []

        if hasattr(torch, "cuda"):
            try:
                if torch.cuda.is_available():
                    device_type = (
                        "rocm" if getattr(torch.version, "hip", None) else "cuda"
                    )
                    for idx in range(torch.cuda.device_count()):
                        devices.append(f"{device_type}:{idx}")
            except Exception:  # pragma: no cover - defensive, CUDA runtime differences
                pass

        xpu = getattr(torch, "xpu", None)
        if xpu is not None:
            try:
                if torch.xpu.is_available():
                    for idx in range(torch.xpu.device_count()):
                        devices.append(f"xpu:{idx}")
            except Exception:  # pragma: no cover - defensive, XPU runtime differences
                pass

        return devices

    def _safe_query_metric(self, device_key: str, handle: Device):
        """Best-effort query on SMI handles without surfacing noisy warnings."""
        try:
            return handle.metrics(fast=True)
        except Exception as exc:  # pragma: no cover - defensive, external tool
            if device_key not in self._device_metric_failures:
                log.debug(f"Device-SMI metrics failed for `{device_key}`: {exc}")
                self._device_metric_failures.add(device_key)
            return None

    def _snapshot_device_memory_gib(self) -> Dict[str, float]:
        snapshot: Dict[str, float] = {}
        for device_id, handle in self._device_smi_handles.items():
            metrics = self._safe_query_metric(device_id, handle)
            if metrics is None:
                continue
            snapshot[device_id] = metrics.memory_used / (1024**3)
        return snapshot

    def _snapshot_cpu_memory_gib(self) -> Optional[float]:
        if self._cpu_device_smi is None:
            return None
        metrics = self._safe_query_metric("cpu", self._cpu_device_smi)
        if metrics is None:
            return None
        return metrics.memory_used / (1024**3)

    def device_memory_report(self) -> str:
        snapshot = self._snapshot_device_memory_gib()
        if not snapshot:
            return "n/a"
        parts = [f"{device_id}={value:.1f}GB" for device_id, value in snapshot.items()]
        return ", ".join(parts)

    def _close_device_smi_handles(self) -> None:
        """Gracefully release device monitoring handles."""
        for handle in self._device_smi_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._device_smi_handles.clear()

        if self._cpu_device_smi is not None:
            try:
                self._cpu_device_smi.close()
            except Exception:
                pass
            self._cpu_device_smi = None

    def attach_log_columns(self, columns: Any) -> None:
        """Expose a setter for structured logging helpers."""
        self._log_columns = columns

    # Loop Processor level scoped state data
    def result_save(self, key: str, value: Any) -> None:
        """Persist per-module results for later processors."""
        with self._results_lock:
            # assert self.result_get(key) is None, f"key: {key} already exists in `self.result`"
            self._results[key] = value

    def result_get(self, key: str, default: Any = None) -> Any:
        """Retrieve per-module results stored during processing."""
        with self._results_lock:
            return self._results.get(key, default)

    def result_pop(self, key: str, default: Any = None) -> Any:
        """Remove and return processor-scoped results."""
        with self._results_lock:
            return self._results.pop(key, default)

    def results(self) -> Dict[str, Any]:
        """Expose the raw result cache for read-only access."""
        return self._results

    def collect_memory_info(self, layer_index: int) -> None:
        """Snapshot device and CPU memory usage for the current layer."""
        _ = layer_index  # reserved for subclasses that need the index
        device_snapshot = self._snapshot_device_memory_gib()
        if device_snapshot:
            total_gpu_memory = sum(device_snapshot.values())
            self.gpu_memorys.append(total_gpu_memory)

        cpu_memory = self._snapshot_cpu_memory_gib()
        if cpu_memory is not None:
            self.cpu_memorys.append(cpu_memory)

    def log_plotly(self) -> None:
        pass

    def set_calibration_dataset(
        self, calibration_dataset: Optional[Iterable[Dict[str, Any]]]
    ) -> None:
        """Allow processors to reuse preprocessed calibration samples."""
        if calibration_dataset is None:
            self.calibration_dataset = []
            self.num_batches = 0
            return

        dataset_list = list(calibration_dataset)
        self.calibration_dataset = dataset_list
        self.num_batches = len(dataset_list)

    def set_fwd_time(self, fwd_time: float) -> None:
        """Record the measured forward pass duration for reporting."""
        self.fwd_time = fwd_time

    def formatted_fwd_time(self) -> str:
        fwd_time = self.fwd_time if self.fwd_time is not None else 0.0
        return f"{fwd_time:.3f}"

    # called first
    def preprocess(self, module: NamedModule, **kwargs):
        """Optional hook executed before forward activation capture."""
        pass

    # after preproces, this process may be skipped due to dynamic override (lora adapter = None)
    def is_skipped(self, module: NamedModule) -> bool:
        """Return True when the module should be skipped for this processor."""
        pass

    def receive_input_cache(self, input_cache: InputCache) -> None:
        """Inject the cached inputs produced by ModuleProcessor."""
        self.inputs_cache = input_cache

    # called after every module generate
    # may be called multiple times due to batch
    def receive_layer_inputs(self, layer_inputs: List[List[Tensor]]) -> None:
        """Store the per-layer hidden states captured during forwards."""
        self.inputs_cache.layer_inputs = layer_inputs

    def clear_cache_data(self) -> None:
        """Reset transient caches between processor passes."""
        self.tasks = {}
        self.inputs_cache.layer_inputs = []

    def pre_process_fwd_hook(
        self, name: str
    ) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Return a forward pre-hook used during activation capture."""
        pass

    # do work and return processor.self state which will updated/merged
    def process(self, module: NamedModule, device: torch.device = None):
        """Primary per-module processing entrypoint."""
        pass

    # last step, after all loop processor is called
    # submodule_finalize is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule, model: BaseNanoModel, **kwargs):
        """Clean up any per-module state once all processors complete."""
        pass
        # self.offload_to_disk(module=module)

    # last step, after all loop processor is called
    # finalize is called in reverse after all next sequential processes are called
    def finalize(self, model: BaseNanoModel, **kwargs):
        """Release resources captured for the lifetime of the processor."""
        self._close_device_smi_handles()
        del self.inputs_cache
        del self._results

        # TODO make this file delete based on user toggle
        # cleanup temp log file
        # if os.path.exists(self.log_tmp_log_file_name):
        #     os.remove(file_path)

    def release_calibration_dataset(self) -> None:
        """Drop the cached calibration dataset to free memory."""
        del self.calibration_dataset
        self.num_batches = 0

    def number_batches(self) -> int:
        """Expose the number of preprocessed calibration batches."""
        return self.num_batches

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Return True if this processor should consume the shared dataset."""
        dataset = getattr(self, "calibration_dataset", None)
        return bool(dataset)

    def name(self) -> str:
        """Name used for logging and user-facing summaries."""
        return self.__class__.__name__.replace("Processor", "").lower()


def get_max_memory() -> str:
    """Return a succinct GPU memory snapshot for the first two devices."""
    stats_0 = torch.cuda.memory_stats(DEVICE_0)
    active_0 = stats_0.get("active_bytes.all.current", 0) / 1024**2

    if torch.cuda.device_count() > 1:
        stats_1 = torch.cuda.memory_stats(DEVICE_1)
        active_1 = stats_1.get("active_bytes.all.current", 0) / 1024**2
        return f"{active_0:.2f}MB, {active_1:.2f}MB"

    return f"{active_0:.2f}MB"
