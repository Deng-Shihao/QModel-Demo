import json
import threading
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tqdm import tqdm
import torch
from random_word import RandomWords
from torch import Tensor
from torch.nn import Module

from .. import DEVICE_THREAD_POOL
from ..looper.input_cache import InputCache
from ..looper.named_module import NamedModule
from ..models import BaseNanoModel
from ..models.writer import (
    PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, PROCESS_LOG_NAME,
    PROCESS_LOG_TIME, PROCESS_USED_MEMORY, QUANT_LOG_DAMP, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES
)
from ..quantization.config import QuantizeConfig


# ---------------------------
# Setup standard logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

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


class LoopProcessor:
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
        self.lock = threading.Lock()
        self._results: Dict[str, Any] = {}
        self._results_lock = threading.Lock()

        self.tokenizer = tokenizer
        self.qcfg = qcfg
        self.require_fwd = require_fwd
        self.fwd_after_process = fwd_after_process
        self.fwd_all_modules_in_single_pass = fwd_all_modules_in_single_pass

        self.inputs_cache: InputCache = InputCache(None, None, None, None)
        self.tasks = {}

        self.fwd_time = None
        self.layer_count = None

        self.gpu_memorys = []
        self.cpu_memorys = []
        self.durations = []
        self.module_names = []

        self.log = []
        self.log_call_count = 0

        current_time = datetime.now().strftime("%m_%d_%Y_%Hh_%Mm_%Ss")
        self.log_tmp_log_file_name = f"{self.name()}_log_{RandomWords().get_random_word()}_time_{current_time}.log"

        # ---- 移除 Device 相关逻辑 ----
        # self._device_smi_handles = self._init_device_smi_handles()
        # self._cpu_device_smi = self._init_cpu_device_handle()
        # self._device_metric_failures: Set[str] = set()

        # prepare dataset
        if calibration is not None:
            if len(calibration) == 0:
                raise ValueError("Calibration dataset must not be empty.")

            if prepare_dataset_func is None:
                raise ValueError("prepare_dataset_func must be provided when calibration data is supplied.")

            calibration = prepare_dataset_func(
                calibration_dataset=calibration,
                calibration_dataset_concat_size=calibration_concat_size,
                calibration_dataset_sort=calibration_sort,
                batch_size=batch_size
            )

            log.info(f"Preparing calibration dataset with {len(calibration)} samples...")

            total_input_ids_length = 0
            max_input_id_length = 0
            for row in tqdm(calibration, desc="Checking calibration inputs"):
                input_ids = row["input_ids"]
                if isinstance(input_ids, torch.Tensor):
                    if input_ids.dim() <= 2:
                        input_ids_length = input_ids.shape[-1]
                    else:
                        raise ValueError(
                            f"Expected 1D/2D tensor for 'input_ids', but got {input_ids.dim()}D tensor."
                        )
                else:
                    input_ids_length = len(input_ids)

                max_input_id_length = max(max_input_id_length, input_ids_length)
                total_input_ids_length += input_ids_length

            avg = total_input_ids_length / len(calibration)
            log.info(f"Average input length: {avg:.1f}, Max: {max_input_id_length}")

            self.num_batches = len(calibration)
            self.calibration_dataset = calibration
        else:
            self.num_batches = 0
            self.calibration_dataset = []

        self._batch_tls = threading.local()

    # -------------------------------
    # 以下方法基本保留
    # -------------------------------

    def log_save_async(self, stat):
        DEVICE_THREAD_POOL.submit_serial("cpu", self._async_log_writer, stat)

    def _async_log_writer(self, stat):
        with open(self.log_tmp_log_file_name, 'a') as f:
            json.dump(stat, f, indent=4)
            f.write("\n")

    def collect_memory_info(self, layer_index: int):
        """Removed device_smi metrics. Only report torch CUDA memory."""
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            self.gpu_memorys.append(mem_alloc)
            log.info(f"[Layer {layer_index}] GPU Memory: {mem_alloc:.2f} GB (reserved {mem_reserved:.2f} GB)")

    def finalize(self, model: BaseNanoModel, **kwargs):
        del self.inputs_cache
        del self._results
        log.info("Finalized LoopProcessor and released resources.")

    def name(self) -> str:
        return "LoopProcessor"


def get_max_memory() -> str:
    """Simplified: Remove DEVICE_0 / DEVICE_1 dependency."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    current_device = torch.cuda.current_device()
    stats = torch.cuda.memory_stats(current_device)
    active = stats.get("active_bytes.all.current", 0) / 1024 ** 2
    peak = stats.get("active_bytes.all.peak", 0) / 1024 ** 2
    return f"Device {current_device}: {active:.2f}MB (peak {peak:.2f}MB)"
