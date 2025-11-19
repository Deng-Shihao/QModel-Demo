"""A processor that calculates and annotates the required padding for tensor-parallel execution."""

from __future__ import annotations

import math
from typing import Dict

import torch

from ..quantization.gptq import get_number_of_rows_and_cols
from ..utils.logger import setup_logger
from .base_processor import BaseProcessor
from .named_module import NamedModule

log = setup_logger()


class TensorParallelWeightProcessor(BaseProcessor):
    """Calculates padding for tensor-parallelism and annotates modules.

    Quantization backends that shard weights across tensor-parallel ranks (e.g., 2, 4, or 8)
    require that weight dimensions be divisible by the number of ranks. This processor
    pre-computes the necessary padding and stores it in the module's state. Downstream
    processes can then use this metadata to create padded weights without modifying
    the original model parameters.
    """

    # The tensor-parallel ranks we need to support.
    _TP_TARGETS = (2, 4, 8)

    def __init__(self, *args, **kwargs):
        # This processor does not perform forward passes or calculate weight differences.
        kwargs.pop("calculate_w_wq_diff", None)
        kwargs.setdefault("require_fwd", False)
        kwargs.setdefault("fwd_after_process", False)
        super().__init__(*args, **kwargs)

        # The target multiple is the Least Common Multiple (LCM) of the TP ranks.
        # This ensures that the padded dimension is divisible by 2, 4, and 8.
        self._target_multiple = math.lcm(*self._TP_TARGETS)

    def preprocess(self, module: NamedModule):
        """A hook that runs before processing any modules. No setup is needed here."""
        pass

    def is_skipped(self, module: NamedModule) -> bool:
        """Determines if this processor should skip a given module. It is always active."""
        return False

    def pre_process_fwd_hook(self, name: str):
        """Returns a forward hook. This processor doesn't need to inspect activations."""

        def _noop(module, inputs, output):
            return None

        return _noop

    def process(self, module: NamedModule):
        """Computes and stores the tensor-parallel padding info for a module."""
        target = module.module
        if not hasattr(target, "weight"):
            return

        # Calculate the required padding for the weight's column dimension.
        pad_info = self._compute_padding(module)

        # If no padding is needed, ensure no old state remains.
        if pad_info["pad_cols"] == 0:
            module.state.pop("tp_pad_info", None)
            return

        # Annotate the module with the padding information for later stages.
        module.state["tp_pad_info"] = pad_info

        log.debug(
            "Annotated module %s for TP padding: original_cols=%d, pad_cols=%d",
            module.full_name,
            pad_info["original_columns"],
            pad_info["pad_cols"],
        )

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Checks for a calibration dataset. Not needed for this processor."""
        return True

    def name(self) -> str:
        """Returns the name of the processor."""
        return "tp-pre-pad"

    def _compute_padding(self, named_module: NamedModule) -> Dict[str, int]:
        """Calculates the number of columns to pad for tensor-parallel compatibility."""
        _, columns = get_number_of_rows_and_cols(named_module)

        # Calculate how many columns we need to add to make the total divisible by the target multiple.
        # Example: columns=100, target=8. 100 % 8 = 4. (8 - 4) % 8 = 4. We need to pad 4 columns.
        # Example: columns=128, target=8. 128 % 8 = 0. (8 - 0) % 8 = 0. No padding needed.
        pad_cols = (
            self._target_multiple - (columns % self._target_multiple)
        ) % self._target_multiple

        return {
            "pad_cols": pad_cols,
            "target_multiple": self._target_multiple,
            "original_columns": columns,
        }


__all__ = ["TensorParallelWeightProcessor"]
