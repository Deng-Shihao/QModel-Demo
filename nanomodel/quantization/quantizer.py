# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

from typing import Optional

import torch
import torch.nn as nn

from ..quantization import QuantizeConfig
from ..utils.logger import setup_logger

log = setup_logger()

HF_OPTIMUM = "hf_optimum"


def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: torch.Tensor,
    requires_groupwise_processing: bool,
) -> torch.Tensor:
    """Quantize tensor values given precomputed scale and zero-point parameters."""
    if maxq < 0:
        high = (x > scale / 2).float() * scale
        low = (x < zero / 2).float() * zero
        return high + low

    if requires_groupwise_processing:
        q = torch.clamp(torch.round(x / scale), -maxq, maxq)
        return scale * q

    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    """Base quantizer that computes scale/zero parameters and applies quantization."""

    def __init__(
        self, qcfg: QuantizeConfig, shape: int = 1, name: Optional[str] = None
    ):
        super().__init__()

        self.qcfg = qcfg
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

        self.name = name

    def requires_groupwise_processing(self) -> bool:
        """Override in subclasses that rely on groupwise scales."""
        return False

    # FIXME, optimum shouldn't call this directly, it should call hf_configure
    def configure(
        self,
        perchannel=False,
        grid=100,
        maxshrink=0.8,
        trits=False,
        bits: int = 4,  # for hf compat
        sym: bool = False,  # for hf compat
    ):
        """Configure quantizer metadata coming from calibration/export flows."""
        if self.name == HF_OPTIMUM:
            self.qcfg.bits = bits
            self.qcfg.sym = sym

        groupwise = self.requires_groupwise_processing()
        device, dtype = self.scale.device, self.scale.dtype
        max_range = (
            2 ** (self.qcfg.bits - 1) - 1 if groupwise else 2**self.qcfg.bits - 1
        )
        self.maxq = torch.tensor(max_range, device=device, dtype=dtype)

        self.perchannel = perchannel
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1, device=device, dtype=dtype)

    def find_params(self, x: torch.Tensor, weight: bool = False) -> None:
        """Compute optimal scale and zero-point values for the provided tensor."""
        dev = x.device
        dtype = x.dtype
        groupwise = self.requires_groupwise_processing()
        self.maxq = self.maxq.to(device=dev, dtype=dtype)

        shape = x.shape
        original_shape = shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        zero_reference = torch.zeros(x.shape[0], device=dev, dtype=dtype)
        xmin = torch.minimum(x.min(1)[0], zero_reference)
        xmax = torch.maximum(x.max(1)[0], zero_reference)

        if self.qcfg.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            negative_mask = xmin < 0
            if torch.any(negative_mask):
                xmin[negative_mask] = -xmax[negative_mask]
        zero_range = (xmin == 0) & (xmax == 0)
        xmin[zero_range] = -1
        xmax[zero_range] = 1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            if groupwise:
                self.scale = xmax / self.maxq
                self.zero = torch.zeros_like(self.scale)
            else:
                self.scale = (xmax - xmin) / self.maxq
                if self.qcfg.sym:
                    self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
                else:
                    self.zero = torch.round(-xmin / self.scale)

        if self.qcfg.mse > 0.0:
            best = torch.full([x.shape[0]], float("inf"), device=dev, dtype=dtype)
            max_steps = int(self.maxshrink * self.grid)
            for i in range(max_steps):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = xmax1 / self.maxq if groupwise else (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.qcfg.sym else self.zero
                q = quantize(
                    x,
                    scale1.unsqueeze(1),
                    zero1.unsqueeze(1),
                    self.maxq,
                    groupwise,
                )
                q -= x
                q.abs_()
                q.pow_(self.qcfg.mse)
                err = torch.sum(q, 1)
                improved = err < best
                if torch.any(improved):
                    best[improved] = err[improved]
                    self.scale[improved] = scale1[improved]
                    self.zero[improved] = zero1[improved]
        if not self.perchannel:
            if weight:
                repeat_size = original_shape[0]
            else:
                repeat_size = (
                    original_shape[1] if len(original_shape) != 3 else original_shape[2]
                )
            self.scale = self.scale.repeat(repeat_size)
            self.zero = self.zero.repeat(repeat_size)

        if weight:
            reshape = [-1] + [1] * (len(original_shape) - 1)
            self.scale = self.scale.reshape(reshape)
            self.zero = self.zero.reshape(reshape)
            return
        if len(original_shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(original_shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(original_shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization to the provided tensor using cached parameters."""
        return quantize(
            x,
            self.scale,
            self.zero,
            self.maxq,
            self.requires_groupwise_processing(),
        )

    # def enabled(self):
    #     return self.maxq > 0

    # def ready(self):
    # return torch.all(self.scale != 0)


__all__ = ["Quantizer"]
