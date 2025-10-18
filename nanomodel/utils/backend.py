from enum import Enum


class BACKEND(str, Enum):
    AUTO = "auto"  # choose the optimal local kernel based on quant_config compatibility
    AUTO_TRAINABLE = "auto_trainable" # choose the optimal trainable local kernel for post-quant training

    # gptq
    TORCH_FUSED = "torch_fused" # optimized for Intel XPU
    TORCH = "torch" # GOOD: about 80% of triton
    TRITON = "triton" # VERY GOOD: all-around kernel
    MARLIN = "marlin" # FASTEST: marlin reduce ops in fp32 (higher precision -> more accurate, slightly slower)
    MARLIN_FP16 = "marlin_fp16" # FASTEST and then some: marlin reduce ops in fp16 (lower precision -> less accurate, slightly faster)

    # awq
    GEMM = "gemm"
    GEMV = "gemv"
    GEMV_FAST = "gemv_fast"

    # external
    VLLM = "vllm" # External inference engine: CUDA + ROCm + IPEX
    SGLANG = "sglang" # External inference engine: CUDA + ROCm
    MLX = "mlx" # External inference engine: Apple MLX on M1+ (Apple Silicon)
