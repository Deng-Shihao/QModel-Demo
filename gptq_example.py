"""End-to-end GPTQ quantization flow for a small Qwen model."""
import os
import logging
from transformers import AutoTokenizer
from nanomodel import AutoNanoModel, QuantizeConfig, get_best_device
from nanomodel.quantization import KERNEL, METHOD

# Ensure deterministic device ordering and a more memory-efficient allocator.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

pretrained_model_id = "Qwen/Qwen3-1.7B"
quantized_model_id = "./quantized_models/qwen3-1.7b-gptq-4bit"


def main():
    """Quantize a pretrained model with GPTQ and run a single generation."""
    logger = logging.getLogger("NanoModel")
    logger.info("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)

    # Capture a tiny calibration dataset as tokenized inputs; real workflows should use more data.
    calibration_dataset = [
        tokenizer(
            "gptq is a method that compresses large language models by converting"
            " their weights to lower precision (like 4-bit) after training, making"
            " them smaller and faster with minimal accuracy loss."
        )
    ]

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=128,
        quant_method=METHOD.GPTQ,  # switch to METHOD.AWQ or METHOD.GPTQ as needed
        kernel=KERNEL.GPTQ,
    )

    logger.info("Loading pretrained model for quantization...")
    model = AutoNanoModel.load(pretrained_model_id, quantize_config)

    logger.info("Quantizing model...")
    model.quantize(calibration_dataset)

    logger.info(f"Saving quantized model to: {quantized_model_id}")
    model.save(quantized_model_id)

    model = AutoNanoModel.load(quantized_model_id, device=get_best_device())

    # Run a quick generation to validate the quantized weights.
    prompt_inputs = tokenizer("LLMs is", return_tensors="pt").to(model.device)
    generated = model.generate(**prompt_inputs)[0]
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

    # quantize_config = QuantizeConfig(
        # bits=4,  # default 4-bit [2, 3, 4, 8]
        # group_size=128,  # default 128

        # act_order: Optional[bool] = field(default=None)
        # act_group_aware: Optional[bool] = field(default=None)
        # static_groups: bool = field(default=False)
        # sym: bool = field(default=True)
        # true_sequential: bool = field(default=True)

        # lm_head = False (default)
        # quant_method = GPTQ (default)
        # mse: float = field(default=0.0)
        # mock_quantization: bool = field(default=False, metadata={"help": "Skip heavy computations for fast model loading validation"})

        # hessian_chunk_size (default=None)
        # hessian_chunk_bytes (default=None)
        # hessian_use_bfloat16_staging (default=False)

    # )
