"""End-to-end GPTQ quantization flow for a small Qwen model."""

import os
import logging

from datasets import load_dataset
from transformers import AutoTokenizer
from nanomodel import AutoNanoModel, QuantizeConfig, get_best_device
from nanomodel.quantization import KERNEL, METHOD

# Ensure deterministic device ordering and a more memory-efficient allocator.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

model_id_path = "Qwen/Qwen3-4B-Instruct-2507"
quant_model_path = "/home/sd24191/git_project/QModel-Demo/quantized_models/Qwen3-4B-Instruct-2507-GPTQ-4bit"


def main():
    """Quantize a pretrained model with GPTQ and run a single generation."""
    logger = logging.getLogger("NanoModel")
    logger.info("Loading tokenizer...")

    # load calibration dataset
    calibration_dataset = load_dataset(
        "allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train"
    ).select(range(1024))["text"]

    # quant_config
    quant_config = QuantizeConfig(
        bits=4,
        group_size=128,
    )

    # quantization
    logger.info("Loading pretrained model for quantization...")
    model = AutoNanoModel.load(model_id_path, quant_config)
    logger.info("Quantizing model...")
    model.quantize(calibration_dataset)
    logger.info(f"Saving quantized model to: {quant_model_path}")
    model.save(quant_model_path)

    # Run a quick generation to validate the quantized weights.
    tokenizer = AutoTokenizer.from_pretrained(model_id_path, use_fast=True)
    model = AutoNanoModel.load(quant_model_path, device=get_best_device())
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

    # quant_config = QuantizeConfig(
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
