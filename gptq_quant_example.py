import os
import logging
from transformers import AutoTokenizer
from nanomodel import AutoNanoModel, QuantizeConfig

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

pretrained_model_id = "Qwen/Qwen3-1.7B"
quantized_model_id = "qwen3-1.7-4bit"

# Main pipeline
def main():
    logger = logging.getLogger("NanoModel")
    logger.info("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)

    # Build calibration dataset
    calibration_dataset = [
        tokenizer("NanoModel is an easy-to-use model quantization library with user-friendly APIs, based on GPTQ algorithm.")
    ]

    quantize_config = QuantizeConfig(
        bits=4,          # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
    )

    # Load un-quantized model (by default, the model will always be loaded into CPU memory)
    # model = AutoNanoModel.load(pretrained_model_id, quantize_config)
    logger.info("Loading pretrained model for quantization...")
    model = AutoNanoModel.load(pretrained_model_id, quantize_config)

    # Quantize model
    # The calibration_dataset should be a list of dicts with keys "input_ids" and "attention_mask"
    logger.info("Quantizing model...")
    model.quantize(calibration_dataset)

    # Save quantized model
    logger.info(f"Saving quantized model to: {quantized_model_id}")
    model.save(quantized_model_id)

# Entry point
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
