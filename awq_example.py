import os
import logging
from transformers import AutoTokenizer
from nanomodel import AutoNanoModel, QuantizeConfig, get_best_device
from nanomodel.quantization import FORMAT, METHOD, QUANT_CONFIG_FILENAME

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

pretrained_model_id = "Qwen/Qwen3-1.7B"
quantized_model_id = "qwen3-1.7b-awq-4bit"

# Main pipeline
def main():
    logger = logging.getLogger("NanoModel")
    logger.info("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)

    calibration_dataset = [tokenizer("GPTQ is a method that compresses large language models by converting their weights to lower precision (like 4-bit) after training, making them smaller and faster with minimal accuracy loss.")]

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=128,
        quant_method=METHOD.AWQ,  # switch to METHOD.AWQ or METHOD.GPTQ as needed
        format=FORMAT.GEMV,        # FORMAT.MARLIN / FORMAT.GEMM / FORMAT.GEMV also available
    ) 

    logger.info("Loading pretrained model for quantization...")
    model = AutoNanoModel.load(pretrained_model_id, quantize_config)

    logger.info("Quantizing model...")
    model.quantize(calibration_dataset)

    logger.info(f"Saving quantized model to: {quantized_model_id}")
    model.save(quantized_model_id)

    model = AutoNanoModel.load(quantized_model_id, device=get_best_device())

    # inference with model.generate
    print(tokenizer.decode(model.generate(**tokenizer("GPTQ is", return_tensors="pt").to(model.device))[0]))

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

    # quantize_config = QuantizeConfig(
    #     bits=4,  # default 4-bit [2, 3, 4, 8]
    #     group_size=128,  # default 128
    #     quant_method=METHOD.AWQ

        # desc_act: Optional[bool] = field(default=None)
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

        # rotation ["hadamard", "random"]
        # is_marlin_format: bool = False
        # zero_point: bool = field(default=True)
    # )