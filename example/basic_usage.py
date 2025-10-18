import os

from transformers import AutoTokenizer
from nanomodel import AutoNanoModel, QuantizeConfig, get_best_device

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

pretrained_model_id = "Qwen/Qwen3-0.6B"
quantized_model_id = "qwen3-0.6-4bit"


def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)
    calibration_dataset = [
        tokenizer("NanoModel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.")
    ]

    quantize_config = QuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    # model = GPTQModel.load(pretrained_model_id, quantize_config)
    model = AutoNanoModel.load(pretrained_model_id, quantize_config)

    # quantize model, the calibration_dataset should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(calibration_dataset)

    # save quantized model
    model.save(quantized_model_id)

    # save quantized model using (safetensors)
    model.save(quantized_model_id)

    # load quantized model to the first GPU
    device = get_best_device()
    model = AutoNanoModel.load(quantized_model_id, device=device)

    # download quantized model from Hugging Face Hub and load to the first GPU
    # model = GPTQModel.from_quantized(repo_id, device="cuda:0",)

    # inference with model.generate
    print(tokenizer.decode(model.generate(**tokenizer("NanoModel is", return_tensors="pt").to(model.device))[0]))


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
