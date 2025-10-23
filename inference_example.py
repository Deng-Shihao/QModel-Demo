import os

from transformers import AutoTokenizer

from nanomodel import AutoNanoModel, get_best_device

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

pretrained_model_id = "Qwen/Qwen3-1.7B"
quantized_model_id = "qwen3-1.7b-gptq-4bit"

def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)

    device = get_best_device()
    model = AutoNanoModel.load(quantized_model_id, device=device)

    # inference with model.generate
    print(tokenizer.decode(model.generate(**tokenizer("What model youa are ?", return_tensors="pt").to(model.device))[0]))

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
