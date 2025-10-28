"""Load a pre-quantized checkpoint and run a quick inference pass."""
import os

from transformers import AutoTokenizer

from nanomodel import AutoNanoModel, get_best_device

# Normalize CUDA detection and allocator behaviour for consistent runs.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

pretrained_model_id = "Qwen/Qwen3-1.7B"
quantized_model_id = "qwen3-1.7b-gptq-4bit"

def main():
    """Load the quantized weights and print a one-shot model response."""
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)

    device = get_best_device()
    model = AutoNanoModel.load(quantized_model_id, device=device)

    # Run a quick generation to confirm the quantized model is usable.
    prompt_inputs = tokenizer("What model you are ?", return_tensors="pt").to(model.device)
    generated = model.generate(**prompt_inputs)[0]
    print(tokenizer.decode(generated))

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
