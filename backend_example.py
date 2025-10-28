"""Demonstrate backend selection for quantized and runtime-generated models."""
import os
import subprocess
import sys
from argparse import ArgumentParser

from transformers import AutoTokenizer

from nanomodel import BACKEND, AutoNanoModel, QuantizeConfig, get_best_device

# Normalize CUDA device discovery and torch allocator behavior across platforms.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

pretrained_model_id = "Qwen/Qwen3-1.7B"
quantized_model_id = "qwen3-1.7b-gptq-4bit"

def main():
    """Select a runtime backend, quantize if needed, then run a short generation."""
    global quantized_model_id

    parser = ArgumentParser()
    parser.add_argument("--backend", choices=['AUTO', 'TRITON',  'MARLIN', 'CUDA', 'SGLANG', 'VLLM'])
    args = parser.parse_args()

    backend = BACKEND(args.backend.lower())
    device = get_best_device(backend)

    if backend == BACKEND.SGLANG:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm>=0.8.5"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sglang[srt]>=0.3.2"])
    elif backend == BACKEND.VLLM:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm>=0.8.5"])

    prompt = "What model you are ?"

    if backend == BACKEND.SGLANG or backend == BACKEND.VLLM:
        quantized_model_id = "qwen3-1.7B-GPTQ-4bit"

        if backend == BACKEND.SGLANG:
            # SGLang piggybacks on vLLM for quantized execution; ensure both are installed.
            subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm>=0.8.5"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sglang[srt]>=0.3.2"])
            model = AutoNanoModel.load(
                quantized_model_id,
                device=device,
                backend=backend,
                disable_flashinfer=True
            )

            output = model.generate(prompts=prompt, temperature=0.8, top_p=0.95)
            model.shutdown()
            del model
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm>=0.8.5"])
            model = AutoNanoModel.load(
                quantized_model_id,
                device=device,
                backend=backend,
                disable_flashinfer=True,
            )
            output = model.generate(prompts=prompt, temperature=0.8, top_p=0.95)[0].outputs[0].text
            model.shutdown()
            del model
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)
        examples = [
            tokenizer(
                "I am in Paris and I can't wait to explore its beautiful"
            )
        ]

        # Configure GPTQ quantization; MARLIN can disable act_order for faster inference.
        quantize_config = QuantizeConfig(
            bits=4,  # quantize model to 4-bit
            group_size=128,  # it is recommended to set the value to 128
            # set to False can significantly speed up inference but the perplexity may slightly bad
            act_order=False if backend == BACKEND.MARLIN else True,
        )

        # load un-quantized model, by default, the model will always be loaded into CPU memory
        model = AutoNanoModel.load(pretrained_model_id, quantize_config)

        # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
        model.quantize(examples)

        # save quantized model
        model.save(quantized_model_id)
        tokenizer.save_pretrained(quantized_model_id)

        model = AutoNanoModel.load(
            quantized_model_id,
            device=device,
            backend=backend,
        )

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model.generate(**inp, num_beams=1, max_new_tokens=10)
        output = tokenizer.decode(res[0])

    print(f"Prompt: {prompt}, Generated text: {output}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
