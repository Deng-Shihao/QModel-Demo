"""Interactive chat session powered by a quantized NanoModel checkpoint."""

import logging
import os
import time
from typing import Dict, List, Optional

from transformers import AutoTokenizer

from nanomodel import AutoNanoModel, get_best_device

import torch

# Environment configuration for predictable CUDA ordering and expandable allocator
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

PRETRAINED_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
QUANTIZED_MODEL_ID = "/home/sd24191/git_project/QModel-Demo/quantized_models/Qwen3-4B-Instruct-2507-GPTQ-4bit"


def format_bytes(num_bytes: int) -> str:
    """Convert raw byte counts into a human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024


def collect_memory_stats(device: torch.device) -> Optional[Dict[str, int]]:
    """Return memory statistics for the active device if available."""
    device_type = device.type

    if device_type == "cuda":
        torch.cuda.synchronize(device)
        return {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
        }

    if device_type == "mps":
        stats = {
            "allocated": torch.mps.current_allocated_memory(),
        }
        max_alloc_fn = getattr(torch.mps, "max_memory_allocated", None)
        if callable(max_alloc_fn):
            stats["max_allocated"] = max_alloc_fn()
        return stats

    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {"rss": mem_info.rss}
    except ImportError:
        return None


def build_chat_prompt(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Render the conversation into a single prompt. Prefer the tokenizer's chat template
    when available to keep model-specific system formatting intact.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Fallback prompt if the tokenizer does not ship a chat template.
    conversation = []
    for message in messages:
        role = message["role"].capitalize()
        conversation.append(f"{role}: {message['content']}")
    conversation.append("Assistant:")
    return "\n".join(conversation)


def chat():
    """Run an interactive REPL loop that tracks latency and memory usage."""
    logging.getLogger("NanoModel").info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_ID, use_fast=True)

    device = get_best_device()
    logging.getLogger("NanoModel").info("Loading quantized model...")
    model = AutoNanoModel.load(QUANTIZED_MODEL_ID, device=device)
    model_device = (
        model.device
        if isinstance(model.device, torch.device)
        else torch.device(model.device)
    )

    if model_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model_device)

    chat_history: List[Dict[str, str]] = []
    print("Interactive chat is ready. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        chat_history.append({"role": "user", "content": user_input})
        turn_start = time.perf_counter()

        prompt = build_chat_prompt(tokenizer, chat_history)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        if model_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model_device)

        generate_kwargs = {
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }

        gen_start = time.perf_counter()
        output_ids = model.generate(**inputs, **generate_kwargs)
        gen_duration = time.perf_counter() - gen_start

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        assistant_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generated_tokens = new_tokens.shape[-1]
        tokens_per_second = (
            generated_tokens / gen_duration
            if gen_duration > 0 and generated_tokens > 0
            else 0.0
        )

        if not assistant_text:
            assistant_text = "[No response generated]"

        chat_history.append({"role": "assistant", "content": assistant_text})
        print(f"Assistant: {assistant_text}")

        memory_stats = collect_memory_stats(model_device)

        total_duration = time.perf_counter() - turn_start
        if memory_stats:
            stats_str = ", ".join(
                f"{key}={format_bytes(value)}" for key, value in memory_stats.items()
            )
            print(
                f"[Metrics] {generated_tokens} tokens | total={total_duration:.2f}s "
                f"| generate={gen_duration:.2f}s | throughput={tokens_per_second:.2f} tok/s | Memory: {stats_str}"
            )
        else:
            print(
                f"[Metrics] {generated_tokens} tokens | total={total_duration:.2f}s "
                f"| generate={gen_duration:.2f}s | throughput={tokens_per_second:.2f} tok/s"
            )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    chat()
