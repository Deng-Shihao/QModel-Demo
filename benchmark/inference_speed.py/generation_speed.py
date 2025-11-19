"""Benchmark helper to measure end-to-end text generation speed with NanoModel."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from transformers import AutoTokenizer

from nanomodel import AutoNanoModel, get_best_device
from nanomodel.utils.backend import BACKEND


# Normalize CUDA detection and allocator behaviour for consistent runs.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


LOGGER = logging.getLogger("generation_speed")


DEFAULT_TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MODEL_ID = "/home/sd24191/git_project/QModel-Demo/quantized_models/Qwen3-4B-Instruct-2507-GPTQ-4bit"

DEFAULT_PROMPTS: Sequence[str] = (
    "Explain the benefits of post-training quantization for LLMs.",
    "List two trade-offs when compressing transformer weights.",
    "Provide a short haiku about efficient inference.",
    "What optimizations improve generation throughput on GPUs?",
)


@dataclass
class RunResult:
    """Container that holds per-run measurements."""

    run_index: int
    elapsed: float
    new_tokens: int
    tokens_per_second: float
    memory: Optional[Dict[str, int]] = None
    sample: Optional[str] = None


def format_bytes(num_bytes: int) -> str:
    """Convert a raw byte count to a human-readable string."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def collect_memory_stats(device: torch.device) -> Optional[Dict[str, int]]:
    """Return backend-specific memory stats when available."""
    device_type = device.type
    if device_type == "cuda":
        torch.cuda.synchronize(device)
        return {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
        }
    if device_type == "mps":
        stats: Dict[str, int] = {"allocated": torch.mps.current_allocated_memory()}
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


def format_memory(stats: Optional[Dict[str, int]]) -> str:
    if not stats:
        return "N/A"
    return ", ".join(f"{key}={format_bytes(value)}" for key, value in stats.items())


def build_prompt_pool(prompt: Optional[str], prompt_file: Optional[str]) -> List[str]:
    """Load prompts from CLI arguments or fall back to built-ins."""
    if prompt_file:
        file_path = Path(prompt_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file {file_path} does not exist.")
        prompts = [
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not prompts:
            raise ValueError(f"No prompts found inside {prompt_file}.")
        return prompts

    if prompt:
        return [prompt]

    return list(DEFAULT_PROMPTS)


def cycle_batch(prompts: Sequence[str], batch_size: int, run_index: int) -> List[str]:
    """Return a batch of prompts, cycling through the list as runs progress."""
    if batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")
    if not prompts:
        raise ValueError("At least one prompt is required.")

    batch: List[str] = []
    start = (run_index * batch_size) % len(prompts)
    for offset in range(batch_size):
        batch.append(prompts[(start + offset) % len(prompts)])
    return batch


def resolve_device(device_like: Optional[str | torch.device]) -> torch.device:
    if isinstance(device_like, torch.device):
        return device_like
    if device_like is None:
        return torch.device(get_best_device())
    return torch.device(device_like)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def count_new_tokens(
    generated: torch.LongTensor,
    input_ids: torch.LongTensor,
    pad_token_id: Optional[int],
) -> List[int]:
    """Compute the number of newly generated tokens per sample."""
    generated_cpu = generated.detach().to("cpu")
    input_cpu = input_ids.detach().to("cpu")

    if pad_token_id is not None:
        prompt_lengths = (input_cpu != pad_token_id).sum(dim=1)
    else:
        prompt_lengths = torch.full(
            (input_cpu.size(0),),
            fill_value=input_cpu.size(1),
            dtype=torch.long,
        )

    per_sample: List[int] = []
    for idx in range(generated_cpu.size(0)):
        prompt_len = int(prompt_lengths[idx])
        tail = generated_cpu[idx, prompt_len:]
        if pad_token_id is None:
            per_sample.append(int(tail.numel()))
            continue

        valid_tokens = 0
        for token_id in tail.tolist():
            if token_id == pad_token_id:
                break
            valid_tokens += 1
        per_sample.append(valid_tokens)

    return per_sample


def decode_sample(
    tokenizer: AutoTokenizer,
    generated: torch.LongTensor,
    input_ids: torch.LongTensor,
    pad_token_id: Optional[int],
) -> str:
    """Decode only the newly generated tokens for the first sample."""
    input_cpu = input_ids[0].detach().to("cpu")
    generated_cpu = generated[0].detach().to("cpu")
    if pad_token_id is not None:
        prompt_len = int((input_cpu != pad_token_id).sum().item())
    else:
        prompt_len = input_cpu.size(0)
    tail = generated_cpu[prompt_len:]
    if pad_token_id is not None:
        collected: List[int] = []
        for token_id in tail.tolist():
            if token_id == pad_token_id:
                break
            collected.append(token_id)
        new_tokens_list = collected
    else:
        new_tokens_list = tail.tolist()
    if not new_tokens_list:
        return "[no new tokens]"
    return tokenizer.decode(new_tokens_list, skip_special_tokens=True).strip()


def maybe_write_csv(path: Optional[str], results: Iterable[RunResult]) -> None:
    if not path:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "elapsed_s", "new_tokens", "tokens_per_second"])
        for result in results:
            writer.writerow(
                [
                    result.run_index,
                    f"{result.elapsed:.6f}",
                    result.new_tokens,
                    f"{result.tokens_per_second:.4f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure decoding throughput for quantized NanoModel checkpoints.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Path or Hugging Face repo ID for the quantized checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER_ID,
        help="Tokenizer identifier (defaults to the pretrained model ID).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=BACKEND.AUTO.value,
        choices=[backend.value for backend in BACKEND],
        help="Execution backend to use when loading the quantized weights.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to benchmark. Overrides built-in prompts.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to a newline-delimited prompt file. Overrides --prompt.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts to run per request.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of measured runs.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up iterations before measurement.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used when --sample is enabled.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold used when --sample is enabled.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable nucleus sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Optional repetition penalty passed to model.generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to place the model on (e.g. cuda:0, cpu). Defaults to get_best_device().",
    )
    parser.add_argument(
        "--track-memory",
        action="store_true",
        help="Report backend memory stats alongside throughput.",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=0,
        help="Print the decoded output for the first N runs.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write per-run metrics as CSV.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity level for the benchmark logs.",
    )
    return parser.parse_args()


def build_generate_kwargs(args: argparse.Namespace) -> Dict:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.sample,
    }
    if args.sample:
        kwargs.update(
            temperature=args.temperature,
            top_p=args.top_p,
        )
    return kwargs


def benchmark(args: argparse.Namespace) -> List[RunResult]:
    prompts = build_prompt_pool(args.prompt, args.prompt_file)
    device = resolve_device(args.device)

    LOGGER.info("Loading tokenizer %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Tokenizer is missing `pad_token_id` and `eos_token_id`; please provide a compatible tokenizer."
        )
    tokenizer.padding_side = "left"

    LOGGER.info(
        "Loading quantized model %s on %s via backend '%s'",
        args.model,
        device,
        args.backend,
    )
    model = AutoNanoModel.load(
        args.model,
        device=device,
        backend=BACKEND(args.backend),
    )

    model_device = (
        torch.device(model.device)
        if not isinstance(model.device, torch.device)
        else model.device
    )
    generate_kwargs = build_generate_kwargs(args)

    # Warm-up iterations help kernels settle (e.g., Triton compilation).
    if args.warmup > 0:
        LOGGER.info("Running %d warm-up iteration(s)...", args.warmup)
        for idx in range(args.warmup):
            batch = cycle_batch(prompts, args.batch_size, idx)
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(model_device)
            synchronize_if_needed(model_device)
            _ = model.generate(
                **inputs,
                **generate_kwargs,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            synchronize_if_needed(model_device)

    LOGGER.info(
        "Starting benchmark (%d run(s), batch size %d, %d max new tokens)...",
        args.runs,
        args.batch_size,
        args.max_new_tokens,
    )

    results: List[RunResult] = []
    for run_idx in range(args.runs):
        batch = cycle_batch(prompts, args.batch_size, run_idx)
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model_device)

        if args.track_memory and model_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model_device)

        synchronize_if_needed(model_device)
        start_time = time.perf_counter()
        generated = model.generate(
            **inputs,
            **generate_kwargs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        synchronize_if_needed(model_device)
        elapsed = time.perf_counter() - start_time

        new_tokens_per_sample = count_new_tokens(
            generated,
            inputs["input_ids"],
            tokenizer.pad_token_id,
        )
        total_new_tokens = sum(new_tokens_per_sample)
        tokens_per_second = (
            total_new_tokens / elapsed if elapsed > 0 and total_new_tokens > 0 else 0.0
        )

        memory_stats = collect_memory_stats(model_device) if args.track_memory else None
        sample_output: Optional[str] = None
        if args.print_samples and run_idx < args.print_samples:
            sample_output = decode_sample(
                tokenizer,
                generated,
                inputs["input_ids"],
                tokenizer.pad_token_id,
            )

        LOGGER.info(
            "Run %d/%d: %.2fs | %d new tokens | %.2f tok/s%s",
            run_idx + 1,
            args.runs,
            elapsed,
            total_new_tokens,
            tokens_per_second,
            f" | Memory: {format_memory(memory_stats)}" if memory_stats else "",
        )
        if sample_output:
            LOGGER.info("Sample output: %s", sample_output)

        results.append(
            RunResult(
                run_index=run_idx + 1,
                elapsed=elapsed,
                new_tokens=total_new_tokens,
                tokens_per_second=tokens_per_second,
                memory=memory_stats,
                sample=sample_output,
            )
        )

    maybe_write_csv(args.csv, results)
    return results


def summarize(results: Sequence[RunResult]) -> None:
    if not results:
        LOGGER.warning("No benchmark results to summarize.")
        return

    total_tokens = sum(run.new_tokens for run in results)
    total_time = sum(run.elapsed for run in results)
    throughputs = [
        run.tokens_per_second for run in results if run.tokens_per_second > 0
    ]
    avg_throughput = total_tokens / total_time if total_time > 0 else 0.0

    LOGGER.info("======== Benchmark Summary ========")
    LOGGER.info("Total new tokens: %d", total_tokens)
    LOGGER.info("Total time: %.2f s", total_time)
    LOGGER.info("Average throughput: %.2f tok/s", avg_throughput)
    if throughputs:
        LOGGER.info("Per-run avg: %.2f tok/s", statistics.mean(throughputs))
        LOGGER.info("Per-run median: %.2f tok/s", statistics.median(throughputs))
        LOGGER.info("Best run: %.2f tok/s", max(throughputs))
        LOGGER.info("Worst run: %.2f tok/s", min(throughputs))
    LOGGER.info("===================================")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=getattr(logging, args.log_level.upper()),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    results = benchmark(args)
    summarize(results)


if __name__ == "__main__":
    main()
