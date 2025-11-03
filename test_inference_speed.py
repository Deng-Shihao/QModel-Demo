"""Command-line benchmark helper to measure NanoModel inference throughput."""

import argparse
import os
import time
from typing import Iterable, Sequence

from transformers import AutoTokenizer

from nanomodel import AutoNanoModel, get_best_device
from nanomodel.utils.backend import BACKEND
from nanomodel.utils.logger import setup_logger
from nanomodel.utils.torch import torch_empty_cache


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


logger = setup_logger(__name__)


DEFAULT_PROMPTS: Sequence[str] = (
    "I am in Paris and I",
    "The capital of the United Kingdom is",
    "The largest ocean on Earth is",
    "The world’s longest river is",
    "The tallest mountain in the world is",
    "The currency used in Japan is",
    "How to consult a dictionary?",
    "What is the boiling point of water in degrees Celsius?",
    "Which is the most widely used Internet search engine in the world?",
    "What is the official language of France?",
)


def _load_prompts_from_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        prompts = [line.strip() for line in fh if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path!r}.")
    return prompts


class InferenceSpeed:
    """Benchmark helper that measures tokens/second for quantized checkpoints."""

    NATIVE_MODEL_ID = "/monster/data/model/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2"
    BITBLAS_NATIVE_MODEL_ID = "/monster/data/model/opt-125M-autoround-lm_head-false-symTrue"

    def __init__(
        self,
        *,
        prompts: Sequence[str] | None = None,
        num_runs: int = 20,
        max_new_tokens: int = 10,
        negative_delta: float = 0.25,
        positive_delta: float = 0.25,
    ) -> None:
        if num_runs <= 0:
            raise ValueError("`num_runs` must be a positive integer.")
        if max_new_tokens <= 0:
            raise ValueError("`max_new_tokens` must be a positive integer.")
        if negative_delta < 0 or positive_delta < 0:
            raise ValueError("`negative_delta` and `positive_delta` must be non-negative.")

        self.prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)
        if not self.prompts:
            raise ValueError("At least one prompt is required to benchmark inference speed.")

        self.num_runs = num_runs
        self.max_new_tokens = max_new_tokens
        self.negative_delta = negative_delta
        self.positive_delta = positive_delta

    def _log_run_stats(
        self,
        backend: BACKEND,
        times: Iterable[float],
        tokens: Iterable[int],
        sum_time: float,
        sum_tokens: int,
        avg_tokens_per_second: float,
        *,
        phase: str = "",
    ) -> None:
        """Emit structured stats for warm-up and benchmark phases."""
        times_list = list(times)
        tokens_list = list(tokens)
        phase_text = phase or ""
        phase_label = f"{phase_text} Result Info" if phase_text else "Result Info"
        token_label = (
            "New Tokens (Size Per Batch Request)"
            if phase_text.lower() == "warm-up"
            else "New Tokens"
        )
        header = f"**************** {backend.value} {phase_label}****************"
        footer = f"****************  {backend.value} {phase_label} End****************"
        logger.info(
            "%s\nTimes: %s\n%s: %s\nSum Times: %.4f\nSum New Tokens: %s\n"
            "New Token Per Second: %.2f token/s\n%s",
            header,
            times_list,
            token_label,
            tokens_list,
            sum_time,
            sum_tokens,
            avg_tokens_per_second,
            footer,
        )

    def inference(
        self,
        model_path: str,
        backend: BACKEND | str,
        *,
        tokens_per_second: float | None = None,
        assert_result: bool | None = None,
        optimize: bool = False,
        fullgraph: bool = False,
        warmup_runs: int = 0,
        device: str | None = None,
    ) -> float:
        """Run the inference benchmark and optionally validate throughput."""
        backend_enum = backend if isinstance(backend, BACKEND) else BACKEND(backend)
        if warmup_runs < 0:
            raise ValueError("`warmup_runs` must be >= 0.")

        runtime_device = device or get_best_device()
        model = AutoNanoModel.from_quantized(
            model_path,
            backend=backend_enum,
            device=runtime_device,
        )

        if optimize:
            model.optimize(fullgraph=fullgraph)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer(
            self.prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            padding_side="left",
        ).to(model.device)

        warmup_times: list[float] = []
        warmup_tokens: list[int] = []
        measured_times: list[float] = []
        measured_tokens: list[int] = []

        # Kernels like BitBLAS, IPEX, and Triton perform JIT compilation, so run warm-up
        # iterations before collecting speed measurements.
        if warmup_runs > 0:
            pb = logger.pb(range(warmup_runs)).title("Warmup")
            for _ in pb:
                start_time = time.time()
                result = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
                elapsed_time = time.time() - start_time
                warmup_times.append(elapsed_time)

                batch_input_len = inputs["input_ids"].shape[1]
                for row in result:
                    new_token_count = len(row[batch_input_len:])
                    warmup_tokens.append(new_token_count)

            warmup_sum_time = sum(warmup_times)
            warmup_sum_tokens = sum(warmup_tokens)
            warmup_avg = (
                round(warmup_sum_tokens / warmup_sum_time, 2)
                if warmup_sum_time > 0
                else 0.0
            )
            self._log_run_stats(
                backend_enum,
                warmup_times,
                warmup_tokens,
                warmup_sum_time,
                warmup_sum_tokens,
                warmup_avg,
                phase="Warm-up",
            )

        pb = logger.pb(range(self.num_runs)).title("Run")
        for _ in pb:
            start_time = time.time()
            result = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            elapsed_time = time.time() - start_time
            measured_times.append(elapsed_time)

            batch_input_len = inputs["input_ids"].shape[1]
            for row in result:
                new_token_count = len(row[batch_input_len:])
                measured_tokens.append(new_token_count)

        if not measured_times:
            raise RuntimeError("Benchmark produced no measurements; check `num_runs`.")

        sum_time = sum(measured_times)
        sum_tokens = sum(measured_tokens)
        avg_tokens_per_second = (
            round(sum_tokens / sum_time, 2) if sum_time > 0 else 0.0
        )

        self._log_run_stats(
            backend_enum,
            measured_times,
            measured_tokens,
            sum_time,
            sum_tokens,
            avg_tokens_per_second,
        )

        should_assert = tokens_per_second is not None if assert_result is None else assert_result
        if should_assert and tokens_per_second is not None and sum_time > 0:
            diff_pct = (avg_tokens_per_second / tokens_per_second) * 100
            negative_pct = 100 * (1 - self.negative_delta)
            positive_pct = 100 * (1 + self.positive_delta)

            message = (
                f"{backend_enum.value}: Actual tokens Per Second: {avg_tokens_per_second}, "
                f"expected = `{tokens_per_second}` diff {diff_pct:.2f}% "
                f"is out of the expected range [{negative_pct}-{positive_pct}%]"
            )

            if not (negative_pct <= diff_pct <= positive_pct):
                raise AssertionError(message)

        del model
        torch_empty_cache()
        return avg_tokens_per_second


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark throughput for quantized NanoModel checkpoints."
    )
    parser.add_argument(
        "model_path",
        help="Local path or Hugging Face repo ID for the quantized checkpoint.",
    )
    parser.add_argument(
        "--backend",
        default=BACKEND.AUTO.value,
        choices=[b.value for b in BACKEND],
        help="Execution backend to use for inference (default: auto).",
    )
    parser.add_argument(
        "--expected-tps",
        type=float,
        default=None,
        help="Optional expected tokens/sec baseline; violations raise AssertionError.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of measured runs to execute (default: 20).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Warm-up iterations to run before measuring (default: 0).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Number of new tokens to generate per request (default: 10).",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional path to a newline-separated list of prompts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit device to place the model on (e.g., cuda:0, cpu).",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable torch.compile-based optimization for the quantized model.",
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="Attempt full-graph compilation when --optimize is set.",
    )
    parser.add_argument(
        "--neg-delta",
        type=float,
        default=0.25,
        help="Allowed fractional slowdown (0.25 -> tolerate 25%% slowdown).",
    )
    parser.add_argument(
        "--pos-delta",
        type=float,
        default=0.25,
        help="Allowed fractional speedup (0.25 -> tolerate 25%% speedup).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = (
        _load_prompts_from_file(args.prompts_file)
        if args.prompts_file is not None
        else None
    )

    tester = InferenceSpeed(
        prompts=prompts,
        num_runs=args.runs,
        max_new_tokens=args.max_new_tokens,
        negative_delta=args.neg_delta,
        positive_delta=args.pos_delta,
    )

    avg_tps = tester.inference(
        args.model_path,
        args.backend,
        tokens_per_second=args.expected_tps,
        warmup_runs=args.warmup_runs,
        optimize=args.optimize,
        fullgraph=args.fullgraph,
        device=args.device,
    )

    logger.info("Benchmark complete. Average throughput: %.2f token/s", avg_tps)


if __name__ == "__main__":
    main()
