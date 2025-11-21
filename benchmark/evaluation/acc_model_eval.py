"""Run lm-eval on an original model and its GPTQ quantized version, save CSV."""

import csv
import logging
import os
from typing import Dict, Iterable, List, Tuple

import torch

from nanomodel import AutoNanoModel
from nanomodel.utils.eval import EVAL
from transformers import AutoModelForCausalLM

model_id = "Qwen/Qwen3-4B-Instruct-2507"
gptq_model_id = (
    "/home/sd24191/git_project/QModel-Demo/quantized_models/"
    "Qwen3-4B-Instruct-2507-GPTQ-4bit"
)

csv_output_path = "acc_comparison.csv"
tasks_to_eval = [
    EVAL.LM_EVAL.MMLU,
    EVAL.LM_EVAL.ARC_CHALLENGE,
    EVAL.LM_EVAL.GSM8K_COT,
    EVAL.LM_EVAL.GSM8K_PLATINUM_COT,
]


def _iter_numeric_metrics(results: Dict) -> Iterable[Tuple[str, str, float]]:
    """Yield (task, metric, value) for numeric entries inside an lm-eval result dict."""
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                # skip nested entries such as stderr/bootstrap errors
                continue
            try:
                yield task, metric_name, float(metric_value)
            except (TypeError, ValueError):
                continue


def flatten_eval_results(eval_result: Dict) -> List[Tuple[str, str, str, float]]:
    """Flatten lm-eval results + groups to a list of (task, metric, section, value)."""
    flattened: List[Tuple[str, str, str, float]] = []

    core_results = eval_result.get("results", {})
    flattened.extend(
        (task, metric, "results", value)
        for task, metric, value in _iter_numeric_metrics(core_results)
    )

    group_results = eval_result.get("groups", {})
    flattened.extend(
        (task, metric, "groups", value)
        for task, metric, value in _iter_numeric_metrics(group_results)
    )

    return flattened


def write_csv(rows: List[List[str]]) -> None:
    """Append comparison rows to CSV with header creation."""
    file_exists = os.path.isfile(csv_output_path)
    with open(csv_output_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["model_label", "model_id", "task", "metric", "section", "value"])
        writer.writerows(rows)


def evaluate_model(model_label: str, model_path: str, is_quantized: bool) -> Dict:
    logging.info("Running lm-eval for %s (%s)", model_label, model_path)
    # AutoNanoModel.eval cannot load un-quantized checkpoints directly, so load
    # baseline with transformers and pass the model instance; quantized models
    # can be passed as a path.
    if not is_quantized:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model_or_path = base_model
    else:
        model_or_path = model_path

    return AutoNanoModel.eval(
        model_or_path,
        framework=EVAL.LM_EVAL,
        tasks=tasks_to_eval,
    )


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    models_to_eval = [
        {"label": "baseline", "path": model_id, "quantized": False},
        {"label": "gptq_4bit", "path": gptq_model_id, "quantized": True},
    ]

    csv_rows: List[List[str]] = []
    for entry in models_to_eval:
        label = entry["label"]
        path = entry["path"]
        quantized = entry["quantized"]

        eval_output = evaluate_model(label, path, quantized)
        for task, metric, section, value in flatten_eval_results(eval_output):
            csv_rows.append([label, path, task, metric, section, f"{value:.4f}"])

    if not csv_rows:
        logging.warning("No metrics found; CSV not written.")
        return

    write_csv(csv_rows)
    logging.info("Comparison complete. Results saved to %s", csv_output_path)


if __name__ == "__main__":
    main()
