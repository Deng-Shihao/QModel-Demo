from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from .data import collate_data

try:  # Optional dependency for huggingface datasets support
    from datasets import Dataset as HFDataset
    from datasets import IterableDataset as HFIterableDataset
except Exception:  # pragma: no cover - datasets may not be installed
    HFDataset = None
    HFIterableDataset = None

if TYPE_CHECKING:
    from ..models.base import BaseNanoModel


def batched(iterable: Iterable[Any], n: int, process_func):
    """Yield fixed sized batches from iterable, optionally applying a transform."""
    assert n >= 1, "batch size must be at least one"
    from itertools import islice

    iterator = iter(iterable)

    while batch := tuple(islice(iterator, n)):
        if process_func is None:
            yield batch
        else:
            yield [process_func(item) for item in batch]


def prepare_calibration_dataset(
    model: "BaseNanoModel",
    calibration_dataset: Union[
        List[Dict[str, Union[List[int], torch.LongTensor]]],
        List[str],
        List[List[int]],
        List[int],
        Iterable[Dict[str, Any]],
    ],
    calibration_dataset_concat_size: Optional[int] = None,
    calibration_dataset_sort: Optional[str] = None,
    batch_size: int = 1,
    calibration_data_min_length: int = 10,
    calibration_concat_separator: Optional[str] = None,
    logger=None,
):
    """
    Normalize the user provided calibration data into token batches that every processor can use.

    This mirrors GPTQModel's helper to keep the dataset preparation logic centralised and easier
    to maintain. All heavy lifting (tokenization, trimming, concatenation, batching) lives here.
    """

    def _materialize_examples(dataset):
        hf_dataset_types: Tuple[type, ...] = tuple(
            dataset_type
            for dataset_type in (HFDataset, HFIterableDataset)
            if dataset_type is not None
        )

        if isinstance(dataset, str):
            raise ValueError(
                "Quantize: calibration dataset must be iterable, not a single string."
            )

        if hf_dataset_types and isinstance(dataset, hf_dataset_types):
            return list(dataset)

        try:
            return list(dataset)
        except TypeError as exc:
            raise ValueError(
                "Quantize: calibration dataset must be iterable and materializable."
            ) from exc

    def _require_tokenizer(reason: str):
        if model.tokenizer is None:
            raise ValueError(f"A tokenizer must be provided when {reason}.")

    def _to_2d_long_tensor(value: Any, name: str, idx: int) -> torch.Tensor:
        try:
            tensor = torch.as_tensor(value, dtype=torch.long)
        except Exception as exc:
            raise ValueError(
                f"Failed to convert `{name}` to a tensor for calibration item {idx}."
            ) from exc

        if tensor.ndim == 0:
            raise ValueError(
                f"`{name}` for item {idx} must be 1D or 2D, but got a scalar."
            )
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(
                f"`{name}` for item {idx} must be 1D or 2D, but got rank {tensor.ndim}."
            )
        return tensor

    def _pack_ids(ids_value: Any, mask_value: Any, idx: int):
        ids_tensor = _to_2d_long_tensor(ids_value, "input_ids", idx)

        if mask_value is None:
            mask_tensor = torch.ones_like(ids_tensor)
        else:
            mask_tensor = _to_2d_long_tensor(mask_value, "attention_mask", idx)
            if mask_tensor.shape != ids_tensor.shape:
                if mask_tensor.numel() == ids_tensor.numel():
                    mask_tensor = mask_tensor.reshape(ids_tensor.shape)
                else:
                    raise ValueError(
                        f"Shape mismatch for item {idx}: input_ids is {ids_tensor.shape}, attention_mask is {mask_tensor.shape}."
                    )

        return {
            "input_ids": ids_tensor.detach(),
            "attention_mask": mask_tensor.detach(),
        }

    def _tokenize_text_value(text_value: Any, idx: int):
        _require_tokenizer("calibration data contains raw text")
        tokenized = model.tokenizer(
            text_value, add_special_tokens=True, return_tensors="pt"
        )
        return _pack_ids(
            tokenized["input_ids"], tokenized.get("attention_mask"), idx
        )

    def _tokenize_messages_value(messages_value: Any, idx: int):
        _require_tokenizer("calibration data uses the `messages` feature")
        apply_fn = getattr(model.tokenizer, "apply_template", None)
        if apply_fn is None:
            raise ValueError(
                "Tokenizer must have `apply_template` to handle `messages` calibration data."
            )

        try:
            templated = apply_fn(messages_value, tokenize=False)
        except TypeError:
            templated = apply_fn(messages_value)

        if templated is None:
            raise ValueError(
                f"tokenizer.apply_template returned None for calibration item {idx}."
            )

        if hasattr(templated, "get") and "input_ids" in templated:
            return _pack_ids(
                templated["input_ids"], templated.get("attention_mask"), idx
            )
        if isinstance(templated, str):
            return _tokenize_text_value(templated, idx)
        if torch.is_tensor(templated) or (
            isinstance(templated, (list, tuple))
            and templated
            and isinstance(templated[0], int)
        ):
            return _pack_ids(templated, None, idx)

        raise ValueError(
            f"tokenizer.apply_template returned an unsupported type {type(templated)} for item {idx}."
        )

    def _convert_tensor_to_list(
        tensor: Union[torch.Tensor, List[List[int]], List[int]]
    ) -> List[List[int]]:
        if isinstance(tensor, torch.Tensor):
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return tensor.long().cpu().numpy().tolist()
        if isinstance(tensor[0], list):
            return tensor
        return [tensor]

    def _process_and_tokenize_examples(raw_examples: List[Any]):
        processed_examples = []
        for idx, example in enumerate(raw_examples):
            if isinstance(example, Mapping):
                example_dict = dict(example)
                if "messages" in example_dict:
                    processed_examples.append(
                        _tokenize_messages_value(example_dict["messages"], idx)
                    )
                elif "text" in example_dict:
                    processed_examples.append(
                        _tokenize_text_value(example_dict["text"], idx)
                    )
                elif "input_ids" in example_dict:
                    processed_examples.append(
                        _pack_ids(
                            example_dict["input_ids"],
                            example_dict.get("attention_mask"),
                            idx,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported calibration dict at index {idx}: keys={list(example_dict.keys())}"
                    )
            elif isinstance(example, str):
                processed_examples.append(_tokenize_text_value(example, idx))
            elif isinstance(example, (list, tuple)):
                if all(isinstance(x, int) for x in example):
                    processed_examples.append(_pack_ids(list(example), None, idx))
                else:
                    raise ValueError(
                        f"List-based calibration example at index {idx} must contain only integers."
                    )
            elif torch.is_tensor(example):
                processed_examples.append(_pack_ids(example, None, idx))
            else:
                raise ValueError(
                    f"Unsupported calibration example type {type(example)} at index {idx}."
                )
        return processed_examples

    def _trim_and_filter_examples(
        tokenized_examples: List[Dict[str, torch.Tensor]], min_length: int
    ):
        max_positions, max_positions_source = model._resolve_sequence_length_limit()
        trimmed_row_count = 0
        longest_trimmed_row = 0

        new_calibration_dataset = []
        too_short_count = 0

        for example in tokenized_examples:
            input_ids = _convert_tensor_to_list(example["input_ids"])
            attention_mask = _convert_tensor_to_list(example["attention_mask"])

            if max_positions is not None:
                trimmed = False
                trimmed_input_ids, trimmed_attention_mask = [], []
                for row_ids, row_mask in zip(input_ids, attention_mask):
                    if len(row_ids) > max_positions:
                        trimmed = True
                        trimmed_row_count += 1
                        longest_trimmed_row = max(longest_trimmed_row, len(row_ids))
                        trimmed_input_ids.append(row_ids[:max_positions])
                        trimmed_attention_mask.append(row_mask[:max_positions])
                    else:
                        trimmed_input_ids.append(row_ids)
                        trimmed_attention_mask.append(row_mask)
                if trimmed:
                    input_ids, attention_mask = (
                        trimmed_input_ids,
                        trimmed_attention_mask,
                    )

            if len(input_ids[0]) <= min_length:
                too_short_count += 1
                continue

            new_calibration_dataset.append(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )

        if logger is not None and too_short_count > 0:
            logger.warning(
                f"Quantize: {too_short_count} inputs with length <= {min_length} were removed."
            )
        if logger is not None and trimmed_row_count > 0:
            logger.info(
                f"Quantize: Trimmed {trimmed_row_count} calibration rows to {max_positions} tokens (source: {max_positions_source}, longest: {longest_trimmed_row})."
            )

        return new_calibration_dataset

    def _concatenate_examples(
        examples: List[Dict[str, List[List[int]]]],
        concat_size: int,
        separator: str,
    ):
        _require_tokenizer("`calibration_dataset_concat_size` is specified")

        concatenated_data = []
        input_ids_buffer, attention_mask_buffer = [], []
        current_length = 0

        new_line_tokens = model.tokenizer(separator, return_tensors="pt")
        new_line_ids = _convert_tensor_to_list(new_line_tokens["input_ids"])[0]
        new_line_mask = _convert_tensor_to_list(new_line_tokens["attention_mask"])[0]
        new_line_len = len(new_line_ids)

        for example in examples:
            ids, mask = example["input_ids"][0], example["attention_mask"][0]

            if current_length + len(ids) + new_line_len >= concat_size:
                if current_length > 0:
                    remaining = concat_size - current_length
                    if remaining > new_line_len:
                        input_ids_buffer.extend(new_line_ids)
                        input_ids_buffer.extend(ids[: remaining - new_line_len])
                        attention_mask_buffer.extend(new_line_mask)
                        attention_mask_buffer.extend(mask[: remaining - new_line_len])

                    concatenated_data.append(
                        {
                            "input_ids": [input_ids_buffer],
                            "attention_mask": [attention_mask_buffer],
                        }
                    )

                input_ids_buffer = ids[:concat_size]
                attention_mask_buffer = mask[:concat_size]
                current_length = len(input_ids_buffer)
            else:
                if current_length > 0:
                    input_ids_buffer.extend(new_line_ids)
                    attention_mask_buffer.extend(new_line_mask)
                    current_length += new_line_len

                input_ids_buffer.extend(ids)
                attention_mask_buffer.extend(mask)
                current_length += len(ids)

        if input_ids_buffer:
            padding_len = concat_size - len(input_ids_buffer)
            if padding_len > 0:
                input_ids_buffer.extend([model.tokenizer.pad_token_id] * padding_len)
                attention_mask_buffer.extend([0] * padding_len)
            concatenated_data.append(
                {
                    "input_ids": [input_ids_buffer],
                    "attention_mask": [attention_mask_buffer],
                }
            )

        return concatenated_data

    def _sort_examples(
        examples: List[Dict[str, List[List[int]]]], sort_mode: Optional[str]
    ):
        sort_mode = (sort_mode or "").lower()
        if sort_mode in {"asc", "desc"}:
            if logger is not None:
                logger.info(
                    f"Calibration: Sorting by length in {sort_mode}ending order."
                )
            return sorted(
                examples,
                key=lambda item: len(item["input_ids"][0]),
                reverse=sort_mode == "desc",
            )
        elif sort_mode == "shuffle":
            if logger is not None:
                logger.info("Calibration: Shuffling dataset randomly.")
            shuffled = examples[:]
            random.shuffle(shuffled)
            return shuffled
        else:
            if logger is not None:
                logger.info("Calibration: Using native dataset order.")
            return examples

    def _batch_examples(examples: List[Dict[str, Any]], batch_size: int):
        if getattr(model, "support_batch_quantize", True):
            batched_dataset = [
                collate_data(
                    examples[start : start + batch_size],
                    model.tokenizer.pad_token_id,
                )
                for start in range(0, len(examples), batch_size)
            ]

            if logger is not None:
                total_padded = sum(
                    (batch["attention_mask"] == 0).sum().item()
                    for batch in batched_dataset
                )
                total_non_padded = sum(
                    (batch["attention_mask"] == 1).sum().item()
                    for batch in batched_dataset
                )
                logger.info(
                    f"Calibration: Total tokens: {total_non_padded + total_padded} ({total_non_padded} non-padded, {total_padded} padded)."
                )

            return batched_dataset
        else:
            return [
                {"input_ids": torch.tensor(block["input_ids"], dtype=torch.long)}
                for block in examples
            ]

    raw_examples = _materialize_examples(calibration_dataset)
    if not raw_examples:
        raise ValueError("Quantize: calibration dataset is empty.")

    processed_examples = _process_and_tokenize_examples(raw_examples)

    trimmed_examples = _trim_and_filter_examples(
        processed_examples, min_length=calibration_data_min_length
    )

    if calibration_dataset_concat_size:
        separator = (
            calibration_concat_separator
            if calibration_concat_separator is not None
            else "\n\n"
        )
        final_examples = _concatenate_examples(
            trimmed_examples, concat_size=calibration_dataset_concat_size, separator=separator
        )
    else:
        final_examples = trimmed_examples

    sorted_examples = _sort_examples(final_examples, sort_mode=calibration_dataset_sort)

    return _batch_examples(sorted_examples, batch_size=batch_size)
