from __future__ import annotations

import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union


import numpy
import torch
from huggingface_hub import list_repo_files
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ..quantization import METHOD, QUANT_CONFIG_FILENAME
from ..utils import BACKEND
from ..utils.eval import EVAL
from ..utils.logger import setup_logger
from .base import BaseNanoModel, QuantizeConfig

from .definitions.llama import LlamaNanoModel
from .definitions.qwen2 import Qwen2NanoModel
from .definitions.qwen3 import Qwen3NanoModel

__all__ = ["AutoNanoModel"]

log = setup_logger()

_DEFAULT_SEED = 233
_QUANT_CONFIG_CANDIDATES = (QUANT_CONFIG_FILENAME, "quant_config.json")


def _ensure_runtime_env() -> None:
    """Populate environment defaults required for consistent runtime behavior."""
    if os.environ.get("PYTORCH_ALLOC_CONF") is None:
        os.environ["PYTORCH_ALLOC_CONF"] = (
            "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7"
        )

    if os.environ.get("CUDA_DEVICE_ORDER") is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if "CUDA_VISIBLE_DEVICES" in os.environ and "ROCR_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)


def _enable_mps_fallback_if_needed() -> None:
    """Enable MPS fallback on macOS to cover missing aten ops."""
    if sys.platform != "darwin":
        return

    fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback_env and fallback_env.lower() in {"1", "true", "yes", "on"}:
        return

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    log.info(
        "ENV: Auto enabling PYTORCH_ENABLE_MPS_FALLBACK=1 to cover missing MPS aten ops."
    )


def _seed_rng(seed: int = _DEFAULT_SEED) -> None:
    """Seed RNGs to keep quantization and inference deterministic."""
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def _resolve_backend(backend: Union[str, BACKEND]) -> BACKEND:
    """Normalize backend values to the enum."""
    return backend if isinstance(backend, BACKEND) else BACKEND(backend)


@lru_cache(maxsize=None)
def _list_remote_files(repo_id: str) -> set[str]:
    """Cache huggingface repo file listings to limit API calls."""
    return set(list_repo_files(repo_id=repo_id))


def _has_local_quant_config(model_path: Path) -> bool:
    return any((model_path / name).exists() for name in _QUANT_CONFIG_CANDIDATES)


def _has_remote_quant_config(model_id: str) -> bool:
    files = _list_remote_files(repo_id=model_id)
    return any(name in files for name in _QUANT_CONFIG_CANDIDATES)


def _is_model_quantized(model_id_or_path: str, *, config: AutoConfig) -> bool:
    """Determine whether the model has been quantized by NanoModel-compatible tooling."""
    quant_cfg = getattr(config, "quantization_config", None)
    if quant_cfg and isinstance(quant_cfg, dict):
        quant_format = quant_cfg.get("quant_format", "").lower()
        return quant_format in {METHOD.GPTQ, METHOD.AWQ}

    model_path = Path(model_id_or_path)
    if model_path.is_dir():
        return _has_local_quant_config(model_path)

    try:
        return _has_remote_quant_config(model_id_or_path)
    except Exception:  # pragma: no cover - networking edge cases
        log.debug(
            "Falling back to AutoConfig quantization detection for %s build; "
            "list_repo_files lookup failed.",
            model_id_or_path,
        )
        return False


def check_and_get_model_type(
    model_dir: str,
    trust_remote_code: bool = False,
    *,
    config: Optional[AutoConfig] = None,
) -> str:
    """Validate model type support and return a normalized key."""
    cfg = config or AutoConfig.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code
    )
    model_type = cfg.model_type.lower()
    if model_type not in MODEL_DICT:
        raise TypeError(f"{cfg.model_type} isn't supported yet.")
    return model_type


_ensure_runtime_env()
_enable_mps_fallback_if_needed()
_seed_rng()

MODEL_DICT = {
    "llama": LlamaNanoModel,
    "qwen2": Qwen2NanoModel,
    "qwen3": Qwen3NanoModel,
}


class AutoNanoModel:
    """Factory helpers that pick the correct NanoModel implementation for a given checkpoint."""

    def __init__(self):
        raise EnvironmentError(
            "NanoModel is not designed to be instantiated\n"
            "use `NanoModel.from_pretrained` to load pretrained model and prepare for quantization via `.quantize()`.\n"
            "use `NanoModel.from_quantized` to inference with post-quantized model."
        )

    @classmethod
    def load(
        cls,
        model_id_or_path: Optional[str],
        quantize_config: Optional[QuantizeConfig | Dict] = None,
        device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        backend: Union[str, BACKEND] = BACKEND.AUTO,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """Automatically choose between quantized and pre-quantized loading."""
        if not model_id_or_path:
            raise ValueError("`model_id_or_path` is required to load a model.")

        if isinstance(model_id_or_path, str):
            model_id_or_path = model_id_or_path.strip()
            if not model_id_or_path:
                raise ValueError("`model_id_or_path` cannot be an empty string.")

        # normalize config to cfg instance
        if isinstance(quantize_config, Dict):
            quantize_config = QuantizeConfig(**quantize_config)

        backend_enum = _resolve_backend(backend)

        model_cfg = AutoConfig.from_pretrained(
            model_id_or_path, trust_remote_code=trust_remote_code
        )
        is_model_quantized = _is_model_quantized(
            model_id_or_path=model_id_or_path,
            config=model_cfg,
        )

        if is_model_quantized:
            model = cls.from_quantized(
                model_id_or_path=model_id_or_path,
                device_map=device_map,
                device=device,
                backend=backend_enum,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        else:
            model = cls.from_pretrained(
                model_id_or_path=model_id_or_path,
                quantize_config=quantize_config,
                device_map=device_map,
                device=device,
                trust_remote_code=trust_remote_code,
                backend=backend_enum,
                **kwargs,
            )

        return model

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        quantize_config: Optional[QuantizeConfig] = None,
        trust_remote_code: bool = False,
        **model_init_kwargs,
    ) -> BaseNanoModel:
        """Load a pretrained model and prepare it for quantization."""
        backend_enum = _resolve_backend(model_init_kwargs.pop("backend", BACKEND.AUTO))

        cfg = AutoConfig.from_pretrained(
            model_id_or_path, trust_remote_code=trust_remote_code
        )

        if getattr(cfg, "quantization_config", None):
            log.warning(
                "As the model is already quantized, loading will be done via from_quantized.\n"
                "To apply quantization yourself, provide an unquantized model path or ID, and load it using from_pretrained with quantize_config."
            )
            fallback_kwargs = dict(model_init_kwargs)
            device_map = fallback_kwargs.pop("device_map", None)
            device = fallback_kwargs.pop("device", None)
            return cls.from_quantized(
                model_id_or_path=model_id_or_path,
                device_map=device_map,
                device=device,
                backend=backend_enum,
                trust_remote_code=trust_remote_code,
                **fallback_kwargs,
            )

        if quantize_config is None:
            raise ValueError(
                "QuantizeConfig must be provided when loading from pretrained weights."
            )

        if quantize_config and quantize_config.dynamic:
            log.warning(
                "Full support for NanoModel’s per-module dynamic quantization is now included "
                "in the latest vLLM and SGLang, but hf transformers have not yet added this capability."
            )

        model_type = check_and_get_model_type(
            model_id_or_path, trust_remote_code, config=cfg
        )
        return MODEL_DICT[model_type].from_pretrained(
            pretrained_model_id_or_path=model_id_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
        cls,
        model_id_or_path: Optional[str],
        device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
        device: Optional[Union[str, int]] = None,
        backend: Union[str, BACKEND] = BACKEND.AUTO,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> BaseNanoModel:
        """Load a quantized NanoModel checkpoint."""
        cfg = AutoConfig.from_pretrained(
            model_id_or_path, trust_remote_code=trust_remote_code
        )
        model_type = check_and_get_model_type(
            model_id_or_path, trust_remote_code, config=cfg
        )

        return MODEL_DICT[model_type].from_quantized(
            model_id_or_path=model_id_or_path,
            device_map=device_map,
            device=device,
            backend=_resolve_backend(backend),
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def eval(
        cls,
        model_or_id_or_path: str = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        tasks: Union[
            EVAL.LM_EVAL,
            EVAL.EVALPLUS,
            List[EVAL.LM_EVAL],
            List[EVAL.EVALPLUS],
            EVAL.MMLU_PRO,
            List[EVAL.MMLU_PRO],
        ] = None,  # set to None to fix mutable warning
        framework: Union[
            Type[EVAL.LM_EVAL], Type[EVAL.EVALPLUS], Type[EVAL.MMLU_PRO]
        ] = EVAL.LM_EVAL,
        batch_size: Union[int, str] = 1,
        trust_remote_code: bool = False,
        output_path: Optional[str] = None,
        llm_backend: str = "nanomodel",
        backend: BACKEND = BACKEND.AUTO,  # nanomodel arg only
        random_seed: int = 1234,  # only for framework=EVAL.LM_EVAL backend=vllm
        model_args: Dict[
            str, Any
        ] = None,  # only for framework=EVAL.LM_EVAL backend=vllm
        ntrain: int = 1,  # only for framework=EVAL.MMLUPRO
        **args,
    ):
        from peft import PeftModel

        if model_args is None:
            model_args = {}
        if tasks is None:
            if framework == EVAL.LM_EVAL:
                tasks = [EVAL.LM_EVAL.ARC_CHALLENGE]
            elif framework == EVAL.MMLU_PRO:
                tasks = [EVAL.MMLU_PRO.MATH]
            else:
                tasks = [EVAL.EVALPLUS.HUMAN]

        elif not isinstance(tasks, List):
            tasks = [tasks]

        if framework is None:
            raise ValueError("Eval parameter: `framework` cannot be set to None")

        if not isinstance(tasks, list):
            raise ValueError("Eval parameter: `tasks` must be of List type")

        if llm_backend not in ["nanomodel", "vllm"]:
            raise ValueError("Eval framework support llm_backend: [nanomodel, vllm]")

        if llm_backend == "vllm":
            if "tensor_parallel_size" not in model_args:
                try:
                    cuda_devices = (
                        torch.cuda.device_count() if torch.cuda.is_available() else 0
                    )
                except Exception:
                    cuda_devices = 0
                if cuda_devices:
                    model_args["tensor_parallel_size"] = cuda_devices
            if "gpu_memory_utilization" not in model_args:
                model_args["gpu_memory_utilization"] = 0.90

        if isinstance(model_or_id_or_path, str):
            load_backend = backend
            load_kwargs = {}

            if llm_backend == "vllm":
                disallowed_keys = {
                    "pretrained",
                    "tokenizer",
                    "nanomodel",
                    "trust_remote_code",
                    "backend",
                    "model_id_or_path",
                }
                load_kwargs = {
                    k: v for k, v in model_args.items() if k not in disallowed_keys
                }

            backend_name = (
                load_backend.value
                if isinstance(load_backend, BACKEND)
                else str(load_backend)
            )
            log.info(f"Eval: loading using backend = `{backend_name}`")
            model = AutoNanoModel.load(
                model_id_or_path=model_or_id_or_path,
                backend=load_backend,
                trust_remote_code=trust_remote_code,
                **load_kwargs,
            )
            model_id_or_path = model_or_id_or_path
        elif isinstance(model_or_id_or_path, BaseNanoModel) or isinstance(
            model_or_id_or_path, (PreTrainedModel, PeftModel)
        ):
            model = model_or_id_or_path
            model_id_or_path = model.config.name_or_path  #
        else:
            raise ValueError(
                f"`model_or_id_or_path` is invalid. expected: `model instance or str` actual: `{model_or_id_or_path}`"
            )

        if tokenizer is None:
            if isinstance(model, BaseNanoModel):
                tokenizer = model.tokenizer
            elif isinstance(model, PreTrainedModel):
                tokenizer = AutoTokenizer.from_pretrained(
                    model.config.name_or_path,
                    trust_remote_code=trust_remote_code,
                )
            elif isinstance(model_id_or_path, str) and model_id_or_path.strip():
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id_or_path.strip(),
                    trust_remote_code=trust_remote_code,
                )

        if tokenizer is None:
            raise ValueError(
                "Tokenizer: Auto-loading of tokenizer failed with `model_or_id_or_path`. Please pass in `tokenizer` as argument."
            )

        if llm_backend == "nanomodel":  # vllm loads tokenizer
            model_args["tokenizer"] = tokenizer

        if framework == EVAL.LM_EVAL:
            from lm_eval.utils import make_table  # hack: circular import

            for task in tasks:
                if task not in EVAL.get_task_enums():
                    raise ValueError(
                        f"Eval.lm_eval supported `tasks`: `{EVAL.get_all_tasks_string()}`, actual = `{task}`"
                    )

            model_name = "hf" if llm_backend == "nanomodel" else llm_backend

            if llm_backend == "nanomodel":
                model_args["nanomodel"] = True
            model_args["pretrained"] = model_id_or_path

            try:
                from lm_eval import simple_evaluate
                from lm_eval.models.huggingface import HFLM
            except BaseException:
                raise ValueError(
                    "lm_eval is not installed. Please install via `pip install nanomodel[eval]`."
                )

            if llm_backend == "nanomodel" and model is not None:
                model_name = HFLM(
                    pretrained=model,
                    batch_size=batch_size,
                    trust_remote_code=trust_remote_code,
                )

            gen_kwargs = args.pop("gen_kwargs", None)

            # use model.generation_config whenever possible
            if gen_kwargs is None:
                # TODO: move to utils
                if hasattr(model, "generation_config") and isinstance(
                    model.generation_config, GenerationConfig
                ):
                    gen_dict = {
                        "do_sample": model.generation_config.do_sample,
                        "temperature": model.generation_config.temperature,
                        "top_k": model.generation_config.top_k,
                        "top_p": model.generation_config.top_p,
                        "min_p": model.generation_config.min_p,
                    }
                    gen_kwargs = ",".join(
                        f"{key}={value}"
                        for key, value in gen_dict.items()
                        if value not in ["", {}, None, []]
                    )
                else:
                    gen_kwargs = "temperature=0.0,top_k=50"  # default

            log.info(f"LM-EVAL: `gen_kwargs` = `{gen_kwargs}`")

            # lm-eval has very low scores if apply_chat_template is enabled
            apply_chat_template = args.pop(
                "apply_chat_template", False
            )  # args.pop("apply_chat_template", True if tokenizer.chat_template is not None else False)
            log.info(f"LM-EVAL: `apply_chat_template` = `{apply_chat_template}`")

            results = simple_evaluate(
                model=model_name,
                model_args=model_args,
                tasks=[task.value for task in tasks],
                batch_size=batch_size,
                apply_chat_template=apply_chat_template,
                gen_kwargs=gen_kwargs,
                random_seed=random_seed,
                numpy_random_seed=random_seed,
                torch_random_seed=random_seed,
                fewshot_random_seed=random_seed,
                **args,
            )

            if results is None:
                raise ValueError("lm_eval run fail, check your code!!!")

            print("--------lm_eval Eval Result---------")
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))
            print("--------lm_eval Result End---------")
            return results
        elif framework == EVAL.EVALPLUS:
            for task in tasks:
                if task not in EVAL.get_task_enums():
                    raise ValueError(
                        f"evalplus support tasks: {EVAL.get_all_tasks_string()}"
                    )
            from ..utils.eval import evalplus, evalplus_make_table

            results = {}
            for task in tasks:
                base_formatted, plus_formatted, result_path = evalplus(
                    model=model_id_or_path,
                    dataset=task.value,
                    batch=batch_size,
                    trust_remote_code=trust_remote_code,
                    output_file=output_path,
                    backend=llm_backend,
                )
                results[task.value] = {
                    "base tests": base_formatted,
                    "base + extra tests": plus_formatted,
                    "results_path": result_path,
                }
            print("--------evalplus Eval Result---------")
            evalplus_make_table(results)
            print("--------evalplus Result End---------")
            return results
        elif framework == EVAL.MMLU_PRO:
            for task in tasks:
                if task not in EVAL.get_task_enums():
                    raise ValueError(
                        f"eval support tasks: {EVAL.get_all_tasks_string()}"
                    )
            from ..utils.mmlupro import mmlupro

            selected_subjects = ",".join(tasks)
            results = mmlupro(
                model,
                tokenizer,
                save_dir=output_path,
                seed=random_seed,
                selected_subjects=selected_subjects,
                ntrain=ntrain,
                batch_size=batch_size,
            )

            print("--------MMLUPro Eval Result---------")
            print(results)
            print("--------MMLUPro Result End---------")
            return results
        else:
            raise ValueError(
                "Eval framework support: EVAL.LM_EVAL, EVAL.EVALPLUS, EVAL.MMLUPRO"
            )
