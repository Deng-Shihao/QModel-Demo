# NanoModel

NanoModel is an opinionated quantization toolkit for large language models. It wraps the research-grade GPTQ/AWQ/QQQ kernels behind a lightweight Python API so you can quantize, package, and serve transformer models without juggling device placement, calibration loops, or backend-specific details.

- Works directly with Hugging Face `transformers` configs and checkpoints.
- Autodetects whether a checkpoint is already quantized and switches between `from_pretrained` and `from_quantized`.
- Ships optimized CUDA, CPU, and MLX kernels (built on demand) with unified abstractions for GPTQ, AWQ, and Marlin.
- Provides fit-for-purpose utilities such as perplexity evaluation, background device thread pools, and Hugging Face Hub helpers.
- Supports LLaMA, Qwen2, and Qwen3 families out of the box with a modular registry for adding new architectures.

## Installation

### 1. Install PyTorch

Install a PyTorch build that matches your accelerator and Python version. Follow the official instructions at <https://pytorch.org/get-started/locally/>. NanoModel targets PyTorch 2.1+ and benefits from 2.2+ for the latest quant kernels.

### 2. Install NanoModel

```bash
git clone <repository-url>
cd NanoModel
python -m pip install --upgrade pip setuptools wheel
pip install -v gptqmodel --no-build-isolation
```

#### Controlling extension builds

- `BUILD_CUDA_EXT=0 python -m pip install .` skips compiling GPU kernels (useful for CPU-only or constrained CI).
- `CUDA_ARCH_LIST="8.0;8.6;9.0"` overrides the detected compute capabilities when you need to cross-compile wheels.

#### Optional extras

```bash
python -m pip install ".[vllm]"     # vLLM runtime integration
python -m pip install ".[sglang]"   # SGLang runtime integration
python -m pip install ".[eval]"     # lm-eval and EvalPlus helpers
python -m pip install ".[mlx]"      # Apple MLX kernels for Apple Silicon
```

## Quickstart

The `AutoNanoModel` helper decides whether to quantize or simply load an existing quantized checkpoint.

```python
from transformers import AutoTokenizer
from nanomodel import AutoNanoModel, QuantizeConfig, get_best_device
from nanomodel.quantization import FORMAT, METHOD

model_id = "Qwen/Qwen3-0.6B"
quantized_dir = "qwen3-0.6B-gptq"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# Prepare calibration data (list of tokenized samples that provide representative activations).
calibration_dataset = [
    tokenizer("NanoModel packs GPTQ-style quantization behind familiar APIs.")
]

quantize_config = QuantizeConfig(
    bits=4,
    group_size=128,
    quant_method=METHOD.GPTQ,  # switch to METHOD.AWQ or METHOD.QQQ as needed
    format=FORMAT.GPTQ,        # FORMAT.MARLIN / FORMAT.GEMM / FORMAT.GEMV also available
)

# 1. Load a full-precision checkpoint and quantize it in-place.
model = AutoNanoModel.load(model_id, quantize_config)
model.quantize(calibration_dataset)
model.save(quantized_dir)          # produces safetensors + quantize_config.json

# 2. Load the quantized model back on the best available device.
device = get_best_device()
model = AutoNanoModel.load(quantized_dir, device=device)

prompt = tokenizer("NanoModel is", return_tensors="pt").to(model.device)
output = model.generate(**prompt, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Running the script above yields a 4-bit GPTQ checkpoint that can be re-used in Hugging Face Transformers, NanoModel runtimes, or exported to third-party inference engines.

### Using Hugging Face `quantization_config`

NanoModel writes metadata that Hugging Face Transformers understands. You can skip the `AutoNanoModel.load(..., quantize_config=...)` call and instead let Transformers do the heavy lifting:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = ["NanoModel exposes quantization with minimal ceremony."]
gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)

quantized = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    quantization_config=gptq_config,
)
quantized.save_pretrained("./qwen3-0.6B-gptq")
tokenizer.save_pretrained("./qwen3-0.6B-gptq")
```

## Configuration Highlights

- **Precision & grouping**: `QuantizeConfig.bits` supports 2/3/4/8-bit flows with `group_size` control (128 is the recommended speed/quality trade-off).
- **Dynamic overrides**: Provide `QuantizeConfig.dynamic` with regex-based module overrides to tune bits, enable/disable quantization, or adjust LoRA-specific knobs for subsets of layers.
- **Quantization methods**: Select GPTQ, AWQ, or QQQ; NanoModel maps the right kernels and checkpoint format automatically.
- **Formats & kernels**: `format=FORMAT.MARLIN` unlocks fused Marlin kernels, while AWQ adds `FORMAT.GEMM`, `FORMAT.GEMV`, and `FORMAT.GEMV_FAST`. Use `backend=BACKEND.TRITON`, `BACKEND.MARLIN`, or `BACKEND.VLLM` (via `from nanomodel.utils.backend import BACKEND`) when loading to route inference to specialized engines.
- **Offloading**: `offload_to_disk=True` drastically reduces CPU RAM during calibration by spilling intermediate tensors to disk.
- **Evaluation**: `nanomodel.utils.perplexity.Perplexity` helps track perplexity regressions on datasets such as WikiText2.

## Examples

The `example/` directory contains runnable scripts:

- `example/basic_usage.py` – end-to-end quantization and generation loop.
- `example/basic_usage_wikitext2.py` – quantize with WikiText2 calibration and compute average perplexity.
- `example/transformers_usage.py` – integrate NanoModel metadata with pure Hugging Face Transformers workflows.

Invoke them with `python example/basic_usage.py` after installing the project (optionally set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `PYTORCH_ALLOC_CONF=expandable_segments:True` for more predictable GPU usage).

## Performance Tips

- Export `PYTORCH_ENABLE_MPS_FALLBACK=1` on macOS to cover missing MPS kernels automatically.
- Use `PYTHON_GIL=0` (Python 3.13t+) for multi-GPU quantization pipelines; NanoModel detects the free-threading runtime and adjusts worker pools.
- `DEVICE_THREAD_POOL` handles background warm-up for CUDA, XPU, MPS, and CPU. Call `nanomodel.DEVICE_THREAD_POOL.sync()` before measuring latency-critical sections if you want deterministic timing.
- Check VRAM usage via `nanomodel.utils.device.get_gpu_usage_memory()` while experimenting with group sizes or mixed backends.

## Contributing

NanoModel is evolving quickly. Contributions are welcome—particularly new model definitions, quantization kernels, and recipes for emerging inference backends. Open an issue or PR with:

1. A concise description of the change.
2. Reproduction steps or scripts (preferably added to `example/`).
3. Any profiling data, accuracy deltas, or environment notes that help reviewers validate the patch.

## Project Status

The public interface (`AutoNanoModel`, `QuantizeConfig`, `get_best_device`, and the model registry under `nanomodel.models`) is considered alpha. Expect API consolidation as we broaden hardware coverage and finalize the rename from the legacy GPTQModel codebase. Pin to a specific release if you ship production workloads.

## Citation
```shell
# GPTQ
@article{frantar-gptq,
  title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers}, 
  author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  journal={arXiv preprint arXiv:2210.17323},
  year={2022}  
}

# AWQ
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}

# GPTQ Marlin Kernel
@article{frantar2024marlin,
  title={MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models},
  author={Frantar, Elias and Castro, Roberto L and Chen, Jiale and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2408.11743},
  year={2024}
}
```