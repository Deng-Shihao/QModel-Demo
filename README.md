- A simple-to-use toolkit for model quantification & deployment

TODO:

- LoRA - Recover accuracy
- Benchmark
- …

## Usage

### Install

---

- uv

```bash
uv venv # Create a virtual env for project
uv pip install --upgrade pip setuptools wheel

# Install the corresponding PyTorch with CUDA
# Detail: https://pytorch.org/get-started/locally/
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt

uv pip install -v . --no-build-isolation # Build locally
```

### Quantize with NanoModel

---

```python
from datasets import load_dataset
from nanomodel import AutoNanoModel, QuantizeConfig
from nanomodel.quantization import KERNEL, METHOD # For AWQ quantization

pretrained_model_id = "Qwen/Qwen3-4B-Instruct-2507"
quantized_model_id = "/home/sd24191/git_project/QModel-Demo/quantized_models/Qwen3-4B-Instruct-2507-GPTQ-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))["text"]

## Default gptq algorithm
quant_config = QuantizeConfig(bits=4, group_size=128)

## Awq algorithm
# quant_config = QuantizeConfig(
#        bits=4,
#        group_size=128,
#        quant_method=METHOD.AWQ,  # Switch to METHOD.GPTQ if you prefer GPTQ kernels.
#        kernel=KERNEL.GEMM,  # Alternative kernels: KERNEL.GEMM for matmul-based inference.
# )

# Increase the `batch_size` to match GPU/VRAM specifications, thereby accelerating quantisation speed.
# model.quantize(calibration_dataset, batch_size=2)
model = AutoNanoModel.load(model_id, quant_config)
model.save(quant_path)
```

### Inference

---

- Basic

```python
from nanomodel import AutoNanoModel

model = nanomodel.load("/home/sd24191/git_project/QModel-Demo/quantized_models/Qwen3-4B-Instruct-2507-GPTQ-4bit")
result = model.generate("LLMs is ")[0] # tokens

print(model.tokenizer.decode(result)) # string output
```

### Eval

---

```python
from nanomodel import AutoNanoModel
from nanomodel.utils.eval import EVAL

model_id = "/home/sd24191/git_project/QModel-Demo/quantized_models/Qwen3-4B-Instruct-2507-GPTQ-4bit"
# model_id = "/home/sd24191/git_project/QModel-Demo/quantized_models/Qwen3-4B-Instruct-2507-AWQ-4bit"

# `lm-eval` framework evaluate the model
lm_eval_data = AutoNanoModel.eval(
    model_id, framework=EVAL.LM_EVAL, tasks=[EVAL.LM_EVAL.ARC_CHALLENGE]
)

# evalplus framework evaluate the model
evalplus_data = AutoNanoModel.eval(
    model_id, framework=EVAL.EVALPLUS, tasks=[EVAL.EVALPLUS.HUMAN]
)
```

### Quantization Method & Feature

---

| Method | Inference Kernel |
| --- | --- |
| GPTQ ✅ | Marlin, Triton, Torch |
| AWQ ✅ | GEMM, Marlin |

### Citation

```python
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