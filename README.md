# NanoModel

## Available NOW !!!

## Installation

```shell
uv venv # create venv

uv pip install --upgrade pip setuptools wheel

# CUDA 12.0 Only for  8.9|9.0 (option)
export TORCH_CUDA_ARCH_LIST="8.9" # setting computer capability 8.9 (nvcc)

pip install -v . --no-build-isolation

# Force CUDA compilation CUDA extensions
# export BUILD_CUDA_EXT=1
# BUILD_CUDA_EXT=1 pip install -v . --no-build-isolation

# Build .whl bdist_whee
python setup.py bdist_wheel
# dist/nanomodel-0.1.0+cu121torch2.5.1-cp310-cp310-linux_x86_64.whl

# Debug for dev
uv pip install -e .
```

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

# Group Aware Reordering (GAR)
@article{gar,
  title={Dual Precision Quantization for Efficient and Accurate Deep Neural Networks Inference, CVPRW 2025.},
  author={T. Gafni, A. Karnieli, Y. Hanani},
  journal={arXiv preprint arXiv:2505.14638},
  year={2025}
}

# GPTQ Marlin Kernel
@article{frantar2024marlin,
  title={MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models},
  author={Frantar, Elias and Castro, Roberto L and Chen, Jiale and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2408.11743},
  year={2024}
}
```

### update
gptq.py

Documented and cleaned workspace/cache helpers to clarify device selection and resizing logic, including docstrings for _device_cache_key, _lease_workspace, and related methods (nanomodel/quantization/gptq.py (lines 22-205)).
Added _quantize_block_vectorized and reused it across the mock/non-grouped paths to remove duplicated clamp/round code while preserving semantics (nanomodel/quantization/gptq.py (lines 233-267), nanomodel/quantization/gptq.py (lines 748-764)).
Refactored process_batch to delegate reshaping to _reshape_inputs_for_hessian, tightened OOM handling, and added targeted logging/docstrings for easier reasoning about Hessian staging (nanomodel/quantization/gptq.py (lines 269-460)).
Clarified quantization flow with docstrings, logging cleanups, and explanatory comments around activation ordering, static groups, and Hessian damping (nanomodel/quantization/gptq.py (lines 556-845)).
Verified syntax with python3 - <<'PY' ... ast.parse(...) ... (py_compile attempted but blocked by sandbox cache permissions).

Next steps: 1) Run ruff check nanomodel/quantization/gptq.py to keep formatting consistent. 2) Execute the existing quantization tests/recipes (e.g., pytest -k gptq or python example/basic_usage.py) to ensure runtime behavior still matches expectations.


3) Group-Aware Reordering (GAR): To further mitigate accuracy degradation, we introduce a weight quantization strategy that prioritizes important weights without inference overhead. Notably, GAR is not limited to our specific setting; it is applicable to other quantization schemes, making it a versatile technique for improving the accuracy of quantized models without introducing additional overhead. Ablation studies further confirm its effectiveness.
