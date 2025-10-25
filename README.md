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