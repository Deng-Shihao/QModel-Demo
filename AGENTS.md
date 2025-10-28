# Repository Guidelines

## Project Structure & Module Organization
The core Python package lives in `nanomodel/` with subpackages for model loaders (`models/`), quantization logic (`quantization/`), runtime processors (`processors/`), and performance-critical modules (`nn_modules/`). CUDA and C++ kernels reside in `nanomodel_ext/`, and they are compiled when the wheel is built or when `BUILD_CUDA_EXT=1` is set. Example entry points live both in the root (e.g., `gptq_example.py`, `chat_example.py`) and under `example/` for end-to-end calibration and Hugging Face workflows.

## Build, Test, and Development Commands
- `uv venv && source .venv/bin/activate`: create and activate a local virtual environment.
- `uv pip install -e .[test,quality]`: install the project in editable mode with linting and test extras.
- `BUILD_CUDA_EXT=1 pip install -v . --no-build-isolation`: force a rebuild of the CUDA extensions against local toolchains.
- `python example/basic_usage.py --model meta-llama/Llama-3-8B`: run a reference quantization + inference pipeline.
- `pytest`: execute the test suite (add `-k <pattern>` to target specific scenarios).
- `ruff check nanomodel`: run the configured formatter/linter for quick hygiene checks.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and type hints where they clarify tensor contracts. Modules and functions use `snake_case`, classes use `PascalCase`, and constants use `UPPER_SNAKE`. Document public APIs briefly, prefer logging via `nanomodel.utils.logger`, run `ruff check --fix`, and keep lines under 99 characters.

## Testing Guidelines
Install test dependencies via `uv pip install -e .[test]`, place new tests under `tests/`, and name files `test_<feature>.py`. Use `pytest` fixtures to stub GPU-heavy paths, mark long-running quantization checks with `@pytest.mark.cuda` or `@pytest.mark.slow`, and confirm `python wikitext2_example.py --max-seq-len 128` still runs on CPU.

## Commit & Pull Request Guidelines
Git history favors concise, imperative messages (`fix setup.py`, `act_order`). Mirror that style with one short line under 72 characters. Pull requests should include a summary, reproduction commands, affected hardware targets (e.g., CUDA 12.0, SM 89), and screenshots or logs when behavior changes. Link issues and highlight configuration updates.

## Security & Configuration Notes
Do not commit model weights, API tokens, or calibration datasets. Set `TORCH_CUDA_ARCH_LIST` to the minimal required SMs (for example, `export TORCH_CUDA_ARCH_LIST="8.9"`), keep secrets in ignored `.env` files, and point `HF_HOME` to a path that remains outside version control when downloading models.
