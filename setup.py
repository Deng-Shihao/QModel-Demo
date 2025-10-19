import os
import subprocess
import sys
from pathlib import Path

import re # pcre
from setuptools import find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


# ---------------------------
# Helpers (no torch required)
# ---------------------------

def _read_env(name, default=None):
    v = os.environ.get(name)
    return v if (v is not None and str(v).strip() != "") else default


def _probe_cmd(args, timeout=6):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except Exception:
        return None


def _bool_env(name, default=False):
    v = _read_env(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def _detect_rocm_version():
    v = _read_env("ROCM_VERSION")
    if v:
        return v
    hip = _probe_cmd(["hipcc", "--version"])
    if hip:
        m = re.search(r"\b([0-9]+\.[0-9]+)\b", hip)
        if m:
            return m.group(1)
    try:
        p = Path("/opt/rocm/.info/version")
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None


def _detect_cuda_arch_list():
    """Return TORCH_CUDA_ARCH_LIST style string for the *installed* GPUs only.
    Priority:
      1) CUDA_ARCH_LIST env override (verbatim)
      2) nvidia-smi compute_cap (actual devices)
    """
    env_arch = _read_env("CUDA_ARCH_LIST")
    if env_arch:
        return env_arch

    smi_out = _probe_cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if smi_out:
        caps = []
        for line in smi_out.splitlines():
            cap = line.strip()
            if not cap:
                continue
            try:
                major, minor = cap.split(".", 1)
                caps.append(f"{int(major)}.{int(minor)}")
            except Exception:
                if cap.isdigit():
                    caps.append(f"{cap}.0")
        caps = sorted(set(caps), key=lambda x: (int(x.split(".")[0]), int(x.split(".")[1])))
        if caps:
            return ";".join(caps)

    raise Exception("Could not get compute capability from nvidia-smi. Please check nvidia-utils package is installed.")


def _parse_arch_list(s: str):
    return [tok for tok in re.split(r"[;\s,]+", s) if tok.strip()]


def _has_cuda_v8_from_arch_list(arch_list):
    try:
        vals = []
        for a in arch_list:
            base = a.split("+", 1)[0]
            vals.append(float(base))
        return any(v >= 8.0 for v in vals)
    except Exception:
        return False


def _detect_cxx11_abi():
    v = _read_env("CXX11_ABI")
    if v in ("0", "1"):
        return int(v)
    return 1


def _detect_torch_version() -> str:
    out = _probe_cmd(["uv", "pip", "show", "torch"])
    if out:
        m = re.search(r"^Version:\s*([^\s]+)\s*$", out, flags=re.MULTILINE)
        if m:
            return m.group(1)

    for cmd in (["pip", "show", "torch"], [sys.executable, "-m", "pip", "show", "torch"]):
        out = _probe_cmd(cmd)
        if out:
            m = re.search(r"^Version:\s*([^\s]+)\s*$", out, flags=re.MULTILINE)
            if m:
                return m.group(1)

    out = _probe_cmd(["conda", "list", "torch"])
    if out:
        for line in out.splitlines():
            if line.strip().startswith("torch"):
                parts = re.split(r"\s+", line.strip())
                if len(parts) >= 2 and re.match(r"^\d+\.\d+(\.\d+)?", parts[1]):
                    return parts[1]

    try:
        import importlib.metadata as im
        version = im.version("torch")
        if version:
            return version
    except Exception:
        pass

    raise Exception("Unable to detect torch version via uv/pip/conda/importlib. Please install torch >= 2.7.1")


def _major_minor(v: str) -> str:
    parts = v.split(".")
    return ".".join(parts[:2]) if parts else v


def _version_geq(version: str | None, major: int, minor: int = 0) -> bool:
    if not version:
        return False
    try:
        parts = re.split(r"[._-]", version)
        ver_major = int(parts[0]) if parts else 0
        ver_minor = int(parts[1]) if len(parts) > 1 else 0
        return (ver_major, ver_minor) >= (major, minor)
    except Exception:
        return False


def _nvcc_release_version() -> str | None:
    out = _probe_cmd(["nvcc", "--version"])
    if not out:
        print("NVCC not found: For Ubuntu, run `sudo update-alternatives --config cuda` to fix path.")
        return None
    match = re.search(r"release\s+(\d+)\.(\d+)", out)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return None


def _detect_cuda_version() -> str | None:
    v = os.environ.get("CUDA_VERSION")
    if v and v.strip():
        return v.strip()
    return _nvcc_release_version()


def _detect_nvcc_version() -> str | None:
    return _nvcc_release_version()


def get_version_tag() -> str:
    if BUILD_CUDA_EXT != "1":
        return "cpu"
    if ROCM_VERSION:
        return f"rocm{ROCM_VERSION}"
    if not CUDA_VERSION:
        raise Exception("Trying to compile for CUDA, but no CUDA/ROCm version detected.")
    torch_suffix = f"torch{_major_minor(TORCH_VERSION)}"
    CUDA_VERSION_COMPACT = "".join(CUDA_VERSION.split("."))
    base = f"cu{CUDA_VERSION_COMPACT[:3]}"
    return f"{base}{torch_suffix}"


# ---------------------------
# Env and versioning
# ---------------------------
TORCH_VERSION = _read_env("TORCH_VERSION")
RELEASE_MODE = _read_env("RELEASE_MODE")
CUDA_VERSION = _read_env("CUDA_VERSION")
ROCM_VERSION = _read_env("ROCM_VERSION")
TORCH_CUDA_ARCH_LIST = _read_env("TORCH_CUDA_ARCH_LIST")
NVCC_VERSION = _read_env("NVCC_VERSION")

if not TORCH_VERSION:
    TORCH_VERSION = _detect_torch_version()
if not CUDA_VERSION:
    CUDA_VERSION = _detect_cuda_version()
if not ROCM_VERSION:
    ROCM_VERSION = _detect_rocm_version()
if not NVCC_VERSION:
    NVCC_VERSION = _detect_nvcc_version()

SKIP_ROCM_VERSION_CHECK = _read_env("SKIP_ROCM_VERSION_CHECK")
FORCE_BUILD = _bool_env("NANOMODEL_FORCE_BUILD", False)

BUILD_CUDA_EXT = _read_env("BUILD_CUDA_EXT")
if BUILD_CUDA_EXT is None:
    BUILD_CUDA_EXT = "1" if (CUDA_VERSION or ROCM_VERSION) else "0"

if ROCM_VERSION and not SKIP_ROCM_VERSION_CHECK:
    try:
        if float(ROCM_VERSION) < 6.2:
            sys.exit(
                "NanoModel's compatibility with ROCm < 6.2 has not been verified. "
                "Set SKIP_ROCM_VERSION_CHECK=1 to proceed."
            )
    except Exception:
        pass

CUDA_ARCH_LIST = _detect_cuda_arch_list() if (BUILD_CUDA_EXT == "1" and not ROCM_VERSION) else None

if not TORCH_CUDA_ARCH_LIST and CUDA_ARCH_LIST:
    archs = _parse_arch_list(CUDA_ARCH_LIST)
    kept = []
    for arch in archs:
        try:
            base = arch.split("+", 1)[0]
            if float(base) >= 6.0:
                kept.append(arch)
            else:
                print(f"we do not support this compute arch: {arch}, skipped.")
        except Exception:
            kept.append(arch)

    TORCH_CUDA_ARCH_LIST = ";".join(kept)
    os.environ["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST

    print(f"CUDA_ARCH_LIST: {CUDA_ARCH_LIST}")
    print(f"TORCH_CUDA_ARCH_LIST: {TORCH_CUDA_ARCH_LIST}")

version_vars = {}
exec("exec(open('nanomodel/version.py').read()); version=__version__", {}, version_vars)
nanomodel_version = version_vars["version"]

# -----------------------------
# Prebuilt wheel download config
# -----------------------------
DEFAULT_WHEEL_URL_TEMPLATE = "https://github.com/ModelCloud/NanoModel/releases/download/{tag_name}/{wheel_name}"
WHEEL_URL_TEMPLATE = os.environ.get("NANOMODEL_WHEEL_URL_TEMPLATE")
WHEEL_BASE_URL = os.environ.get("NANOMODEL_WHEEL_BASE_URL") 
WHEEL_TAG = os.environ.get("NANOMODEL_WHEEL_TAG")          

def _resolve_wheel_url(tag_name: str, wheel_name: str) -> str | None:
    if WHEEL_URL_TEMPLATE:
        tmpl = WHEEL_URL_TEMPLATE
        if ("{wheel_name}" in tmpl) or ("{tag_name}" in tmpl):
            return tmpl.format(tag_name=tag_name, wheel_name=wheel_name)
        return (tmpl + wheel_name) if tmpl.endswith("/") else (tmpl + "/" + wheel_name)

    if WHEEL_BASE_URL:
        base = WHEEL_BASE_URL
        return (base + wheel_name) if base.endswith("/") else (base + "/" + wheel_name)

    return DEFAULT_WHEEL_URL_TEMPLATE.format(tag_name=tag_name, wheel_name=wheel_name)

# Decide HAS_CUDA_V8 without torch
HAS_CUDA_V8 = False
if CUDA_ARCH_LIST:
    HAS_CUDA_V8 = not ROCM_VERSION and _has_cuda_v8_from_arch_list(_parse_arch_list(CUDA_ARCH_LIST))
else:
    smi = _probe_cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if smi:
        try:
            caps = [float(x.strip()) for x in smi.splitlines() if x.strip()]
            HAS_CUDA_V8 = any(cap >= 8.0 for cap in caps)
        except Exception:
            HAS_CUDA_V8 = False

if RELEASE_MODE == "1":
    nanomodel_version = f"{nanomodel_version}+{get_version_tag()}"

include_dirs = ["nanomodel_ext"]

extensions = []
additional_setup_kwargs = {}

# ---------------------------
# Build CUDA/ROCm extensions (only when enabled)
# ---------------------------
def _env_enabled(val: str) -> bool:
    if val is None:
        return True
    return str(val).strip().lower() not in ("0", "false", "off", "no")


def _env_enabled_any(names, default="1") -> bool:
    for n in names:
        if n in os.environ:
            return _env_enabled(os.environ.get(n))
    return _env_enabled(default)


BUILD_MARLIN = _env_enabled_any(os.environ.get("NANOMODEL_BUILD_MARLIN", "1"))
BUILD_AWQ = _env_enabled(os.environ.get("NANOMODEL_BUILD_AWQ", "1"))

if BUILD_CUDA_EXT == "1":
    try:
        from torch.utils import cpp_extension as cpp_ext  # type: ignore
    except Exception:
        if FORCE_BUILD:
            sys.exit(
                "FORCE_BUILD is set but PyTorch C++ extension headers are unavailable. "
                "Install torch build deps first or unset NANOMODEL_FORCE_BUILD."
            )
        cpp_ext = None

    if cpp_ext is not None:
        extra_link_args = []
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17", "-DENABLE_BF16"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-DENABLE_BF16",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            ],
        }

        if sys.platform == "win32":
            extra_compile_args["cxx"] = ["/O2", "/std:c++17", "/openmp", "/DNDEBUG", "/DENABLE_BF16"]

        CXX11_ABI = _detect_cxx11_abi()
        extra_compile_args["cxx"] += [f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}"]
        extra_compile_args["nvcc"] += [f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}"]

        if not ROCM_VERSION:
            extra_compile_args["nvcc"] += [
                "-static-global-template-stub=false",
                "--threads", "8",
                "--optimize=3",
                "-Xptxas", "-v,-O3,-dlcm=ca",
                "-lineinfo",
                "-Xfatbin", "-compress-all",
                "-diag-suppress=179,39,177",
            ]
        else:
            def _hipify_compile_flags(flags):
                modified_flags = []
                for flag in flags:
                    if flag.startswith("-") and "CUDA" in flag and not flag.startswith("-I"):
                        parts = flag.split("=", 1)
                        if len(parts) == 2:
                            flag_part, value_part = parts
                            modified_flag_part = flag_part.replace("CUDA", "HIP", 1)
                            modified_flags.append(f"{modified_flag_part}={value_part}")
                        else:
                            modified_flags.append(flag.replace("CUDA", "HIP", 1))
                    else:
                        modified_flags.append(flag)
                return modified_flags
            extra_compile_args["nvcc"] = _hipify_compile_flags(extra_compile_args["nvcc"])

        if sys.platform != "win32":
            if not ROCM_VERSION and HAS_CUDA_V8:
                if BUILD_MARLIN:
                    marlin_kernel_dir = Path("nanomodel_ext/marlin")
                    marlin_kernel_files = sorted(marlin_kernel_dir.glob("kernel_*.cu"))

                    if not marlin_kernel_files:
                        generator_script = marlin_kernel_dir / "generate_kernels.py"
                        if generator_script.exists():
                            print("Regenerating marlin template instantiations for parallel compilation...")
                            subprocess.check_call([sys.executable, str(generator_script)])
                            marlin_kernel_files = sorted(marlin_kernel_dir.glob("kernel_*.cu"))

                    if not marlin_kernel_files:
                        raise RuntimeError(
                            "No generated marlin kernel templates detected. Run generate_kernels.py before building."
                        )

                    marlin_template_kernel_srcs = [str(path) for path in marlin_kernel_files]
                    extensions += [
                        cpp_ext.CUDAExtension(
                            "nanomodel_marlin_kernels",
                            [
                                "nanomodel_ext/marlin/marlin_cuda.cpp",
                                "nanomodel_ext/marlin/gptq_marlin.cu",
                                "nanomodel_ext/marlin/gptq_marlin_repack.cu",
                                "nanomodel_ext/marlin/awq_marlin_repack.cu",
                            ] + marlin_template_kernel_srcs,
                            extra_link_args=extra_link_args,
                            extra_compile_args=extra_compile_args,
                        )
                    ]

            if BUILD_AWQ:
                extensions += [
                    cpp_ext.CUDAExtension(
                        "nanomodel_awq_kernels",
                        [
                            "nanomodel_ext/awq/pybind_awq.cpp",
                            "nanomodel_ext/awq/quantization/gemm_cuda_gen.cu",
                            "nanomodel_ext/awq/quantization/gemv_cuda.cu",
                        ],
                        extra_link_args=extra_link_args,
                        extra_compile_args=extra_compile_args,
                    ),
                    cpp_ext.CUDAExtension(
                        "nanomodel_awq_v2_kernels",
                        [
                            "nanomodel_ext/awq/pybind_awq_v2.cpp",
                            "nanomodel_ext/awq/quantization_new/gemv/gemv_cuda.cu",
                            "nanomodel_ext/awq/quantization_new/gemm/gemm_cuda.cu",
                        ],
                        extra_link_args=extra_link_args,
                        extra_compile_args=extra_compile_args,
                    )
                ]

        additional_setup_kwargs = {
            "ext_modules": extensions,
            "cmdclass": {"build_ext": cpp_ext.BuildExtension},
        }

# ---------------------------
# Cached wheel fetcher
# ---------------------------

class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        xpu_avail = _bool_env("XPU_AVAILABLE", False)
        if FORCE_BUILD or xpu_avail:
            return super().run()

        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        wheel_filename = f"nanomodel-{nanomodel_version}+{get_version_tag()}-{python_version}-{python_version}-linux_x86_64.whl"

        tag_name = WHEEL_TAG if WHEEL_TAG else f"v{nanomodel_version}"
        wheel_url = _resolve_wheel_url(tag_name=tag_name, wheel_name=wheel_filename)

        print(f"Resolved wheel URL: {wheel_url}\nwheel name={wheel_filename}")

        try:
            import urllib.request as req
            req.urlretrieve(wheel_url, wheel_filename)

            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = (f"nanomodel-{nanomodel_version}-{impl_tag}-{abi_tag}-{plat_tag}")
            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            os.rename(wheel_filename, wheel_path)
        except BaseException:
            print(f"Precompiled wheel not found at: {wheel_url}. Building from source...")
            super().run()


# ---------------------------
# setup()
# ---------------------------
print(f"CUDA {CUDA_ARCH_LIST}")
print(f"HAS_CUDA_V8 {HAS_CUDA_V8}")
print(f"SETUP_KWARGS {additional_setup_kwargs}")
print(f"nanomodel_version={nanomodel_version}")

setup(
    name="nanomodel",
    version=nanomodel_version,
    packages=find_packages(),
    include_package_data=True,
    extras_require={
        "test": ["pytest>=8.2.2", "parameterized"],
        "quality": ["ruff==0.13.0", "isort==6.0.1"],
        "vllm": ["vllm>=0.8.5", "flashinfer-python>=0.2.1"],
        "sglang": ["sglang[srt]>=0.4.6", "flashinfer-python>=0.2.1"],
        "hf": ["optimum>=1.21.2"],
        "eval": ["lm_eval>=0.4.7", "evalplus>=0.3.1"],
        "triton": ["triton>=3.4.0"],
        "openai": ["uvicorn", "fastapi", "pydantic"],
        "mlx": ["mlx_lm>=0.28.2"],
    },
    include_dirs=include_dirs,
    cmdclass=(
        {"bdist_wheel": CachedWheelsCommand, "build_ext": additional_setup_kwargs.get("cmdclass", {}).get("build_ext")}
        if (BUILD_CUDA_EXT == "1" and additional_setup_kwargs)
        else {"bdist_wheel": CachedWheelsCommand}
    ),
    ext_modules=additional_setup_kwargs.get("ext_modules", []),
)
