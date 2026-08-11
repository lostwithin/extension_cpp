# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "extension_cpp"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def _find_cutlass_include():
    """Locate the CUTLASS/CuTe `include` dir for compiling gemm.cu."""
    # 1) Explicit override.
    env = os.environ.get("CUTLASS_PATH")
    if env:
        for cand in (env, os.path.join(env, "include")):
            if os.path.exists(os.path.join(cand, "cute", "tensor.hpp")):
                return cand
    # 2) Headers shipped by the `nvidia-cutlass` wheel.
    try:
        import cutlass_library

        cand = os.path.join(
            os.path.dirname(cutlass_library.__file__), "source", "include"
        )
        if os.path.exists(os.path.join(cand, "cute", "tensor.hpp")):
            return cand
    except ImportError:
        pass
    return None


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            # CuTe / CUTLASS (gemm.cu) require C++17 and relaxed device
            # constexpr / extended lambdas.
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    # CuTe / CUTLASS headers needed by csrc/cuda/gemm.cu. Prefer the headers
    # shipped by the `nvidia-cutlass` wheel; fall back to a CUTLASS_PATH env var.
    include_dirs = []
    if use_cuda:
        cutlass_include = _find_cutlass_include()
        if cutlass_include is None:
            raise RuntimeError(
                "Could not locate CUTLASS/CuTe headers required by gemm.cu. "
                "Install them with `pip install nvidia-cutlass`, or set "
                "CUTLASS_PATH to a CUTLASS checkout (its 'include' dir)."
            )
        include_dirs.append(cutlass_include)

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Example of PyTorch C++ and CUDA extensions",
    long_description=open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "README.md")
    ).read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/extension-cpp",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
