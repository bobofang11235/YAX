---
id: build-from-source
kind: tool
title: Build vLLM From Source
status: active
tags: [build, install, source, editable, compile, csrc, cmake, development]
aliases: [build vllm, install from source, editable install, compile vllm, dev install]
capabilities:
  - set up an editable vLLM checkout for Python or kernel edits
  - choose the fast precompiled path vs full compile
inputs:
  - a vLLM clone and a matching PyTorch + CUDA/ROCm toolchain
outputs:
  - an importable editable vLLM with a smoke-tested launch
related:
  - workflow:contribute-feature
  - tool:add-new-model
source:
  - knowledge/vllm/development/repo-layout-and-build.md
---

# Build vLLM From Source

## Use When

Use before editing vLLM, to get an editable install that matches the environment.

## How To Use

1. Clone: `git clone https://github.com/vllm-project/vllm`.
2. Python-only edits (fast): `VLLM_USE_PRECOMPILED=1 pip install -e .`.
3. Kernel (`csrc/`) edits: `pip install -e .` (full CMake compile); bound memory
   with `MAX_JOBS`.
4. ROCm: build inside the `Dockerfile.rocm` environment; verify
   `python -c "import torch; print(torch.version.hip)"`.
5. Verify: `python -c "import vllm; print(vllm.__version__)"` + a tiny
   `vllm serve` smoke test.

## Validation

- `import vllm` works and a minimal serve responds.
- PyTorch/CUDA(ROCm) versions match the checkout's expectations.

## Constraints

- Full compile needs the GPU toolchain; precompiled path needs a compatible
  prebuilt.

## Related

- `knowledge/vllm/development/repo-layout-and-build.md`
- `knowledge/vllm/development/contributing-and-testing.md`

## Source

- `knowledge/vllm/development/repo-layout-and-build.md`
