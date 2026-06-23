# Repo Layout And Build

## Use When

Use when you need to find where code lives or build/install vLLM from source to
make changes.

## Lesson

### Directory Map (top level)

- `vllm/` — the Python package:
  - `vllm/v1/` — the V1 engine: `engine/`, `core/` (scheduler, KV cache),
    `worker/`, `executor/`, `attention/`.
  - `vllm/engine/` — `arg_utils.py` (all engine args), legacy engine glue.
  - `vllm/entrypoints/` — `LLM` class and `openai/` server.
  - `vllm/model_executor/` — `models/` (one file per architecture + `registry.py`)
    and `layers/` (attention, quantization, MoE, norms, sampler).
  - `vllm/attention/` — backend selection and shared attention code.
  - `vllm/platforms/` — `cuda.py`, `rocm.py`, `cpu.py`, etc. (capability gating).
  - `vllm/distributed/` — parallel state, communicators, KV transfer.
  - `vllm/config.py`, `vllm/envs.py` — config objects and env var registry.
- `csrc/` — C++/CUDA/HIP kernels (custom ops).
- `benchmarks/` — throughput/latency/serving benchmarks.
- `tests/` — pytest suite, mirrors package layout.
- `docs/` — documentation source.
- `examples/` — runnable usage examples.
- `requirements/`, `setup.py`, `pyproject.toml`, `CMakeLists.txt` — build config.
- `Dockerfile`, `Dockerfile.rocm` — reference CUDA/ROCm build environments.

### Building From Source

Clone first (`git clone https://github.com/vllm-project/vllm`).

- **Python-only editing (fast).** If you only touch `.py` files and have a matching
  prebuilt vLLM/PyTorch, use the precompiled-kernels path so you skip the C++/CUDA
  compile:
  ```bash
  VLLM_USE_PRECOMPILED=1 pip install -e .
  ```
  This installs editable Python over prebuilt binaries — edits to `vllm/*.py` take
  effect immediately.
- **Full build (kernels changed).** Needs the CUDA or ROCm toolchain:
  ```bash
  pip install -e .          # compiles csrc/ via CMake; slow first time
  ```
  Set `MAX_JOBS` to bound parallel compile memory; `CMAKE_BUILD_TYPE` for debug.
- **ROCm.** Build inside the `Dockerfile.rocm` environment (correct ROCm/PyTorch-
  ROCm/AITER). Verify `torch.version.hip` is set.
- **Verify:** `python -c "import vllm; print(vllm.__version__)"` and a tiny
  `vllm serve` smoke test.

## Rules

- Use `VLLM_USE_PRECOMPILED=1 pip install -e .` for Python-only changes; reserve
  the full compile for `csrc/` work.
- Match the source checkout's expected PyTorch/CUDA(ROCm) versions; mismatches
  cause import/ABI errors.
- Find behavior by layer: scheduling → `vllm/v1/core/sched/`, model math →
  `vllm/model_executor/models/`, args → `vllm/engine/arg_utils.py`, env →
  `vllm/envs.py`.

## Avoid

- Running a full CUDA/ROCm compile just to edit Python.
- Editing in `site-packages` instead of an editable checkout.

## Related

- `knowledge/vllm/development/adding-a-model.md`
- `knowledge/vllm/development/custom-ops-and-platforms.md`
- `knowledge/vllm/development/contributing-and-testing.md`
- `knowledge/vllm/architecture/vllm-architecture-overview.md`

## Source

- vLLM repo root; https://docs.vllm.ai/en/latest/contributing/
- https://docs.vllm.ai/en/latest/getting_started/installation/
