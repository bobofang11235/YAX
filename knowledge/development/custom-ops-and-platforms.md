# Custom Ops And Platform Abstraction

## Use When

Use when adding/changing a CUDA/HIP/Triton kernel, exposing it to Python, or
making a feature degrade correctly across CUDA, ROCm, and CPU.

## Lesson

### Custom Ops

- C++/CUDA/HIP kernels live in `csrc/`, built by CMake (`CMakeLists.txt`) and
  bound into Python as `torch.ops`/the `_C` extension. ROCm builds the same
  sources hipified for HIP.
- Triton kernels live in Python under `vllm/` (e.g. attention, MoE, quant) and
  need no C++ compile — the portable path across CUDA/ROCm.
- vLLM's `CustomOp` base lets a layer provide `forward_cuda`, `forward_hip`,
  `forward_cpu`, `forward_native` so each platform picks the best implementation
  with a pure-PyTorch fallback.

### Platform Abstraction

- `vllm/platforms/` (`cuda.py`, `rocm.py`, `cpu.py`, ...) centralizes
  device-specific decisions: supported dtypes, attention backend selection,
  compute-capability gates, comms library.
- New hardware-sensitive behavior goes through the platform layer, not scattered
  `if torch.cuda...` checks.

### Editing Kernels — Workflow

1. Edit `csrc/`; rebuild with `pip install -e .` (full compile).
2. Add/adjust the Python binding and a `CustomOp` `forward_<platform>` path.
3. Provide a `forward_native` fallback so unsupported platforms still run.
4. Gate availability in the platform layer by capability; allow opt-out via
   `VLLM_DISABLED_KERNELS` for debugging.
5. Test numerics against the native fallback, then benchmark.

## Rules

- Prefer Triton for portability when CUDA and ROCm must both work; reserve C++
  kernels for cases that need them.
- Always ship a `forward_native` fallback and a capability gate.
- Make a new kernel disableable (so a regression can be bisected without a
  rebuild).
- Hipify-friendly C++: keep CUDA-specific intrinsics behind macros the ROCm build
  can translate.

## Avoid

- Hard-coding `cuda` assumptions in shared layers; route through `vllm/platforms/`.
- Landing a kernel with no fallback and no test against a reference.

## Related

- `knowledge/architecture/attention-backends.md`
- `knowledge/development/repo-layout-and-build.md`
- `knowledge/rocm/rocm-setup-and-tuning.md`

## Source

- `csrc/`, `vllm/platforms/`, `vllm/model_executor/custom_op.py`
- https://docs.vllm.ai/en/latest/contributing/
