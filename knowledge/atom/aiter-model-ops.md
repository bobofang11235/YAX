# ATOM Model Ops And AITER Kernels

## Use When

Use when changing how ATOM computes a layer, integrating or debugging an AITER
kernel, or reasoning about ATOM's performance. This is ATOM's core differentiator.
Authoritative: `docs/model_ops_guide.md`.

## Lesson

ATOM = **AiTer Optimized Model**. Its reason to exist is wiring
[AITER](https://github.com/ROCm/aiter) kernels into a clean serving engine. Model
layers are thin wrappers that dispatch to AITER implementations.

### AITER Backends

AITER provides three kernel flavors; ATOM selects among them per op:

- **ASM** — hand-written assembly kernels (fastest for targeted shapes/arch).
- **CK** — Composable Kernel (templated C++ GEMM/attention).
- **Triton** — portable Triton kernels.

### Wrapped Ops (`atom/layers/`, see model_ops_guide)

- **Linear** (incl. quantized FP8/MXFP4/INT GEMMs).
- **Attention** (prefill/decode paged attention; MLA for DeepSeek/GLM; DSA sparse
  attention for GLM-5.2).
- **MoE** (fused expert GEMMs, routing/top-k) with expert parallelism.
- **Norm** (RMSNorm), rotary, activation.

A new model reuses these wrappers so quantization, parallelism, and CUDA graphs
work for free.

## Rules

- For an op change, start in `atom/layers/` and `docs/model_ops_guide.md`; the
  actual kernel lives in AITER, not ATOM.
- When a layer is slow or wrong, identify which AITER backend (ASM/CK/Triton) is
  selected for the shape; a fallback or wrong backend is a common cause.
- Keep model files using the wrappers; do not call AITER directly from a model.
- Pin AITER (`amd-aiter`) and ROCm versions with any result — ATOM perf is
  AITER-version-coupled.

## Avoid

- Treating ATOM kernels as the source of a perf/accuracy bug before checking the
  underlying AITER kernel and the selected backend.
- Bypassing the op wrappers (breaks quant/TP/graph integration).

## Related

- `knowledge/atom/quantization.md`
- `knowledge/atom/performance-tuning.md`
- `knowledge/vllm/rocm/rocm-setup-and-tuning.md` (AITER background, shared)

## Source

- `docs/model_ops_guide.md`, `atom/layers/`
- AITER: https://github.com/ROCm/aiter
