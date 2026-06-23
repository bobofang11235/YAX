# Attention Backends

## Use When

Use when choosing or debugging an attention backend, setting
`VLLM_ATTENTION_BACKEND`, hitting "backend does not support ..." errors, or
porting attention behavior across CUDA and ROCm.

## Lesson

Attention is pluggable. vLLM picks a backend automatically based on platform,
GPU, dtype, head size, and features (sliding window, MLA, etc.), but you can
override it with the `VLLM_ATTENTION_BACKEND` environment variable. Backends
implement prefill, decode, and the paged KV read/write.

### Common Backends

CUDA / NVIDIA:

- `FLASH_ATTN` — FlashAttention 2/3; the usual default on supported NVIDIA GPUs.
- `FLASHINFER` — FlashInfer kernels; strong for decode and FP8 KV, needed for
  some features; install `flashinfer-python`.
- `XFORMERS` — fallback for head sizes / cases FlashAttention does not cover.
- `TRITON_ATTN` — Triton kernels; portable fallback.
- `FLASHMLA` / MLA backends — for DeepSeek-style Multi-head Latent Attention.

ROCm / AMD:

- `ROCM_FLASH` / Triton Flash — set `VLLM_USE_TRITON_FLASH_ATTN=1` (default on
  ROCm) for the Triton FlashAttention path.
- `ROCM_AITER_*` — AITER-accelerated attention/MLA when
  `VLLM_ROCM_USE_AITER=1` (MI300-class GPUs).

### How Selection Works

- The platform layer (`vllm/platforms/`) and
  `vllm/attention/selector.py` choose a backend from capability checks.
- Override: `VLLM_ATTENTION_BACKEND=FLASHINFER vllm serve ...`. If the override is
  invalid for the model/GPU, vLLM errors instead of silently downgrading.

## Rules

- Let auto-selection win unless you have a measured reason to override.
- If you hit an unsupported head size / feature, try `XFORMERS` (CUDA) or the
  Triton backend before assuming a bug.
- FP8 KV cache often pairs best with `FLASHINFER` on CUDA.
- Record the backend in any benchmark or accuracy result; it changes both speed
  and (slightly) numerics.
- Backend choice can constrain valid `--block-size`; if you set both, keep them
  compatible.

## Avoid

- Comparing throughput across machines without fixing the backend.
- Assuming the same backend name exists on both CUDA and ROCm — the sets differ.

## Related

- `knowledge/serving/environment-variables.md`
- `knowledge/rocm/rocm-setup-and-tuning.md`
- `knowledge/cuda/cuda-setup-and-tuning.md`

## Source

- `vllm/attention/`, `vllm/v1/attention/backends/`, `vllm/attention/selector.py`
- https://docs.vllm.ai/en/latest/
