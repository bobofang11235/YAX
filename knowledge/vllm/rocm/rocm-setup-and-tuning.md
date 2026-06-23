# ROCm Setup And Tuning

## Use When

Use when building, running, or tuning vLLM on AMD GPUs (MI200/MI250/MI300,
gfx90a/gfx942), or porting a CUDA setup to ROCm.

## Lesson

vLLM supports ROCm via HIP. The platform layer `vllm/platforms/rocm.py` selects
ROCm kernels and backends. The reliable path is the official ROCm Docker image
(`Dockerfile.rocm`) which pins ROCm, PyTorch-ROCm, FlashAttention, AITER, and
hipBLASLt versions that are known to work together — version mismatch is the
number-one ROCm failure.

### Install

- Prefer the prebuilt ROCm container (rocm/vllm) or build from `Dockerfile.rocm`.
- Bare-metal: install ROCm, the matching PyTorch-ROCm wheel, then build vLLM from
  source. Confirm `python -c "import torch; print(torch.version.hip)"`.
- Select GPUs with `HIP_VISIBLE_DEVICES` (or `ROCR_VISIBLE_DEVICES`), not
  `CUDA_VISIBLE_DEVICES`.

### Attention Backend

- Triton FlashAttention is the default on ROCm: `VLLM_USE_TRITON_FLASH_ATTN=1`.
- For MI300 with AITER, MLA and paged-attention kernels are AITER-accelerated.
- Override with `VLLM_ATTENTION_BACKEND` only with a measured reason.

### AITER (AMD's optimized kernel library)

- Master switch: `VLLM_ROCM_USE_AITER=1` (MI300-class). Big throughput gains for
  GEMM, MoE, attention, RMSNorm.
- Per-op toggles to isolate a regression:
  `VLLM_ROCM_USE_AITER_LINEAR`, `..._MOE`, `..._PAGED_ATTN`, `..._RMSNORM`.
- If AITER causes a numerical or build issue, disable it to bisect.

### GEMM Tuning

- `PYTORCH_TUNABLEOP_ENABLED=1` autotunes GEMMs (hipBLASLt/rocBLAS) and caches the
  best kernel; `PYTORCH_TUNABLEOP_TUNING=1` (re)runs tuning. First run is slow;
  results cache to disk. Big help on MI300.
- hipBLASLt is the preferred GEMM library; ensure the container ships a matching
  version.
- `VLLM_ROCM_FP8_PADDING=1` pads FP8 GEMM shapes for better performance.

### Collectives

- RCCL is the ROCm NCCL equivalent; the same `NCCL_*` env names apply
  (`NCCL_DEBUG=INFO`, `NCCL_P2P_DISABLE`, `NCCL_IB_DISABLE`).
- For multi-GPU within a node, verify XGMI/PCIe peer access; `VLLM_SKIP_P2P_CHECK`
  can bypass a flaky check (understand why first).

### Unsupported Arch

- `HSA_OVERRIDE_GFX_VERSION=9.4.2` (or similar) can coax an unsupported card to use
  a near gfx target — experimental, expect rough edges.

## Rules

- Pin and record ROCm, PyTorch-ROCm, FlashAttention/AITER/hipBLASLt versions with
  every result; ROCm regressions are usually version-coupled.
- Bisect AITER and TunableOp with their per-op / on-off switches when chasing a
  regression.
- Treat ROCm-only divergence vs CUDA as backend work (kernels/format), not a model
  bug, until evidence says otherwise.

## Avoid

- Mixing a bare-metal ROCm with a PyTorch wheel built for a different ROCm.
- Forcing a CUDA-only attention backend on ROCm.
- Leaving `PYTORCH_TUNABLEOP_TUNING=1` on in production (re-tunes every start).

## Related

- `knowledge/vllm/rocm/rocm-fp8-formats.md`
- `knowledge/vllm/architecture/attention-backends.md`
- `knowledge/vllm/serving/environment-variables.md`

## Source

- `vllm/platforms/rocm.py`, `Dockerfile.rocm`
- https://docs.vllm.ai/en/latest/getting_started/installation/gpu/rocm.html
- AITER: https://github.com/ROCm/aiter
