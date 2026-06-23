---
id: rocm-bringup
kind: workflow
title: ROCm Model Bringup
status: active
tags: [rocm, amd, mi300, aiter, fp8, tunableop, bringup, hip]
aliases: [run on amd, rocm setup, mi300 bringup, amd gpu, port to rocm]
tools:
  - tool:build-from-source
  - tool:accuracy-divergence
  - tool:benchmark-serving
  - tool:tune-serving-config
triggers:
  - run a model on AMD/ROCm GPUs
  - port a CUDA setup to ROCm
  - debug ROCm-only accuracy or perf issues
validation:
  - correctness matches a trusted reference per platform
  - versions (ROCm/PyTorch/AITER/hipBLASLt) recorded
  - benchmark captured with ROCm env vars noted
source:
  - knowledge/vllm/rocm/rocm-setup-and-tuning.md
---

# ROCm Model Bringup

## Use When

Use to get a model correct and fast on AMD GPUs, or to triage a CUDA-vs-ROCm
divergence.

## Workflow

1. Use the pinned ROCm container (or `Dockerfile.rocm`); confirm `torch.version.hip`
   and `HIP_VISIBLE_DEVICES`. If building, use `tool:build-from-source`.
2. Serve and verify correctness first; if ROCm-only divergence appears, use
   `tool:accuracy-divergence` (check FP8 fnuz vs OCP format and scales).
3. Enable acceleration incrementally: Triton FlashAttention, then
   `VLLM_ROCM_USE_AITER=1`, then `PYTORCH_TUNABLEOP_ENABLED=1`; bisect with per-op
   toggles if a regression appears.
4. Measure with `tool:benchmark-serving` and tune via `tool:tune-serving-config`.
5. Record versions, env vars, and results in a run/handoff.

## Tool Sequence

- `tool:build-from-source`: ROCm editable build when needed.
- `tool:accuracy-divergence`: isolate ROCm-only correctness issues.
- `tool:benchmark-serving` + `tool:tune-serving-config`: performance.

## Validation Gates

- Correctness verified on ROCm against a reference before tuning.
- AITER/TunableOp regressions bisected with per-op switches.
- Full version + env-var set recorded with results.

## Update Policy

Update when a new ROCm version pairing, AITER toggle, or failure mode recurs.

## Related Runs

- 

## Source

- `knowledge/vllm/rocm/rocm-setup-and-tuning.md`
- `knowledge/vllm/rocm/rocm-fp8-formats.md`
