---
id: tune-atom-config
kind: tool
title: Tune ATOM Config
status: active
tags: [atom, rocm, aiter, tuning, throughput, latency, tbo, mxfp4, tunableop, mtp]
aliases: [tune atom, atom throughput, atom latency, tbo tuning, aiter backend, atom performance]
capabilities:
  - pick the right ATOM/AITER knob for a throughput/latency goal on AMD
  - apply one-change-at-a-time tuning with trace evidence
inputs:
  - a goal (throughput/latency/fit) and a benchmark_serving baseline
outputs:
  - a tuned ATOM flag/env set with measured improvement
related:
  - workflow:serve-atom-model
  - tool:launch-atom-server
source:
  - knowledge/atom/performance-tuning.md
---

# Tune ATOM Config

## Use When

Use after an `atom.benchmarks.benchmark_serving` baseline when ATOM throughput is
low or latency high on AMD GPUs.

## How To Use

1. State the goal: throughput, latency, or memory fit.
2. Apply the order from `knowledge/atom/performance-tuning.md`:
   - keep piecewise torch.compile + CUDA graphs (default opt level 3);
   - quantize (FP8 / MXFP4 / `--kv_cache_dtype fp8`; try `--online_quant_config`);
   - MoE: EP + Two-Batch Overlap (TBO) + DP attention;
   - latency: MTP (`--method mtp`);
   - AMD GEMM: `PYTORCH_TUNABLEOP_ENABLED=1`, confirm the AITER backend (ASM/CK/
     Triton) for your shapes;
   - consider P/D disaggregation when prefill/decode contend.
3. Change one knob, re-bench, collect a trace and read the per-kernel breakdown.

## Validation

- Each accepted change has a measured before/after.
- Trace shows the expected AITER kernels (no slow fallback).

## Constraints

- Fix correctness (lm-eval) before tuning; warm up before measuring (compile cost).

## Related

- `knowledge/atom/performance-tuning.md`
- `knowledge/atom/aiter-model-ops.md`
- `knowledge/atom/distributed-tbo.md`

## Source

- `knowledge/atom/performance-tuning.md`
