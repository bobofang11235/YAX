---
id: tune-sglang-config
kind: tool
title: Tune SGLang Config
status: active
tags: [sglang, tuning, throughput, latency, memory, radix, mem-fraction-static, chunked-prefill]
aliases: [tune sglang, sglang throughput, sglang latency, radix cache tuning, mem-fraction-static]
capabilities:
  - pick the right SGLang knob for a throughput/latency/memory goal
  - apply one-change-at-a-time tuning with RadixAttention in mind
inputs:
  - a goal (throughput/latency/fit) and a bench_serving baseline
outputs:
  - a tuned SGLang flag/env set with measured improvement
related:
  - workflow:serve-sglang-model
  - tool:launch-sglang-server
source:
  - knowledge/sglang/performance-tuning.md
---

# Tune SGLang Config

## Use When

Use after a `sglang.bench_serving` baseline when throughput is low, latency high,
or memory tight.

## How To Use

1. State the goal: throughput, latency, or memory fit.
2. Apply the matching order from `sglang/performance-tuning.md`:
   - throughput: raise `--mem-fraction-static`, `--max-running-requests`,
     `--chunked-prefill-size`; keep overlap scheduler + CUDA graph + RadixAttention;
     `--schedule-policy lpm`; quantize / DeepGEMM.
   - latency: moderate `--chunked-prefill-size`, EAGLE3 spec decode, fast attention
     backend.
   - memory: lower `--mem-fraction-static`/`--context-length`, `--kv-cache-dtype fp8`.
3. Change one knob, re-bench with `python -m sglang.bench_serving`, keep or revert.

## Validation

- Each accepted change has a measured before/after.
- Report the RadixAttention cache hit rate for prefix-heavy traffic.

## Constraints

- Fix correctness before tuning; never tune around a numeric bug.

## Related

- `knowledge/sglang/performance-tuning.md`
- `knowledge/sglang/radixattention.md`
- `tool:launch-sglang-server`

## Source

- `knowledge/sglang/performance-tuning.md`
