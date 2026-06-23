---
id: tune-serving-config
kind: tool
title: Tune vLLM Serving Config
status: active
tags: [tuning, throughput, latency, memory, batching, max-num-seqs, knobs]
aliases: [tune throughput, tune latency, batching knobs, improve performance]
capabilities:
  - pick the right knob for a throughput/latency/memory goal
  - apply one-change-at-a-time tuning order
inputs:
  - a goal (throughput, latency, or fit) and a benchmark baseline
outputs:
  - a tuned flag/env set with measured improvement
related:
  - workflow:optimize-throughput
  - tool:benchmark-serving
source:
  - knowledge/serving/performance-tuning.md
---

# Tune vLLM Serving Config

## Use When

Use after a baseline benchmark when throughput is low, latency high, or memory
tight.

## How To Use

1. State the goal: offline throughput, interactive latency, or memory fit.
2. Apply the matching order from `performance-tuning.md`:
   - throughput: raise `--max-num-batched-tokens`, then `--max-num-seqs`, grow KV
     pool, keep CUDA graphs, add TP, quantize.
   - latency: moderate batched-tokens, prefix caching, spec decode, fast decode
     backend.
   - memory: lower `--gpu-memory-utilization`/`--max-model-len`, FP8 KV, quantize.
3. Change one knob, re-benchmark with `tool:benchmark-serving`, keep or revert.

## Validation

- Each accepted change is backed by a measured before/after.
- Decode P99 latency still meets target after throughput changes.

## Constraints

- Fix correctness before tuning; never tune around a numeric bug.

## Related

- `knowledge/serving/performance-tuning.md`
- `tool:benchmark-serving`

## Source

- `knowledge/serving/performance-tuning.md`
