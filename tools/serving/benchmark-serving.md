---
id: benchmark-serving
kind: tool
title: Benchmark vLLM Serving
status: active
tags: [benchmark, throughput, latency, ttft, perf, measure, vllm-bench]
aliases: [vllm bench serve, benchmark, throughput test, latency test, load test]
capabilities:
  - measure throughput, TTFT, and inter-token latency under load
  - produce comparable before/after numbers for a config change
inputs:
  - a running vLLM endpoint or model, input/output length profile, request rate
outputs:
  - tokens/s, TTFT, ITL/TPOT, and preemption observations
related:
  - workflow:optimize-throughput
  - tool:tune-serving-config
source:
  - knowledge/serving/performance-tuning.md
---

# Benchmark vLLM Serving

## Use When

Use to get evidence before/after any performance or config change. Never tune by
guesswork.

## How To Use

1. Pick a realistic input/output length distribution and request rate.
2. Run `vllm bench serve` (or `benchmarks/benchmark_serving.py`) against the
   endpoint; for offline throughput use `vllm bench throughput`.
3. Record: output tokens/s, TTFT, ITL/TPOT, GPU arch, vLLM version, attention
   backend, and every non-default flag/env var.
4. Change exactly one knob, re-run, and append to a small results table.

## Validation

- Results are reproducible across two runs (variance understood).
- Environment (arch/version/backend/flags) recorded with every number.

## Constraints

- Backend, dtype, and version all change numbers; fix them when comparing.

## Related

- `knowledge/serving/performance-tuning.md`
- `tool:tune-serving-config`

## Source

- `knowledge/serving/performance-tuning.md`
