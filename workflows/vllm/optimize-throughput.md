---
id: optimize-throughput
kind: workflow
title: Optimize Throughput And Latency
status: active
tags: [performance, throughput, latency, tuning, benchmark, batching, memory]
aliases: [optimize performance, speed up, tune throughput, reduce latency, improve tokens per second]
tools:
  - tool:benchmark-serving
  - tool:tune-serving-config
  - tool:oom-kv-triage
triggers:
  - throughput too low or latency too high
  - tune a serving config
  - before/after performance validation
validation:
  - baseline and post-change benchmarks recorded
  - one knob changed per measured step
  - environment (arch/version/backend/flags) captured
source:
  - knowledge/vllm/serving/performance-tuning.md
---

# Optimize Throughput And Latency

## Use When

Use to improve tokens/s, TTFT, or ITL with measured evidence rather than guesses.

## Workflow

1. Define the goal: offline throughput, interactive latency, or memory fit.
2. Establish a baseline with `tool:benchmark-serving`.
3. Apply one knob with `tool:tune-serving-config` following the tuning order.
4. Re-benchmark; keep the change only if it improves the target without breaking
   latency/correctness budgets.
5. If a memory wall or preemption appears, use `tool:oom-kv-triage`.
6. Record the final config and the results table in a run.

## Tool Sequence

- `tool:benchmark-serving`: baseline and per-step measurement.
- `tool:tune-serving-config`: pick and apply the next knob.
- `tool:oom-kv-triage`: clear memory walls hit while scaling batch.

## Validation Gates

- Each accepted change has a before/after number.
- Decode P99 latency still within target after throughput changes.

## Update Policy

Update when a new knob, platform booster, or tuning order proves better.

## Related Runs

- 

## Source

- `knowledge/vllm/serving/performance-tuning.md`
- `knowledge/vllm/architecture/scheduler-batching.md`
