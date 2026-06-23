---
id: serve-atom-model
kind: workflow
title: Serve A Model With ATOM
status: active
tags: [atom, rocm, aiter, serving, deploy, openai, amd, bringup, mxfp4, tbo]
aliases: [serve with atom, deploy atom model, run model on atom, atom bringup, aiter optimized model]
tools:
  - tool:launch-atom-server
  - tool:tune-atom-config
triggers:
  - serve a model with ATOM on AMD/ROCm
  - bring up ATOM and tune it
  - run a model on AITER
validation:
  - /v1/models lists the model and chat returns tokens
  - accuracy checked with lm-eval before tuning
  - tuning changes backed by benchmark_serving + trace evidence
source:
  - knowledge/atom/configuration.md
---

# Serve A Model With ATOM

## Use When

Use to bring a model up on ATOM (AMD/ROCm, AITER) and tune it. Consider a recipe
from the ATOM repo for your exact model first.

## Workflow

1. Use the ATOM container (`rocm/atom-dev:latest`) or install AITER + ATOM.
2. Check the `recipes/` for your model family (DeepSeek-R1, Qwen3-235B, GLM-5,
   GPT-OSS, Kimi-K2, ...) and copy its flags.
3. Launch with `tool:launch-atom-server` (mind the ~10 min first-run compile).
4. Smoke-test `/v1/models` + a chat; validate accuracy with `lm-eval` (gsm8k).
5. Baseline with `atom.benchmarks.benchmark_serving`, then tune via
   `tool:tune-atom-config` (one knob at a time), reading trace breakdowns.
6. Record the working command, versions (ROCm/AITER/ATOM), and results in a run.

## Tool Sequence

- `tool:launch-atom-server`: start the endpoint.
- `tool:tune-atom-config`: improve throughput/latency with evidence.

## Validation Gates

- Endpoint healthy and returning coherent tokens; lm-eval accuracy acceptable.
- Performance changes backed by benchmark + trace; AITER backend confirmed.

## Update Policy

Update when a recurring ATOM launch flag, recipe pattern, or failure mode appears.

## Related Runs

- 

## Source

- `knowledge/atom/configuration.md`
- `knowledge/atom/performance-tuning.md`
- `knowledge/atom/vllm-vs-atom.md`
