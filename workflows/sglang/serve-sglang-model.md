---
id: serve-sglang-model
kind: workflow
title: Serve A Model With SGLang
status: active
tags: [sglang, serving, deploy, openai, launch, radix, bringup]
aliases: [serve with sglang, deploy sglang model, run model on sglang, sglang bringup]
tools:
  - tool:launch-sglang-server
  - tool:tune-sglang-config
triggers:
  - serve a model with SGLang
  - deploy an SGLang OpenAI endpoint
  - bring up SGLang and tune it
validation:
  - /v1/models lists the model and chat returns tokens
  - no OOM / KV-fit warnings at startup
  - tuning changes backed by sglang.bench_serving numbers
source:
  - knowledge/sglang/server-args.md
---

# Serve A Model With SGLang

## Use When

Use to bring a model up on SGLang and tune it, including mapping from vLLM habits.

## Workflow

1. Launch minimally with `tool:launch-sglang-server`
   (`python -m sglang.launch_server --model-path <m> --tp-size <gpus>`).
2. Smoke-test `/v1/models` and a chat/completion; sanity-check output quality.
3. If startup OOMs or KV won't fit, lower `--mem-fraction-static` / `--context-length`
   or use `--kv-cache-dtype fp8`.
4. Baseline with `python -m sglang.bench_serving`, then tune via
   `tool:tune-sglang-config` (one knob at a time), keeping RadixAttention on.
5. Record the working command, env vars, and results in a run.

## Tool Sequence

- `tool:launch-sglang-server`: start the endpoint.
- `tool:tune-sglang-config`: improve throughput/latency/fit with evidence.

## Validation Gates

- Endpoint healthy and returning coherent tokens.
- Performance changes backed by `bench_serving`; cache hit rate noted for
  prefix-heavy traffic.

## Update Policy

Update when a recurring SGLang launch flag, failure mode, or tuning step appears.

## Related Runs

- 

## Source

- `knowledge/sglang/server-args.md`
- `knowledge/sglang/performance-tuning.md`
- `knowledge/sglang/vllm-vs-sglang.md`
