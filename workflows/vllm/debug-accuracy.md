---
id: debug-accuracy
kind: workflow
title: Debug Accuracy Divergence
status: active
tags: [accuracy, correctness, divergence, fp8, quantization, backend, debug]
aliases: [wrong output, gibberish, accuracy drop, fix correctness, output mismatch]
tools:
  - tool:accuracy-divergence
  - tool:add-new-model
triggers:
  - model output wrong, gibberish, or worse than baseline
  - accuracy regression after a config or quant change
  - cross-platform numerical mismatch
validation:
  - trusted reference established first
  - one variable bisected at a time
  - first divergent variable identified before broad changes
source:
  - knowledge/vllm/serving/quantization.md
---

# Debug Accuracy Divergence

## Use When

Use when output quality is wrong or has regressed and you must find the cause.

## Workflow

1. Establish a trusted reference (HF Transformers greedy, or fp16 vLLM on a known
   GPU).
2. Use `tool:accuracy-divergence` to bisect one variable at a time: dtype →
   quantization → KV dtype → attention backend → platform FP8 format.
3. If the model itself is newly added/ported, suspect `load_weights` mapping and
   revisit `tool:add-new-model`'s logit A-B.
4. Lock in the corrected config; only then consider performance.
5. Record the first divergent variable and the fix in a run.

## Tool Sequence

- `tool:accuracy-divergence`: structured one-variable bisection.
- `tool:add-new-model`: when a ported model's weight mapping is suspect.

## Validation Gates

- Greedy output matches the reference within tolerance after the fix.
- Precision, backend, and model-code causes are not conflated.

## Update Policy

Update when a recurring divergence cause (e.g. a specific backend/quant combo)
should be promoted into the bisection order.

## Related Runs

- 

## Source

- `knowledge/vllm/serving/quantization.md`
- `knowledge/vllm/rocm/rocm-fp8-formats.md`
- `knowledge/vllm/architecture/attention-backends.md`
