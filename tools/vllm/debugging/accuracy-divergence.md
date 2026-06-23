---
id: accuracy-divergence
kind: tool
title: Accuracy Divergence Triage
status: active
tags: [accuracy, correctness, divergence, fp8, quantization, backend, logits, debug]
aliases: [wrong output, garbage output, accuracy drop, divergence, gibberish, mismatch]
capabilities:
  - isolate the variable causing wrong/low-quality output
  - separate precision from backend from model bugs
inputs:
  - a model+config producing wrong output and a trusted reference
outputs:
  - the first divergent variable and a corrected config
related:
  - workflow:debug-accuracy
  - tool:add-new-model
source:
  - knowledge/vllm/serving/quantization.md
---

# Accuracy Divergence Triage

## Use When

Use when output is wrong, gibberish, or measurably worse than a trusted baseline.

## How To Use

1. Establish a trusted reference (HF Transformers greedy, or fp16 vLLM on a known
   GPU).
2. Bisect one variable at a time, in this order:
   - dtype (bf16/fp16) before any quantization;
   - quantization method/kernel (drop `-q`, try fp16);
   - KV cache dtype (`--kv-cache-dtype auto` vs fp8);
   - attention backend (`VLLM_ATTENTION_BACKEND`);
   - platform format on ROCm (fp8 fnuz vs OCP).
3. The first variable that restores correctness is the culprit.

## Validation

- Greedy output matches the reference within tolerance once the variable is fixed.

## Constraints

- Fix accuracy before any performance tuning.

## Related

- `knowledge/vllm/serving/quantization.md`
- `knowledge/vllm/rocm/rocm-fp8-formats.md`
- `knowledge/vllm/architecture/attention-backends.md`

## Source

- `knowledge/vllm/serving/quantization.md`
