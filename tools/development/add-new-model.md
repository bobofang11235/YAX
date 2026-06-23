---
id: add-new-model
kind: tool
title: Add A New Model To vLLM
status: active
tags: [model, architecture, registry, port, unsupported, development, load-weights]
aliases: [unsupported architecture, new model, port model, register model, add model]
capabilities:
  - implement and register a new model architecture
  - validate it against HF Transformers logits
inputs:
  - HF config/checkpoint with an unsupported architecture name
outputs:
  - a registered vLLM model passing a greedy logit A-B vs HF
related:
  - workflow:serve-new-model
  - tool:build-from-source
source:
  - knowledge/development/adding-a-model.md
---

# Add A New Model To vLLM

## Use When

Use when vLLM errors on an unsupported architecture or you must port a model.

## How To Use

1. Confirm it is missing in `vllm/model_executor/models/registry.py`.
2. Copy the closest existing model file as a starting point.
3. Implement the module reusing vLLM `Attention`/parallel-linear/MoE layers; get
   `load_weights` name-mapping exact.
4. Register the class against the HF `architectures` name.
5. Add interface mixins (LoRA/PP/multimodal) as needed.
6. Validate with a greedy first-token logit A-B vs HF Transformers, then serve.

## Validation

- Greedy logits/output match HF Transformers within tolerance on fixed prompts.
- Tensor-parallel run produces identical output to single-GPU.

## Constraints

- Reuse vLLM layers so TP/quantization/backends work without extra code.

## Related

- `knowledge/development/adding-a-model.md`
- `tool:build-from-source`

## Source

- `knowledge/development/adding-a-model.md`
