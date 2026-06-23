---
id: serve-new-model
kind: workflow
title: Serve A New Model
status: active
tags: [serving, deploy, new-model, openai, launch, bringup]
aliases: [serve model, deploy model, bring up model, host a model]
tools:
  - tool:launch-openai-server
  - tool:add-new-model
  - tool:oom-kv-triage
triggers:
  - serve a model behind the OpenAI API
  - deploy a new or unsupported model
  - first launch of a model on given GPUs
validation:
  - /v1/models lists the model and chat returns tokens
  - no preemption or KV-fit warnings at startup
  - greedy output sane vs a reference
source:
  - knowledge/serving/engine-args-reference.md
---

# Serve A New Model

## Use When

Use to bring a model up on the OpenAI-compatible endpoint, including models vLLM
does not yet support.

## Workflow

1. Check support: is the architecture in the model registry? If not, use
   `tool:add-new-model` first.
2. Launch minimally with `tool:launch-openai-server`
   (`vllm serve <model> -tp <gpus>`); add only required flags.
3. If startup OOMs or warns about KV fit, use `tool:oom-kv-triage`.
4. Smoke test `/v1/models` and a chat/completion; sanity-check output quality.
5. Record the working command + environment in a run.

## Tool Sequence

- `tool:add-new-model`: only when the architecture is unsupported.
- `tool:launch-openai-server`: start the endpoint.
- `tool:oom-kv-triage`: resolve memory/KV startup failures.

## Validation Gates

- Endpoint healthy and returning tokens.
- No KV-fit or preemption warnings; output is coherent vs a reference.

## Update Policy

Update when a recurring launch flag, model quirk, or startup failure pattern
appears.

## Related Runs

- 

## Source

- `knowledge/serving/engine-args-reference.md`
- `knowledge/serving/features-overview.md`
