---
id: launch-openai-server
kind: tool
title: Launch vLLM OpenAI Server
status: active
tags: [serving, openai, deploy, vllm-serve, launch, api]
aliases: [vllm serve, openai server, deploy model, host model]
capabilities:
  - start an OpenAI-compatible vLLM endpoint
  - set parallelism, memory, and context flags correctly
  - smoke-test the endpoint
inputs:
  - model id/path, GPU count, target context length
outputs:
  - a running /v1 endpoint plus a verification curl
related:
  - workflow:serve-new-model
  - tool:tune-serving-config
source:
  - knowledge/vllm/serving/engine-args-reference.md
---

# Launch vLLM OpenAI Server

## Use When

Use to bring up a model behind the OpenAI-compatible API with sane defaults.

## How To Use

1. Minimal launch: `vllm serve <model> --tensor-parallel-size <gpus>`.
2. Add only the flags a constraint requires:
   - context: `--max-model-len <N>`
   - memory: `--gpu-memory-utilization 0.90`, `--kv-cache-dtype fp8`
   - naming/auth: `--served-model-name <name>`, `--api-key <key>`, `--port <p>`
   - quantized checkpoint: usually auto-detected; force with `-q <method>`.
3. Confirm flags exist for the installed version: `vllm serve --help`.
4. Smoke test:
   ```bash
   curl localhost:8000/v1/models
   curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"<name>","messages":[{"role":"user","content":"hi"}]}'
   ```

## Validation

- `/v1/models` lists the model and a chat/completion returns tokens.
- No preemption or "cannot fit max_model_len" warnings in the startup log.

## Constraints

- Set `VLLM_*` env vars before launch (backend, ROCm/CUDA knobs).
- Flag names are version-sensitive; verify against `--help`.

## Related

- `knowledge/vllm/serving/engine-args-reference.md`
- `tool:tune-serving-config`

## Source

- `knowledge/vllm/serving/engine-args-reference.md`
