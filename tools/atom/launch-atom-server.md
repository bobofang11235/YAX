---
id: launch-atom-server
kind: tool
title: Launch ATOM Server
status: active
tags: [atom, rocm, aiter, serving, openai, deploy, launch, api, amd]
aliases: [atom openai server, atom serve, deploy atom, run atom server, aiter optimized model]
capabilities:
  - start ATOM's OpenAI-compatible server on AMD/ROCm
  - set TP / KV dtype / MTP / online-quant flags correctly
  - smoke-test the endpoint
inputs:
  - model path, AMD GPU count, target precision
outputs:
  - a running ATOM /v1 endpoint plus a verification curl
related:
  - workflow:serve-atom-model
  - tool:tune-atom-config
source:
  - knowledge/atom/configuration.md
---

# Launch ATOM Server

## Use When

Use to bring a model up on ATOM's OpenAI-compatible API on AMD/ROCm.

## How To Use

1. In the ATOM container (`rocm/atom-dev:latest`) or after `pip install ./ATOM`:
   ```bash
   # single GPU
   python -m atom.entrypoints.openai_server --model Qwen/Qwen3-0.6B --kv_cache_dtype fp8
   # multi-GPU TP
   python -m atom.entrypoints.openai_server --model deepseek-ai/DeepSeek-R1 --kv_cache_dtype fp8 -tp 8
   # MTP speculative decoding
   ... --method mtp --num-speculative-tokens 3
   ```
2. Confirm flags: `python -m atom.entrypoints.openai_server --help`.
3. Smoke test:
   ```bash
   curl localhost:8000/v1/models
   curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"<served-name>","messages":[{"role":"user","content":"hi"}]}'
   ```

## Validation

- `/v1/models` lists the model and chat returns tokens.
- First run compiles (~10 min); subsequent starts are fast.

## Constraints

- ROCm/AMD only; select GPUs with `HIP_VISIBLE_DEVICES`.
- ATOM flags differ from vLLM (e.g. `--kv_cache_dtype`, `-tp`); verify with `--help`.

## Related

- `knowledge/atom/configuration.md`
- `knowledge/atom/quantization.md`
- `tool:tune-atom-config`

## Source

- `knowledge/atom/configuration.md`
