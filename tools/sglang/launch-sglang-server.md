---
id: launch-sglang-server
kind: tool
title: Launch SGLang Server
status: active
tags: [sglang, serving, openai, deploy, launch, api, srt]
aliases: [sglang launch_server, sglang serve, deploy sglang, run sglang server]
capabilities:
  - start an SGLang OpenAI-compatible endpoint with sane flags
  - map vLLM-style intent to SGLang server args
  - smoke-test the endpoint
inputs:
  - model path, GPU count, target context length
outputs:
  - a running SGLang /v1 endpoint plus a verification curl
related:
  - workflow:serve-sglang-model
  - tool:tune-sglang-config
source:
  - knowledge/sglang/server-args.md
---

# Launch SGLang Server

## Use When

Use to bring a model up on SGLang's OpenAI-compatible API.

## How To Use

1. Minimal: `python -m sglang.launch_server --model-path <model> --tp-size <gpus>`.
2. Add only what a constraint needs:
   - context: `--context-length N`
   - memory: `--mem-fraction-static 0.85`, `--kv-cache-dtype fp8_e4m3`
   - concurrency: `--max-running-requests N`
   - quantized checkpoint: usually auto-detected; force with `--quantization`.
3. Verify flags exist: `python -m sglang.launch_server --help`.
4. Smoke test (default port 30000):
   ```bash
   curl localhost:30000/v1/models
   curl localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"<served-name>","messages":[{"role":"user","content":"hi"}]}'
   ```

## Validation

- `/v1/models` lists the model and chat returns tokens.
- No OOM or KV-fit warnings at startup; RadixAttention active unless disabled.

## Constraints

- Set `SGLANG_*` / `SGL_*` env vars before launch.
- Flag names are version-sensitive; verify against `--help`.

## Related

- `knowledge/sglang/server-args.md`
- `knowledge/sglang/vllm-vs-sglang.md`
- `tool:tune-sglang-config`

## Source

- `knowledge/sglang/server-args.md`
