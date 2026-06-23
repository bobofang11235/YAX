# SGLang Feature Overview

## Use When

Use to check whether SGLang supports something today and the flag to enable it.

## Lesson

### Core Throughput / Memory

- **RadixAttention** — automatic tree-based KV reuse (default; `--disable-radix-cache`
  to turn off). See `radixattention.md`.
- **Zero-overhead / overlap scheduler** — CPU scheduling overlaps GPU compute
  (`--disable-overlap-schedule` to turn off).
- **Chunked prefill** — `--chunked-prefill-size`.
- **CUDA graphs** — on by default (`--disable-cuda-graph`); Breakable CUDA Graphs
  (BCG) speed prefill in newer releases. `--enable-torch-compile` for compile.

### Models

- Decoder LLMs (Llama, Qwen, Mistral/Mixtral, Gemma, Phi, DeepSeek, GPT-OSS,
  Command-R, etc.), **MoE** (DeepSeek-V2/V3, Mixtral, Qwen-MoE) with EP / DP
  attention, and **multimodal** (Qwen-VL, LLaVA-OneVision, InternVL, Pixtral...).
- SGLang-Diffusion extends to image/video diffusion models (separate track).

### Decoding / Sampling

- Sampling (temperature, top-p/k, min-p, penalties, logprobs, seed).
- **Structured/constrained output** — `--grammar-backend xgrammar|outlines|
  llguidance`; SGLang pioneered fast compressed-FSM structured decoding.
- **Tool / function calling** — `--tool-call-parser`; **reasoning parser** —
  `--reasoning-parser`.
- **Speculative decoding** — EAGLE/EAGLE3; **SpecDecode V2 with tree drafting
  (topk>1)** is default in 0.5.13.

### Quantization

- FP8 (incl. blockwise), AWQ, GPTQ, INT8/W8A8, Marlin kernels, DeepGEMM FP8
  (`SGL_ENABLE_JIT_DEEPGEMM`); KV cache FP8 via `--kv-cache-dtype`.

### Scale / Serving

- TP / DP / EP, multi-node, **large-scale expert parallelism** for DeepSeek.
- **Disaggregated prefill/decode (P/D)** and KV transfer.
- **Cache-aware load balancer / router** (sgl-router) for multi-instance.
- OpenAI-compatible API + a native `/generate` endpoint, plus the **frontend DSL**
  (`frontend-language.md`).
- Hardware: NVIDIA (incl. Blackwell GB200/GB300), AMD ROCm (AITER), and others.

## Rules

- Confirm a model is supported in `python/sglang/srt/models/` (or docs) before
  debugging "unsupported architecture".
- Feature availability is version-dependent; check the release notes for the
  installed version.

## Avoid

- Assuming feature flags match vLLM names; see `vllm-vs-sglang.md`.

## Related

- `knowledge/sglang/server-args.md`
- `knowledge/sglang/frontend-language.md`
- `knowledge/sglang/vllm-vs-sglang.md`

## Source

- `python/sglang/srt/models/`, release notes
- https://docs.sglang.ai/
