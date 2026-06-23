# ATOM Feature Overview

## Use When

Use to check whether ATOM supports something today and the flag/guide for it.

## Lesson

### Core

- **ROCm-optimized, AITER kernels** (ASM / CK / Triton) — the central design.
- **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/completions`) + profiler
  endpoints (`/start_profile`, `/stop_profile`).
- **Piecewise torch.compile** — 4 compilation levels with CUDA-graph capture for
  low-latency decode (default level 3).
- **Prefix caching** — xxhash64-based KV block sharing across sequences.

### Parallelism / Scale

- **TP / DP / EP** with **MORI all-to-all**.
- **Two-Batch Overlap (TBO)** — split a batch into two micro-batches to overlap
  expert-parallel comms with compute (DeepSeek-style).
- **P/D disaggregation** — prefill/decode on separate nodes with RDMA KV transfer
  (MORI-IO; Mooncake push-mode), incl. DeepSeek-V4-Pro.

### Quantization

- FP8, MXFP4, INT8, INT4 with auto-detection from HF configs.
- **Online quantization** — re-quantize to PTPC-FP8 / MXFP4 at load via
  `--online_quant_config` (no offline re-packing). KV cache FP8 via
  `--kv_cache_dtype`.

### Decoding

- **Speculative decoding** — Multi-Token Prediction (MTP) with an EAGLE proposer
  (`--method mtp --num-speculative-tokens N`).

### Models

- Llama 2/3/3.1, Qwen3 (+ MoE, + Next/GDN), DeepSeek V2/V3 (MLA, MTP), Mixtral,
  GLM-4-MoE, GLM-5/5.2 (MLA + DSA sparse attention; IndexShare), GPT-OSS (sliding
  window + attention sinks), Kimi-K2, MiMo-V2 (hybrid full+SWA). Multimodal:
  Qwen3.5 image+text.
- Hardware: AMD Instinct (MI300-class) and **experimental Navi 4 (RDNA4/gfx1201)**
  — RX 9070 / R9700.

### Integration

- **vLLM plugin backend** — ATOM as an out-of-tree plugin backend for vLLM.

## Rules

- Confirm a model in `atom/models/` (or `docs/model_support_guide.md`) before
  debugging "unsupported architecture".
- Feature set moves fast and is ROCm-only; check the README "News" and `docs/`.

## Avoid

- Expecting NVIDIA/CUDA support — ATOM targets ROCm.

## Related

- `knowledge/atom/quantization.md`
- `knowledge/atom/distributed-tbo.md`
- `knowledge/atom/aiter-model-ops.md`

## Source

- ATOM README + `docs/`; https://github.com/ROCm/ATOM
