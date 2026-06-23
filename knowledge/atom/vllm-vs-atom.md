# vLLM vs ATOM (and SGLang)

## Use When

Use when deciding among engines, or mapping a concept you know in vLLM to ATOM.

## Lesson

ATOM is a **lightweight, AMD-ROCm-first** engine built on **AITER**, adapted from
nano-vllm. It is much smaller than vLLM/SGLang and intentionally AMD-focused.

### Key Differences

- **Hardware**: ATOM is **ROCm-only** (AMD). vLLM and SGLang are multi-vendor
  (NVIDIA + ROCm + more).
- **Heritage/size**: ATOM derives from **nano-vllm** — small, readable, fast-moving.
  vLLM/SGLang are large, broad frameworks.
- **Kernels**: ATOM is **AITER-first** (ASM/CK/Triton). vLLM/SGLang use AITER on
  ROCm too but default to FlashAttention/FlashInfer on CUDA.
- **Quantization**: ATOM leans into **MXFP4 / PTPC-FP8** and **online
  quantization** at load — more aggressive AMD-oriented precisions.
- **MoE scaling**: ATOM ships **Two-Batch Overlap (TBO)** + MORI all-to-all and
  P/D disaggregation (MORI-IO / Mooncake), targeting DeepSeek/GLM-class MoE on AMD.
- **Relationship**: ATOM can also run **as a vLLM plugin backend** (out-of-tree),
  so it is partly complementary to vLLM, not only an alternative.

### Term / Flag Mapping (verify with --help)

| Intent | vLLM | ATOM |
| --- | --- | --- |
| Launch server | `vllm serve <model>` | `python -m atom.entrypoints.openai_server --model <m>` |
| KV cache dtype | `--kv-cache-dtype` | `--kv_cache_dtype` |
| Tensor parallel | `--tensor-parallel-size` | `-tp` |
| Spec decode | `--speculative-config` | `--method mtp --num-speculative-tokens` |
| Disable graphs | `--enforce-eager` | lower the opt/compile level |
| Online quant | (n/a) | `--online_quant_config` |
| Env namespace | `VLLM_*` | `ATOM_*` |
| Code: runtime | `vllm/v1/` | `atom/` (+ authoritative `docs/`) |

### Choosing

- **ATOM**: AMD/ROCm deployments wanting AITER-tuned kernels, MXFP4/online quant,
  and DeepSeek/GLM-class MoE with TBO + P/D disaggregation; or to use ATOM as a
  vLLM plugin backend on AMD.
- **vLLM**: broadest model/hardware coverage and ecosystem.
- **SGLang**: heavy prefix sharing / branching programs (RadixAttention + DSL),
  DeepSeek MoE at scale.
- Benchmark on the actual target hardware before deciding.

## Rules

- Never copy flags across engines; translate intent and confirm with each
  engine's `--help`.
- For ATOM, the `docs/` guides are authoritative; internal module paths move.

## Avoid

- Considering ATOM for NVIDIA-only deployments.
- Assuming `VLLM_*` env vars or vLLM internals apply to ATOM.

## Related

- `knowledge/atom/architecture.md`
- `knowledge/sglang/vllm-vs-sglang.md`
- `knowledge/shared/performance-estimation.md`

## Source

- ATOM README (vLLM plugin backend, nano-vllm acknowledgement)
- https://github.com/ROCm/ATOM
