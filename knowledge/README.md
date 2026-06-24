# Knowledge

Compact, reusable background for LLM inference-engine work. Cards explain
principles, internals, and reference data that tools and workflows cite. Retrieve
via `python3 scripts/yax.py recommend "<task>"` rather than scanning folders.

Knowledge is namespaced by engine so vLLM, SGLang, and ATOM are equally
first-class, with engine-agnostic notes kept separate.

## Layout

- `vllm/` — vLLM-specific knowledge:
  - `architecture/`: V1 engine internals, PagedAttention + KV cache, scheduler,
    attention backends.
  - `serving/`: engine arguments, environment variables, features, quantization,
    performance tuning.
  - `rocm/`: AMD/ROCm build, AITER, FP8 fnuz, TunableOp, hipBLASLt, RCCL.
  - `cuda/`: NVIDIA/CUDA build, FlashInfer, DeepGEMM, CUDA graphs, NCCL.
  - `development/`: repo layout, build, adding a model, custom ops, contributing,
    codebase map.
- `sglang/` — SGLang-specific knowledge: architecture (SRT), RadixAttention,
  server args, env vars, features, frontend DSL, performance, vLLM-vs-SGLang.
- `atom/` — ATOM (ROCm/ATOM) knowledge: architecture, AITER model ops,
  configuration, env vars, quantization (MXFP4/online), distributed/TBO,
  performance, vLLM-vs-ATOM. ROCm-only.
- `shared/` — engine-agnostic: performance estimation (roofline) and the catalog
  of performance factors.

## Maintenance

- Start new notes from `templates/knowledge-note.md`.
- Put engine-specific notes under `knowledge/<engine>/`; cross-engine notes under
  `knowledge/shared/`.
- Keep cards compact; link to upstream source paths instead of pasting long code.
- Record the engine version/commit a claim was verified against when it is
  version-sensitive.

## Search

```bash
python3 scripts/yax.py knowledge-search "paged attention"
python3 scripts/yax.py knowledge-search "radix attention" --engine sglang
python3 scripts/yax.py knowledge-search "two-batch overlap" --engine atom
```
