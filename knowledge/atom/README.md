# ATOM Knowledge

Compact knowledge for developing and operating
[ATOM](https://github.com/ROCm/ATOM) — "AiTer Optimized Model", a lightweight
**vLLM-like inference engine for AMD ROCm**, built on **AITER** kernels and
adapted from nano-vllm. ATOM is the third engine in YAX and follows the same
harness shape (knowledge + version-tagged code map + sync state).

ATOM is **ROCm-only** (AMD GPUs). If you need NVIDIA, use vLLM or SGLang.

## Notes

- `architecture.md`: engine design, request lifecycle, component map.
- `configuration.md`: config classes / CLI arguments (`atom.entrypoints.openai_server`).
- `environment-variables.md`: `ATOM_*` and ROCm/AITER runtime vars.
- `features-overview.md`: what ATOM supports today.
- `aiter-model-ops.md`: AITER kernel integration (ASM/CK/Triton) — ATOM's core.
- `quantization.md`: FP8/MXFP4/INT8/INT4 and online quantization.
- `distributed-tbo.md`: TP/DP/EP, MORI all-to-all, Two-Batch Overlap, P/D disagg.
- `performance-tuning.md`: tuning order for ATOM.
- `vllm-vs-atom.md`: how ATOM relates to vLLM (and SGLang).

## Code Map

```bash
python3 scripts/yax.py where "<problem>" --engine atom -V 0.1.5
python3 scripts/yax.py where --engine atom --list-areas
python3 scripts/yax.py sync-status --engine atom
```

## Source Of Truth

- Code: `atom/` in the upstream repo; **authoritative docs under `docs/`**
  (architecture, configuration, model support, model ops, scheduling, compilation,
  distributed, serving, environment variables).
- Confirm CLI flags with `python -m atom.entrypoints.openai_server --help`.
- https://rocm.github.io/ATOM/docs , https://github.com/ROCm/ATOM
