# ATOM Configuration (Args)

## Use When

Use when launching `atom.entrypoints.openai_server` / `atom.examples.simple_inference`
or setting config. Authoritative: `docs/configuration_guide.md`; confirm with
`python -m atom.entrypoints.openai_server --help`.

## Lesson

ATOM is configured via config classes + CLI flags. Names below are from the
README/recipes; verify against the installed version (ATOM is young and moving).

### Launch

```bash
# offline
python -m atom.examples.simple_inference --model meta-llama/Meta-Llama-3-8B --kv_cache_dtype fp8
# server (single GPU)
python -m atom.entrypoints.openai_server --model Qwen/Qwen3-0.6B --kv_cache_dtype fp8
# multi-GPU TP
python -m atom.entrypoints.openai_server --model deepseek-ai/DeepSeek-R1 --kv_cache_dtype fp8 -tp 8
```

### Key Flags

- `--model <hf-id-or-path>` — model to serve; `--trust-remote-code` for custom.
- `--kv_cache_dtype fp8` — quantize the KV cache (note: underscore style).
- `-tp N` — tensor parallel; data/expert parallel for MoE (see distributed note).
- Optimization level — piecewise torch.compile level (**default 3**); lower for
  faster startup / debugging (see `compilation` area / guide for the exact flag).
- `--method mtp --num-speculative-tokens N` — MTP speculative decoding.
- `--online_quant_config <...>` — re-quantize at load to PTPC-FP8 / MXFP4.
- `--torch-profiler-dir <dir> --mark-trace` — enable profiling/trace capture.
- Server: `--model`, host/port, served name (verify exact flag names).

### Config / Env Split

- Static behavior → CLI/config (`docs/configuration_guide.md`).
- Kernel/runtime toggles → `ATOM_*` env vars (`docs/environment_variables.md`).

## Rules

- Note ATOM's mixed flag styles (`--kv_cache_dtype` underscore vs `-tp`); copy
  from a recipe and verify with `--help`.
- First run compiles the model (can take ~10 min); expect a slow cold start.
- Don't assume vLLM/SGLang flag names; ATOM has its own (e.g. `--mem-fraction`
  may differ — verify).

## Avoid

- Copying vLLM flags verbatim; translate intent and confirm with `--help`.

## Related

- `knowledge/atom/environment-variables.md`
- `knowledge/atom/quantization.md`
- `knowledge/atom/performance-tuning.md`

## Source

- `docs/configuration_guide.md`; ATOM README/recipes
- https://github.com/ROCm/ATOM
