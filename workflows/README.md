# Workflows

Reusable multi-tool task plans for recurring work. Each card names its triggers,
tool sequence, and validation gates. Retrieve via
`python3 scripts/yax.py recommend "<task>"`. Workflows are namespaced by engine.

## Catalog

### vLLM (`workflows/vllm/`)

- `serve-new-model.md`: bring a model up behind the OpenAI API.
- `optimize-throughput.md`: raise throughput / cut latency with evidence.
- `debug-accuracy.md`: isolate wrong/low-quality output to one variable.
- `contribute-feature.md`: make and land a code change in vLLM.
- `rocm-bringup.md`: get a model correct and fast on AMD/ROCm.

### SGLang (`workflows/sglang/`)

- `serve-sglang-model.md`: bring a model up on SGLang and tune it.

### ATOM (`workflows/atom/`)

- `serve-atom-model.md`: bring a model up on ATOM (AMD/ROCm, AITER) and tune it.

## Maintenance

- Start from `templates/workflow-note.md`; reference only existing tool ids.
- Put engine-specific workflows under `workflows/<engine>/`; tag with the engine.
- Run `python3 scripts/yax.py index` and `python3 scripts/yax.py validate` after
  changes.
