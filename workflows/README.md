# Workflows

Reusable multi-tool task plans for recurring vLLM work. Each card names its
triggers, tool sequence, and validation gates. Retrieve via
`python3 scripts/yax.py recommend "<task>"`.

## Catalog

- `serve-new-model.md`: bring a model up behind the OpenAI API.
- `optimize-throughput.md`: raise throughput / cut latency with evidence.
- `debug-accuracy.md`: isolate wrong/low-quality output to one variable.
- `contribute-feature.md`: make and land a code change in vLLM.
- `rocm-bringup.md`: get a model correct and fast on AMD/ROCm.

## Maintenance

- Start from `templates/workflow-note.md`; reference only existing tool ids.
- Run `python3 scripts/yax.py index` and `python3 scripts/yax.py validate` after
  changes.
