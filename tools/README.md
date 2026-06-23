# Tools

Canonical tool cards: compact, reusable capabilities with metadata for the YAX
runtime. Use `python3 scripts/yax.py recommend "<task>"` or
`python3 scripts/yax.py search "<task>"` instead of scanning this folder.

Tools are namespaced by engine. Refs are id-based (`tool:<id>`), so the folder is
only for human organization.

## Groups

- `vllm/`: vLLM tools
  - `serving/`: launch, benchmark, and tune a vLLM server.
  - `development/`: build from source, add a model.
  - `debugging/`: OOM/KV triage, accuracy divergence triage.
- `sglang/`: SGLang tools (launch server, tune config).
- `atom/`: ATOM tools (launch server, tune config) — AMD/ROCm, AITER.

## Maintenance

- Start from `templates/tool-note.md`.
- Put engine-specific tools under `tools/<engine>/`; tag with the engine.
- Keep each card compact; link to `knowledge/` instead of copying.
- Run `python3 scripts/yax.py index` and `python3 scripts/yax.py validate` after
  changes.
