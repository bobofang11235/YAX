# Tools

Canonical tool cards: compact, reusable vLLM capabilities with metadata for the
YAX runtime. Use `python3 scripts/yax.py recommend "<task>"` or
`python3 scripts/yax.py search "<task>"` instead of scanning this folder.

## Groups

- `serving/`: launch, benchmark, and tune a vLLM server.
- `development/`: build from source, add a model.
- `debugging/`: OOM/KV triage, accuracy divergence triage.

## Maintenance

- Start from `templates/tool-note.md`.
- Keep each card compact and actionable; link to `knowledge/` instead of copying.
- Run `python3 scripts/yax.py index` and `python3 scripts/yax.py validate` after
  changes.
