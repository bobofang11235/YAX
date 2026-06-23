# Devmap (Version-Tagged Code Map)

The "experienced vLLM developer" data: for a given problem, which folders to
check and files to edit — resolved per vLLM **version**, because the tree moves
between releases (notably the V0→V1 engine relocation).

## Files

- `areas.jsonl` — source of truth. One area per line:
  ```json
  {"id": "scheduler", "area": "...", "symptoms": ["preemption", ...],
   "layouts": [
     {"since": "0.8.0", "folders": ["vllm/v1/core/sched/"], "files": [...], "entry": [...], "note": "V1"},
     {"since": "0.0.0", "until": "0.10.999", "folders": ["vllm/core/"], "files": [...], "note": "V0"}
   ]}
  ```
  The first layout whose `[since, until]` range contains the requested version
  wins.
- `versions.json` — version eras, the V0/V1 boundary, and the representative tags
  the index is generated for.

## Use

```bash
python3 scripts/yax.py where "<problem>" -V <vllm-version>   # default: latest
python3 scripts/yax.py where --list-areas
python3 scripts/yax.py index            # regenerates registry/codemap-by-version.json
```

## Maintenance

- This is **not** a tool/workflow/run artifact; it is developer routing data.
- When a path moves in a new vLLM release, add or bound a `layout` here and rerun
  `index`. `validate` warns when the generated index is stale.
- Keep `symptoms` rich and lowercase: they drive `where` matching.
