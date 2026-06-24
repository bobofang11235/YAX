# Devmap (Version-Tagged Code Maps)

The "experienced developer" data: for a given problem, which folders to check and
files to edit — resolved per engine **version**, because the tree moves between
releases (vLLM V0→V1; SGLang pre/post-0.4). One code map per engine.

## Files

Per engine `<engine>` in {`vllm`, `sglang`, `atom`}:

- `<engine>-areas.jsonl` — source of truth. One area per line:
  ```json
  {"id": "scheduler", "area": "...", "symptoms": ["preemption", ...],
   "layouts": [
     {"since": "0.8.0", "folders": ["vllm/v1/core/sched/"], "files": [...], "note": "V1"},
     {"since": "0.0.0", "until": "0.10.999", "folders": ["vllm/core/"], "files": [...], "note": "V0"}
   ]}
  ```
  The first layout whose `[since, until]` range contains the requested version
  wins.
- `<engine>-versions.json` — version eras and the representative tags the index is
  generated for.
- `<engine>-sync-state.json` — the upstream release this map reflects (see
  `scripts/yax.py sync-status -e <engine>`).

Current engines: `vllm-*`, `sglang-*`, and `atom-*` (ROCm/ATOM). ATOM is young;
its area paths are best-effort and each links the authoritative `docs/` guide.

## Use

```bash
python3 scripts/yax.py where "<problem>" --engine vllm  -V <version>   # default: latest
python3 scripts/yax.py where "<problem>" --engine sglang -V 0.5.13
python3 scripts/yax.py where "<problem>" --engine atom   -V 0.1.5
python3 scripts/yax.py where --engine sglang --list-areas
python3 scripts/yax.py index   # regenerates registry/<engine>-codemap-by-version.json
```

## Maintenance

- This is **not** a tool/workflow/run artifact; it is developer routing data.
- When a path moves in a new release, add or bound a `layout` in the right
  `<engine>-areas.jsonl` and rerun `index`. `validate` warns when an index is
  stale.
- To add a new engine, create `<engine>-areas.jsonl` / `-versions.json` /
  `-sync-state.json`, add the name to `ENGINES` in `scripts/yax.py`, and rerun
  `index`.
- Keep `symptoms` rich and lowercase: they drive `where` matching.
