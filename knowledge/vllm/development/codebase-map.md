# vLLM Codebase Map (Experienced-Dev Routing)

## Use When

Use when you have a vLLM problem or change in mind and need to know **which
folders to check and which files to edit** — fast, like an engineer who has
worked in the tree before. Especially use it when developing on a specific vLLM
**version**, because the layout moves between releases.

## Lesson

You rarely need to read the whole repo. Most vLLM work maps to one of ~30 areas,
each with a small set of folders/files and entry symbols. The catch: paths are
**version-dependent**. The biggest shift is the **V0 → V1 engine** relocation
(V1 default since 0.8.x): scheduler, KV cache, engine, worker, attention, and
sampler moved under `vllm/v1/`, while the legacy tree lingered until ~0.11.

So YAX ships a **version-tagged code map** instead of a flat list. You give it a
problem and your version tag; it resolves the right paths for that version.

### Use The Indexer

```bash
# Which files to touch for a problem, for your version:
python3 scripts/yax.py where "requests get preempted and throughput collapses" -V 0.8.5
python3 scripts/yax.py where "add a new model architecture"            # default: latest
python3 scripts/yax.py where "fp8 quantization wrong numbers" -V 0.9.2
python3 scripts/yax.py where --list-areas                              # all areas
```

Output gives, per matching area: folders to check, files to edit, entry-point
symbols, and a note — already resolved to the requested version's layout.

### Data + Generated Index

- Source of truth: `devmap/vllm-areas.jsonl` (one area per line, with version-bounded
  `layouts`) and `devmap/vllm-versions.json` (eras + representative tags).
- Generated: `registry/vllm-codemap-by-version.json` — the resolved map for every
  representative tag (`0.7.3, 0.8.5, 0.10.1, 0.11.0, 0.23.0, latest`). Rebuilt by
  `python3 scripts/yax.py index` (or `codemap-index`).
- SGLang has its own map (`devmap/sglang-areas.jsonl`); query it with
  `--engine sglang`.

### How Version Resolution Works

Each area lists layouts with optional `since`/`until`. The indexer parses the
requested tag (`0.8.5` → `(0,8,5)`; `latest`/`main` → newest) and picks the first
layout whose range contains it. Example — the scheduler area:

- `>= 0.8.0` → `vllm/v1/core/sched/scheduler.py` (V1)
- `<= 0.10.x` → `vllm/core/scheduler.py` (V0)

### Stable vs Moved Areas (quick orientation)

- Mostly **stable** paths: `vllm/engine/arg_utils.py` (flags), `vllm/envs.py`
  (env vars), `vllm/model_executor/models/` (models), `.../layers/quantization/`,
  `vllm/platforms/`, `vllm/distributed/`, `vllm/entrypoints/openai/`, `csrc/`.
- **Moved at V0→V1**: scheduler, KV cache, engine core, worker/runner, executor,
  attention backends, sampler, spec-decode, structured output, detokenizer.
- **Split later**: `vllm/config.py` → `vllm/config/` package (~0.10+).

## Rules

- Always pass `-V <version>` when developing against a pinned vLLM; the default is
  `latest` (V1) and will mislead on older checkouts.
- Treat the map as a **starting point**, then confirm against the actual checkout
  (`git grep`, `ls`); a file may have moved since the area was recorded.
- When you discover a path that moved or a missing area, update
  `devmap/vllm-areas.jsonl` (add a version-bounded layout) and rerun
  `python3 scripts/yax.py index`.
- This is developer knowledge, not a runnable capability — it routes you to code;
  it does not change it.

## Avoid

- Hardcoding `vllm/v1/...` advice for a user on 0.7.x (those paths do not exist
  there).
- Letting the generated `codemap-by-version.json` drift; `validate` warns when it
  is stale.

## Related

- `knowledge/vllm/development/repo-layout-and-build.md`
- `knowledge/vllm/architecture/vllm-architecture-overview.md`
- `devmap/vllm-areas.jsonl`, `devmap/vllm-versions.json`

## Source

- `devmap/vllm-areas.jsonl` (version-tagged), `scripts/yax.py` (`where` command)
- vLLM repo; https://docs.vllm.ai/en/latest/design/
