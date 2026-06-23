# YAX Agent Guide

YAX is an LLM inference-engine engineering toolbox plus compact knowledge base,
covering **vLLM** and **SGLang**. Start from retrieval, load the smallest useful
tools/workflows, execute with validation, and save reusable learning back into
the toolbox. Engine-specific knowledge is under `knowledge/<engine>/`; the code
map is per engine via `--engine`.

## Retrieval Order

1. For unfamiliar or non-trivial vLLM tasks, run
   `python3 scripts/yax.py recommend "<task>"`.
2. Read the top-ranked workflow first when one matches.
3. Read only the tool cards referenced by that workflow or recommendation.
4. Read linked `knowledge/` notes only when deeper background is needed.
5. Read `capture/` only when raw evidence or task history is required.

## Common Routes

- Toolbox search or new task setup:
  `python3 scripts/yax.py recommend "<task>"`
- Tool / workflow catalogs:
  `tools/README.md`, `workflows/README.md`
- vLLM architecture and request lifecycle (V1 engine, scheduler, KV cache):
  `knowledge/architecture/README.md`
- Engine arguments (`EngineArgs` / `vllm serve` flags):
  `knowledge/serving/engine-args-reference.md`
- Environment variables (`VLLM_*` and runtime knobs):
  `knowledge/serving/environment-variables.md`
- What features exist today (quantization, spec decode, LoRA, structured output):
  `knowledge/serving/features-overview.md`
- Throughput / latency / memory tuning:
  `knowledge/serving/performance-tuning.md`
- ROCm / AMD GPU build and tuning:
  `knowledge/rocm/rocm-setup-and-tuning.md`
- CUDA / NVIDIA GPU build and tuning:
  `knowledge/cuda/cuda-setup-and-tuning.md`
- How to edit the codebase, build from source, add a model, run tests:
  `knowledge/development/README.md`
- Which folders/files to check or edit for a problem (version-aware code map):
  `python3 scripts/yax.py where "<problem>" -V <version>` (add `--engine sglang`
  for SGLang) and `knowledge/development/codebase-map.md`
- SGLang (architecture, RadixAttention, server args, env, DSL, vs vLLM):
  `knowledge/sglang/README.md`
- What vLLM version YAX reflects + how to pull newer changes:
  `python3 scripts/yax.py sync-status`, `CHANGELOG.md`, `devmap/sync-state.json`
- Understanding the repository design:
  `README.md`
- Maintaining YAX itself:
  `AGENTS.md`, folder `README.md` files, `templates/`

## Task Loop

1. Describe the task as a short search query.
2. Retrieve matching workflows, tools, and prior runs.
3. Choose the closest workflow or compose the top tools into a plan.
4. Execute with the workflow's validation gates.
5. Leave resumable state: validation evidence, remaining gaps, next start point.
6. At the end of a meaningful task, create or update a `runs/` record.
7. If a new tool, better sequence, or repeated failure appeared, update the
   relevant tool/workflow/knowledge note and regenerate the registry.

## Context Budget Rules

- Prefer registry search over scanning folders.
- Prefer one matching workflow card over many tool cards.
- Prefer promoted knowledge over raw capture notes.
- Do not load all tools or workflows by default.

## Developing An Engine: Route By Version

YAX helps **develop vLLM and SGLang**, where file layouts change between
versions. Before reading code, ask the code map where to look:

```bash
python3 scripts/yax.py where "<problem>" -V 0.8.5                  # vLLM (default)
python3 scripts/yax.py where "<problem>" --engine sglang -V 0.5.13 # SGLang
```

It returns the folders/files/entry-points for that engine+version's layout
(vLLM V0/V1 and SGLang pre/post-0.4 relocations handled automatically). Confirm
exact paths against the checkout. Add or fix areas in `devmap/areas.jsonl`
(vLLM) or `devmap/sglang-areas.jsonl` and rerun `python3 scripts/yax.py index`
when a path has moved.

## vLLM Engineering Rules

- **Pin the environment.** Record vLLM version/commit, PyTorch, CUDA or ROCm,
  driver, GPU arch (SM_xx or gfxNNN). Behavior and flags are version-sensitive.
- **Confirm flags at the source.** Names come from `vllm/engine/arg_utils.py`
  (args) and `vllm/envs.py` (env vars). Verify with `vllm serve --help` on the
  installed version before relying on a flag.
- **Separate correctness from performance.** Fix accuracy/divergence first with a
  trusted baseline, then tune throughput/latency.
- **Validate with evidence.** A serving or kernel change is not done until a
  benchmark (`vllm bench serve` / `benchmarks/`), an accuracy check, or the
  relevant `pytest` has run, or the gap is recorded.
- **Isolate the variable in A/B work.** Change one of {dtype, kv-cache-dtype,
  attention backend, quantization, TP/PP, batch size} at a time.
- **Treat ROCm vs CUDA divergence as backend work** unless evidence shows a
  model or config bug. FP8 fnuz (ROCm) vs OCP (CUDA) formats differ.

## Related Repo Rules

- Record related repositories by GitHub URL, not a host-specific checkout path.
- If vLLM is needed locally but absent, clone it into the workspace first, then
  use repo-relative paths from that clone.

## Maintenance Rules

When developing YAX itself:

1. Preserve the main loop: `task -> retrieve -> execute -> run -> update`.
2. Put raw sources and incomplete material in `capture/`.
3. Put reusable capabilities in `tools/` and recurring procedures in `workflows/`.
4. Put task lineage and outcomes in `runs/`.
5. Put background principles and distillations in `knowledge/`.
6. Regenerate `registry/toolbox-index.json` after changing tools/workflows/runs.
7. Keep provenance: cards link to upstream source files or motivating runs.
8. Do not make raw dumps or long articles default agent context.
9. Run `python3 scripts/yax.py validate` after structural changes.
10. Run `python3 scripts/yax.py eval` after changing card metadata or evals;
    confirm the target query improved and no other case regressed.
11. When syncing knowledge to a newer vLLM: start from
    `python3 scripts/yax.py sync-status`, review only commits newer than
    `synced_to`, then bump `devmap/sync-state.json` and add a `CHANGELOG.md`
    entry recording the new `vLLM synced to` ref.
