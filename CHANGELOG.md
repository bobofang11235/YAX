# YAX Changelog

This log tracks YAX content updates **and the upstream vLLM point each update
reflects**. The machine-readable marker lives in `devmap/sync-state.json`
(`synced_to`). To refresh knowledge next time, review only vLLM commits newer
than the latest entry's `vLLM synced to` ref — you do not need to re-read the
whole history.

## How To Update YAX From Upstream

1. Read the current marker:
   ```bash
   python3 scripts/yax.py sync-status            # prints synced_to + the diff command
   python3 scripts/yax.py sync-status --vllm-path <vllm-clone>   # also runs the diff
   ```
2. List what changed since the last sync (clone vLLM into the workspace if absent):
   ```bash
   git -C <vllm-clone> fetch --tags
   git -C <vllm-clone> log --oneline <synced_to>..origin/main
   ```
3. For relevant changes, update the narrowest artifact:
   - new/renamed flags -> `knowledge/serving/engine-args-reference.md`
   - new/renamed env vars -> `knowledge/serving/environment-variables.md`
   - moved file paths -> add a version-bounded layout in `devmap/areas.jsonl`
   - new features -> `knowledge/serving/features-overview.md`
4. Bump `synced_to` / `synced_on` in `devmap/sync-state.json` to the new ref.
5. Regenerate + verify:
   ```bash
   python3 scripts/yax.py index && python3 scripts/yax.py validate && python3 scripts/yax.py eval
   ```
6. Add a dated entry below with `vLLM synced to: <ref>`.

## Format

Each entry: date, the upstream ref reviewed, then Added / Changed / Fixed bullets.
Newest first.

---

## [Unreleased]

- (pending changes go here)

## 2026-06-23 — Add SGLang as a second engine

- vLLM synced to: **v0.23.0**; SGLang synced to: **v0.5.13** (2026-06-13).
- Made YAX multi-engine: `scripts/yax.py where` / `sync-status` take `--engine
  vllm|sglang`; `index` generates a per-engine code-map index.
- Added SGLang code map: `devmap/sglang-areas.jsonl` (19 areas),
  `devmap/sglang-versions.json`, `devmap/sglang-sync-state.json`.
- Added `knowledge/sglang/`: architecture, RadixAttention, server args, env vars,
  features, frontend DSL, performance tuning, and a vLLM-vs-SGLang comparison.
- Added SGLang tool cards (`launch-sglang-server`, `tune-sglang-config`), the
  `serve-sglang-model` workflow, and golden routing eval cases.

## 2026-06-23 — Performance analysis + estimator

- vLLM synced to: **v0.23.0** (analysis/tooling only; no new upstream review)
- Added: `knowledge/architecture/scheduler-model-size-impact.md` — how the
  scheduler interacts with small/medium/large models and whether scheduling
  changes output.
- Added: `knowledge/serving/performance-factors.md` — full catalog of performance
  levers beyond the scheduler.
- Added: `knowledge/serving/performance-estimation.md` + `scripts/perfcalc.py` —
  rough memory/throughput/TTFT estimates computed from a model config.

## 2026-06-23 — Version-tagged code-map indexer

- vLLM synced to: **v0.23.0** (no new upstream review; tooling change only)
- Added: `devmap/areas.jsonl` (30 version-tagged code-map areas) and
  `devmap/versions.json`.
- Added: `scripts/yax.py where` — version-aware "which folders/files to
  check/edit" indexer; `codemap-index` generation to
  `registry/codemap-by-version.json`; staleness check in `validate`.
- Added: `knowledge/development/codebase-map.md`.

## 2026-06-23 — Initial vLLM knowledge base

- vLLM synced to: **v0.23.0** (2026-06-15; `main` last pushed 2026-06-17)
- Added: YAX harness (runtime, templates, AGENTS/README, registry, evals).
- Added: knowledge for architecture (V1 engine, PagedAttention/KV cache,
  scheduler, attention backends), serving (engine args, env vars, features,
  quantization, performance), ROCm (setup/tuning, FP8 fnuz vs OCP), CUDA
  (setup/tuning), and development (repo layout/build, adding a model, custom
  ops/platforms, contributing/testing).
- Added: 7 tool cards and 5 workflow cards covering serving, tuning,
  benchmarking, building, model bringup, accuracy/OOM debugging, contributing,
  and ROCm bringup.
- Note: knowledge targets the **V1 engine era**; V0 paths are retained in the
  code map for users on <= 0.10.x checkouts.
