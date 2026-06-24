# YAX Changelog

This log tracks YAX content updates **and the upstream point each engine's
update reflects**. The machine-readable markers live in
`devmap/<engine>-sync-state.json` (`synced_to`). To refresh knowledge, review
only commits newer than the latest entry's synced ref — no need to re-read the
whole history.

## How To Update YAX From Upstream

For `<engine>` in {`vllm`, `sglang`, `atom`}:

1. Read the current marker:
   ```bash
   python3 scripts/yax.py sync-status --engine <engine>
   python3 scripts/yax.py sync-status --engine <engine> --repo-path <clone>  # runs the diff
   ```
2. List what changed since the last sync (clone the engine repo if absent):
   ```bash
   git -C <clone> fetch --tags
   git -C <clone> log --oneline <synced_to>..origin/main
   ```
3. For relevant changes, update the narrowest artifact:
   - new/renamed flags -> `knowledge/<engine>/...` serving/server-args note
   - new/renamed env vars -> the engine's environment-variables note
   - moved file paths -> add a version-bounded layout in `devmap/<engine>-areas.jsonl`
   - new features -> the engine's features-overview note
4. Bump `synced_to` / `synced_on` in `devmap/<engine>-sync-state.json`.
5. Regenerate + verify:
   ```bash
   python3 scripts/yax.py index && python3 scripts/yax.py validate && python3 scripts/yax.py eval
   ```
6. Add a dated entry below with `<engine> synced to: <ref>`.

## Format

Each entry: date, the upstream ref reviewed, then Added / Changed / Fixed bullets.
Newest first.

---

## [Unreleased]

- (pending changes go here)

## 2026-06-24 — ATOMesh orchestration note

- Added `knowledge/atom/atomesh-orchestration.md`: ATOMesh as a cluster control
  plane above engines (request routing, prefill/decode disaggregation, KV-aware
  scheduling) that delegates execution to ATOM/vLLM/SGLang; clarifies the
  cluster-level vs engine-level scheduling split. Cross-linked from the ATOM hub
  and distributed note. Source: ROCm Blogs ATOMesh/ATOM, MI355X article.

## 2026-06-24 — Improve multi-engine ergonomics

- Added direct `knowledge-search` and a generated `registry/knowledge-index.json`
  so agents can retrieve knowledge notes without routing through a tool/workflow.
- Added engine-aware scaffolding: `new-tool <engine> <id>` creates
  `tools/<engine>/custom/<id>.md`; `new-workflow <engine> <id>` creates
  `workflows/<engine>/<id>.md`.
- Added `sync-status --all` for a single view of all engine sync baselines.
- Added cross-engine eval cases that distinguish vLLM, SGLang, and ATOM-specific
  concepts (PagedAttention vs RadixAttention vs TBO/MXFP4/online quant).
- Fixed stale two-engine wording in docs/runtime.

## 2026-06-23 — Add ATOM (ROCm/ATOM) as a third engine

- ATOM synced to: **v0.1.5** (2026-06-22). Registered `atom` in `ENGINES`.
- ATOM code map: `devmap/atom-areas.jsonl` (14 areas), `atom-versions.json`,
  `atom-sync-state.json`. Areas link the authoritative `docs/` guides since ATOM
  is young and internal module paths are best-effort.
- `knowledge/atom/`: architecture, configuration, environment variables, features,
  AITER model ops, quantization (MXFP4/online), distributed + TBO + P/D disagg,
  performance tuning, and a vLLM-vs-ATOM comparison.
- ATOM tools (`launch-atom-server`, `tune-atom-config`), the `serve-atom-model`
  workflow, and golden routing eval cases.
- Updated all navigation docs to present vLLM / SGLang / ATOM equally.

## 2026-06-23 — Symmetric multi-engine structure

- Reorganized for engine symmetry: `knowledge/{vllm,sglang,shared}`,
  `tools/{vllm,sglang}`, `workflows/{vllm,sglang}`. Moved engine-agnostic
  perf-estimation/perf-factors notes to `knowledge/shared/`.
- Uniform code-map filenames: `devmap/<engine>-areas.jsonl` (renamed the vLLM
  files from the unprefixed names) and `registry/<engine>-codemap-by-version.json`.
- Simplified `scripts/yax.py` engine resolution to uniform per-engine paths; all
  navigation READMEs (root, knowledge, tools, workflows, devmap, index) updated to
  present vLLM and SGLang equally.
- All file moves done with `git mv` (history preserved); validate + eval still PASS.

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
- Added: `knowledge/vllm/architecture/scheduler-model-size-impact.md` — how the
  scheduler interacts with small/medium/large models and whether scheduling
  changes output.
- Added: `knowledge/shared/performance-factors.md` — full catalog of performance
  levers beyond the scheduler.
- Added: `knowledge/shared/performance-estimation.md` + `scripts/perfcalc.py` —
  rough memory/throughput/TTFT estimates computed from a model config.

## 2026-06-23 — Version-tagged code-map indexer

- vLLM synced to: **v0.23.0** (no new upstream review; tooling change only)
- Added: `devmap/vllm-areas.jsonl` (30 version-tagged code-map areas) and
  `devmap/vllm-versions.json`.
- Added: `scripts/yax.py where` — version-aware "which folders/files to
  check/edit" indexer; `codemap-index` generation to
  `registry/vllm-codemap-by-version.json`; staleness check in `validate`.
- Added: `knowledge/vllm/development/codebase-map.md`.

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
