# Contributing And Testing

## Use When

Use before opening a vLLM PR or when running its tests/linters locally.

## Lesson

vLLM enforces formatting/linting via pre-commit and runs a large pytest suite in
CI. Match local checks to CI to avoid red builds.

### Format / Lint

- Install hooks once: `pip install pre-commit && pre-commit install`.
- Run on demand: `pre-commit run --all-files`. This runs ruff (lint + format),
  isort/import rules, mypy (typed modules), and misc checks.
- Keep edits minimal and in-style; do not reformat unrelated code.

### Tests

- Run a focused subset, not the whole suite, while iterating:
  ```bash
  pytest tests/<area>/test_<thing>.py -k <name> -x
  ```
- GPU tests need a real GPU; many are sharded in CI. Mirror the package path:
  changes in `vllm/v1/core/` → `tests/v1/core/`.
- For model correctness, compare vLLM output/logits against HF Transformers on
  fixed prompts with greedy decoding.
- For kernels, test against the `forward_native` reference and check tolerances.

### Benchmarks As Evidence

- Performance PRs should include before/after from `vllm bench serve` or
  `benchmarks/`, with GPU arch, versions, backend, and flags recorded.

### PR Norms

- One focused change per PR; clear title and motivation.
- Add/adjust tests for new behavior; note any skipped checks and why.
- Link the issue/RFC; flag user-facing flag/env changes in the description.

## Rules

- Run `pre-commit run --all-files` and the relevant `pytest` subset before pushing.
- Provide evidence (test pass, benchmark, or accuracy A-B) — not just confidence.
- Keep the diff scoped; unrelated reformatting gets PRs rejected.

## Avoid

- Pushing without running pre-commit (CI will fail on format/lint).
- Claiming a perf win without a reproducible benchmark and recorded environment.

## Related

- `knowledge/development/repo-layout-and-build.md`
- `knowledge/serving/performance-tuning.md`
- `tools/serving/benchmark-serving.md`

## Source

- vLLM `.pre-commit-config.yaml`, `tests/`, `pyproject.toml`
- https://docs.vllm.ai/en/latest/contributing/overview.html
