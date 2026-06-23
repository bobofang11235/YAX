---
id: contribute-feature
kind: workflow
title: Contribute A Feature To vLLM
status: active
tags: [development, contribute, pr, build, test, kernel, code-change]
aliases: [make a change, open a pr, add a feature, edit vllm, land a patch]
tools:
  - tool:build-from-source
  - tool:add-new-model
  - tool:benchmark-serving
triggers:
  - implement or change vLLM behavior
  - add a model, kernel, or flag
  - prepare a pull request
validation:
  - pre-commit clean and relevant pytest subset passes
  - new behavior covered by a test or A-B
  - perf claims backed by a recorded benchmark
source:
  - knowledge/development/repo-layout-and-build.md
---

# Contribute A Feature To vLLM

## Use When

Use to make a code change in vLLM and prepare it for review.

## Workflow

1. Locate the right layer (engine/scheduler/model/attention/platform) using the
   architecture and repo-layout notes.
2. Set up an editable build with `tool:build-from-source` (precompiled path for
   Python-only edits; full compile for `csrc/`).
3. Implement the change in the narrowest place; for a new model use
   `tool:add-new-model`.
4. Validate: `pre-commit run --all-files`, the relevant `pytest` subset, and a
   correctness A-B. For perf changes, capture before/after with
   `tool:benchmark-serving`.
5. Open a focused PR with motivation, tests, and recorded environment.
6. Record the change and evidence in a run.

## Tool Sequence

- `tool:build-from-source`: editable checkout matching the environment.
- `tool:add-new-model`: when the change is a new architecture.
- `tool:benchmark-serving`: evidence for performance changes.

## Validation Gates

- Format/lint clean; tests pass; evidence attached.
- Diff scoped; no unrelated reformatting.

## Update Policy

Update when a recurring contribution pattern or check should be standardized.

## Related Runs

- 

## Source

- `knowledge/development/repo-layout-and-build.md`
- `knowledge/development/contributing-and-testing.md`
- `knowledge/development/custom-ops-and-platforms.md`
