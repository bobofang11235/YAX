# Development Knowledge

How to read, build, edit, extend, and test the vLLM codebase.

## Notes

- `codebase-map.md`: experienced-dev routing — which folders/files to check/edit
  for a problem, **resolved per vLLM version** via `scripts/yax.py where`.
- `repo-layout-and-build.md`: directory map and building/installing from source
  (editable, Python-only fast path, full CUDA/ROCm build).
- `adding-a-model.md`: register and implement a new model architecture.
- `custom-ops-and-platforms.md`: custom CUDA/HIP/Triton ops and the platform
  abstraction.
- `contributing-and-testing.md`: formatting, pre-commit, running tests, PR norms.

## Start Here For A Code Change

```bash
python3 scripts/yax.py where "<your problem>" -V <your vllm version>
```

## Source Of Truth

- The vLLM repo itself; clone https://github.com/vllm-project/vllm into the
  workspace if not present, then use repo-relative paths.
- https://docs.vllm.ai/en/latest/contributing/
