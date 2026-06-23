# vLLM Knowledge

vLLM-specific knowledge. For engine-agnostic roofline/estimation and the catalog
of performance factors, see `knowledge/shared/`. For SGLang, see
`knowledge/sglang/`.

## Groups

- `architecture/`: V1 engine internals, PagedAttention + KV cache, scheduler and
  continuous batching, attention backends, scheduler-vs-model-size analysis.
- `serving/`: engine arguments, environment variables, the current feature set,
  quantization, and performance tuning.
- `rocm/`: AMD/ROCm build, AITER, FP8 fnuz, TunableOp, hipBLASLt, RCCL.
- `cuda/`: NVIDIA/CUDA build, FlashInfer, DeepGEMM, CUDA graphs, NCCL.
- `development/`: repo layout, build from source, adding a model, custom ops and
  platform abstraction, contributing and testing, and the version-aware codebase
  map.

## Code Map

```bash
python3 scripts/yax.py where "<problem>" -V <vllm-version>   # default engine is vllm
python3 scripts/yax.py sync-status                            # vLLM sync state
```

## Source Of Truth

- Code: `vllm/` in the upstream repo; args in `vllm/engine/arg_utils.py`, env in
  `vllm/envs.py`. Verify with `vllm serve --help`.
- https://docs.vllm.ai/
