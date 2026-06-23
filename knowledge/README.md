# Knowledge

Compact, reusable vLLM background. Cards explain principles, internals, and
reference data that tools and workflows cite. Retrieve via
`python3 scripts/yax.py recommend "<task>"` rather than scanning folders.

## Groups

- `architecture/`: V1 engine internals, PagedAttention + KV cache, scheduler and
  continuous batching, attention backends.
- `serving/`: engine arguments, environment variables, the current feature set,
  quantization, and performance tuning.
- `rocm/`: AMD/ROCm build, AITER, FP8 fnuz, TunableOp, hipBLASLt, RCCL.
- `cuda/`: NVIDIA/CUDA build, FlashInfer, DeepGEMM, CUDA graphs, NCCL.
- `development/`: repo layout, build from source, adding a model, custom ops and
  platform abstraction, contributing and testing.

## Maintenance

- Start new notes from `templates/knowledge-note.md`.
- Keep cards compact; link to upstream `vllm/...` source paths instead of pasting
  long code.
- Record the vLLM version/commit a claim was verified against when it is
  version-sensitive.
