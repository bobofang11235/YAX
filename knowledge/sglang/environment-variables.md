# SGLang Environment Variables

## Use When

Use to change SGLang behavior without code edits: kernel/JIT toggles, profiling,
distributed, and CUDA/ROCm acceleration. Newer releases centralize env handling
in `python/sglang/srt/environ.py`; older ones read `os.environ` ad hoc.

## Lesson

SGLang reads `SGLANG_*` and `SGL_*` variables at startup. Many gate kernels or
JIT paths. Set them before launching. Verify exact names by grepping
`SGLANG_`/`SGL_` in the installed source (or reading `srt/environ.py`), as the set
changes between versions.

### Common SGLang Vars

- `SGL_ENABLE_JIT_DEEPGEMM=1` — JIT DeepGEMM FP8 GEMMs (Hopper/Blackwell); big MoE/
  FP8 throughput.
- `SGLANG_USE_MODELSCOPE=1` — pull models from ModelScope instead of HF.
- `SGLANG_TORCH_PROFILER_DIR=<dir>` — enable the torch profiler; dump traces here.
- `SGLANG_HOST_IP` / `SGLANG_SET_CPU_AFFINITY` — networking / CPU pinning for
  multi-GPU and multi-node.
- `SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1` — skip a TP memory-balance check.
- `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` — allow context beyond the model's
  trained length (accuracy risk; like vLLM `VLLM_ALLOW_LONG_MAX_MODEL_LEN`).
- AMD/ROCm: `SGLANG_USE_AITER=1` — enable AITER kernels on MI300-class GPUs.

(Confirm and discover more with `grep -rn "SGLANG_\|SGL_" python/sglang/srt/` or by
reading `srt/environ.py`.)

### External Runtime (not SGLang, but matter)

- `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` — GPU
  selection.
- `NCCL_*` (CUDA) and NCCL/RCCL (ROCm): `NCCL_DEBUG=INFO`, `NCCL_P2P_DISABLE`,
  `NCCL_IB_DISABLE`.
- `PYTORCH_TUNABLEOP_ENABLED=1` (ROCm) — GEMM autotuning via TunableOp.
- `HF_TOKEN`, `HF_HOME`, `TRANSFORMERS_OFFLINE`.

## Rules

- Set env vars before `sglang.launch_server`; many are read once at startup.
- Record every non-default env var with any benchmark/bug report — they change
  kernels and numerics silently.
- Prefer a server arg when one exists; reach for env vars for JIT/kernel toggles
  and debugging.

## Avoid

- Assuming a vLLM `VLLM_*` var exists in SGLang; the namespaces differ
  (`SGLANG_*` / `SGL_*`).
- Leaving profiler/debug vars on in production.

## Related

- `knowledge/sglang/server-args.md`
- `knowledge/sglang/performance-tuning.md`
- `knowledge/serving/environment-variables.md` (vLLM comparison)

## Source

- `python/sglang/srt/environ.py`, `python/sglang/srt/utils.py`
- https://docs.sglang.ai/
