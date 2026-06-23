# Environment Variables Reference

## Use When

Use when behavior must change without a code edit: selecting backends, tuning
distributed/runtime, debugging, or switching CUDA vs ROCm acceleration. Env vars
complement engine args; the canonical list is `vllm/envs.py`.

## Lesson

vLLM reads `VLLM_*` environment variables at process start. They override
defaults that are not exposed (or not cleanly exposed) as engine args. External
libraries (PyTorch, NCCL/RCCL, CUDA/HIP) add more knobs that affect vLLM. Set
them before launching; many are read once.

### Core vLLM (`VLLM_*`)

- `VLLM_USE_V1=1|0` — V1 engine (default 1). `0` forces legacy V0 (deprecated).
- `VLLM_ATTENTION_BACKEND` — force backend: `FLASH_ATTN`, `FLASHINFER`, `XFORMERS`,
  `TRITON_ATTN`, `ROCM_FLASH`, `FLASHMLA`, etc.
- `VLLM_USE_FLASHINFER_SAMPLER=1` — use FlashInfer sampling kernels.
- `VLLM_FLASH_ATTN_VERSION=2|3` — pin FlashAttention major version.
- `VLLM_WORKER_MULTIPROC_METHOD=fork|spawn` — worker start method.
- `VLLM_ENABLE_V1_MULTIPROCESSING=1` — run EngineCore in its own process.
- `VLLM_LOGGING_LEVEL=DEBUG|INFO|WARNING` — log verbosity.
- `VLLM_CACHE_ROOT` (default `~/.cache/vllm`) — compiled artifacts / torch.compile
  cache. `VLLM_CONFIG_ROOT` — config dir.
- `VLLM_USE_MODELSCOPE=1` — pull models from ModelScope instead of HF.
- `VLLM_HOST_IP` / `VLLM_PORT` — bind/advertise address for distributed setup.
- `VLLM_RPC_TIMEOUT`, `VLLM_ENGINE_ITERATION_TIMEOUT_S` — timeouts (raise for very
  large models / slow startup).
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` — allow `max_model_len` beyond the model's
  trained context (accuracy risk).
- `VLLM_DISABLED_KERNELS` — comma list of custom kernels to disable (debugging).
- `VLLM_TORCH_PROFILER_DIR` — enable the torch profiler; dumps traces here.
- `VLLM_TRACE_FUNCTION=1` — function-level tracing for deep debugging (slow).
- `VLLM_DO_NOT_TRACK=1` — opt out of usage stats.

### MoE / FP8 / Kernels

- `VLLM_USE_DEEP_GEMM=1` — DeepGEMM FP8 GEMMs (Hopper, MoE/FP8 throughput).
- `VLLM_FUSED_MOE_CHUNK_SIZE` — chunk size for fused MoE kernels.
- `VLLM_FLASHINFER_FORCE_TENSOR_CORES=1` — force tensor-core path in FlashInfer.
- `VLLM_TEST_FORCE_FP8_MARLIN=1` — force Marlin FP8 path (testing).

### CPU Backend

- `VLLM_CPU_KVCACHE_SPACE` (GiB) — KV cache size for the CPU backend.
- `VLLM_CPU_OMP_THREADS_BIND` — pin OpenMP threads to cores.

### Multimodal

- `VLLM_MM_INPUT_CACHE_GIB` — multimodal input cache size.
- `VLLM_IMAGE_FETCH_TIMEOUT` — remote image fetch timeout.

### ROCm-Specific (`VLLM_ROCM_*`, see rocm note)

- `VLLM_USE_TRITON_FLASH_ATTN=1` — Triton FlashAttention on ROCm (default on).
- `VLLM_ROCM_USE_AITER=1` — master switch for AITER kernels (MI300-class).
- `VLLM_ROCM_USE_AITER_LINEAR|MOE|PAGED_ATTN|RMSNORM=1` — per-op AITER toggles.
- `VLLM_ROCM_FP8_PADDING=1` — pad FP8 GEMMs for performance.
- `VLLM_ROCM_CUSTOM_PAGED_ATTN=1` — ROCm custom paged-attention kernel.

### External Runtime (not vLLM, but matter)

- `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` — GPU
  selection (CUDA / ROCm).
- `NCCL_*` (CUDA) and `NCCL_*`/RCCL (ROCm), e.g. `NCCL_DEBUG=INFO`,
  `NCCL_P2P_DISABLE`, `NCCL_IB_DISABLE` — collective comms tuning/debug.
- `VLLM_NCCL_SO_PATH` — point vLLM at a specific NCCL/RCCL `.so`.
- `PYTORCH_TUNABLEOP_ENABLED=1` (ROCm) — autotune GEMMs via TunableOp.
- `HSA_OVERRIDE_GFX_VERSION` (ROCm) — spoof gfx arch on unsupported cards.
- `OMP_NUM_THREADS`, `HF_TOKEN`, `HF_HOME`, `TRANSFORMERS_OFFLINE`.

## Rules

- Set env vars **before** `vllm serve`; many are read once at import/startup.
- Record every non-default env var alongside any benchmark or bug report — they
  silently change kernels and numerics.
- Prefer an engine arg when one exists; reach for env vars for backend/kernel
  selection and debugging.

## Avoid

- Leaving debug vars (`VLLM_TRACE_FUNCTION`, `NCCL_DEBUG=INFO`) on in production.
- Forcing a backend/kernel env var that the GPU does not support, then blaming the
  model for the resulting error.

## Related

- `knowledge/vllm/serving/engine-args-reference.md`
- `knowledge/vllm/architecture/attention-backends.md`
- `knowledge/vllm/rocm/rocm-setup-and-tuning.md`
- `knowledge/vllm/cuda/cuda-setup-and-tuning.md`

## Source

- `vllm/envs.py` (authoritative list + defaults)
- https://docs.vllm.ai/en/latest/serving/env_vars.html
