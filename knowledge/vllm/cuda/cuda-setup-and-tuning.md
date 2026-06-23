# CUDA Setup And Tuning

## Use When

Use when installing, running, or tuning vLLM on NVIDIA GPUs, choosing kernels for
a compute capability, or debugging CUDA-specific backend/perf issues.

## Lesson

On NVIDIA, vLLM ships prebuilt CUDA wheels; `pip install vllm` (or `uv pip install
vllm`) is the normal path. The platform layer `vllm/platforms/cuda.py` gates
features on **compute capability** (SM): e.g. FP8 tensor cores need Hopper
(SM90)/Ada (SM89); Marlin needs Ampere (SM80)+. Match kernels to the GPU.

### Install

- `pip install vllm` pulls a CUDA build matched to a PyTorch/CUDA combo. To use a
  different CUDA, install the matching PyTorch first, then build vLLM from source.
- Select GPUs with `CUDA_VISIBLE_DEVICES`.
- Optional acceleration: `pip install flashinfer-python` for the FlashInfer
  backend.

### Attention Backend

- Default: `FLASH_ATTN` (FA2/FA3) on supported GPUs.
- `VLLM_ATTENTION_BACKEND=FLASHINFER` for strong decode and FP8-KV paths;
  `XFORMERS` as a fallback for unusual head sizes.
- `VLLM_FLASH_ATTN_VERSION=3` to pin FA3 on Hopper.

### FP8 / MoE Acceleration

- `VLLM_USE_DEEP_GEMM=1` enables DeepGEMM FP8 GEMMs on Hopper — large gains for
  FP8 dense and MoE.
- FP8 (OCP e4m3) is near-lossless on Hopper/Ada; pair with FlashInfer for KV FP8.
- Marlin/Machete kernels accelerate AWQ/GPTQ/FP8 mixed-precision GEMMs on Ampere+.

### CUDA Graphs / Compile

- On by default (omit `--enforce-eager`). Captures decode graphs to cut launch
  overhead; biggest win at small/medium batch.
- `-O3` / `--compilation-config` raises the torch.compile + piecewise CUDA-graph
  level (more fusion, longer warmup). Cache lives under `VLLM_CACHE_ROOT`.

### Collectives

- NCCL for multi-GPU; tune/debug with `NCCL_DEBUG=INFO`, `NCCL_P2P_DISABLE`,
  `NCCL_IB_DISABLE` (InfiniBand), and pin a library with `VLLM_NCCL_SO_PATH`.
- `VLLM_SKIP_P2P_CHECK=1` bypasses the P2P access check (diagnose first).

## Rules

- Verify the GPU's compute capability supports the kernel/dtype before forcing it
  (FP8 → Hopper/Ada; Marlin → Ampere+).
- Keep CUDA graphs on in production; use `--enforce-eager` only to debug.
- Record SM arch, CUDA/PyTorch/vLLM versions, backend, and env vars with results.

## Avoid

- Forcing FP8 on pre-Hopper hardware and expecting tensor-core speed.
- Mixing a manually installed PyTorch/CUDA with a prebuilt vLLM wheel built for a
  different CUDA.

## Related

- `knowledge/vllm/architecture/attention-backends.md`
- `knowledge/vllm/serving/quantization.md`
- `knowledge/vllm/serving/environment-variables.md`

## Source

- `vllm/platforms/cuda.py`, `Dockerfile`
- https://docs.vllm.ai/en/latest/getting_started/installation/gpu/cuda.html
