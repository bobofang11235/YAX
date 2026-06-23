# Quantization

## Use When

Use when a model does not fit, you want more throughput, or you must serve a
pre-quantized checkpoint. Covers weight, activation, and KV-cache quantization.

## Lesson

Quantization trades a little accuracy for less memory and (often) more speed.
vLLM detects most quantization from the checkpoint's config; `--quantization`/`-q`
forces or selects a method/kernel. Three independent axes:

1. **Weights** (storage + bandwidth): AWQ, GPTQ, FP8, INT8, GGUF, bitsandbytes.
2. **Activations** (compute): FP8 (W8A8-FP8), INT8 (W8A8-INT8).
3. **KV cache** (context memory): `--kv-cache-dtype fp8`.

### Methods

- **FP8 (e4m3)** — `-q fp8` or a pre-quantized FP8 checkpoint. Best on Hopper
  (H100), MI300, and Ada. Near-lossless for most models; can run W8A8.
  On ROCm the on-GPU format is **fp8 fnuz** (e4m3fnuz), distinct from CUDA OCP
  e4m3 — see the ROCm note.
- **AWQ** — `-q awq` / `awq_marlin`. 4-bit weights, activation-aware; great
  quality/size; use the Marlin kernel on supported NVIDIA GPUs for speed.
- **GPTQ** — `-q gptq` / `gptq_marlin`. 4-bit weights; very common checkpoints.
- **compressed-tensors** — unified format (llm-compressor) covering FP8/INT8/W4A16;
  auto-detected.
- **bitsandbytes** — `-q bitsandbytes` with `--load-format bitsandbytes`; on-the-fly
  4/8-bit, convenient but slower than Marlin paths.
- **GGUF** — `--load-format gguf`; load llama.cpp-style quant checkpoints (limited
  model coverage).
- **INT8 (W8A8)** — SmoothQuant-style; good on GPUs without strong FP8.
- **Marlin / Machete** — fast mixed-precision GEMM kernels used under AWQ/GPTQ/FP8,
  not standalone formats.

### Choosing

- Have a pre-quantized checkpoint → just serve it; vLLM auto-detects. Pass `-q`
  only to pick a faster kernel variant (e.g. `gptq_marlin`).
- Need to fit a big model on limited VRAM → AWQ/GPTQ 4-bit weights.
- Hopper/MI300 and want speed with minimal quality loss → FP8 (W8A8).
- Context (KV) is the bottleneck, not weights → `--kv-cache-dtype fp8`.

## Rules

- Always validate accuracy after quantizing (a small eval / A-B vs fp16). Treat KV
  FP8 and activation FP8 as separate risks.
- Match the kernel to the GPU: Marlin needs Ampere+; FP8 wants Hopper/Ada/MI300.
- Record method + kernel + GPU arch in any benchmark; they change both speed and
  numerics.

## Avoid

- Stacking aggressive weight quant + FP8 KV without measuring quality.
- Assuming `-q fp8` behaves identically on CUDA and ROCm (format differs).

## Related

- `knowledge/vllm/serving/performance-tuning.md`
- `knowledge/vllm/rocm/rocm-fp8-formats.md`
- `knowledge/vllm/serving/engine-args-reference.md`

## Source

- `vllm/model_executor/layers/quantization/`
- https://docs.vllm.ai/en/latest/features/quantization/
