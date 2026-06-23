# ROCm FP8 Formats (fnuz vs OCP)

## Use When

Use when FP8 accuracy or throughput differs between AMD and NVIDIA, when porting
an FP8 checkpoint, or when FP8 results look wrong only on ROCm.

## Lesson

FP8 e4m3 has two incompatible encodings in practice:

- **OCP e4m3** (`e4m3` / "OCP FN") — used on NVIDIA Hopper/Ada. Has inf/NaN
  representations in the standard OCP layout.
- **fnuz e4m3** (`e4m3fnuz`, "finite, no -0, unsigned-zero") — used by AMD CDNA
  (MI300). Different bias and no inf, reclaiming one exponent value for range.

The same raw bytes mean different numbers in the two formats. vLLM/PyTorch handle
the mapping internally (`torch.float8_e4m3fn` vs `torch.float8_e4m3fnuz`), but an
FP8 checkpoint or KV cache produced for one platform must be interpreted with the
right format on the other. Scaling factors must match the chosen format.

## Rules

- When an FP8 model is correct on CUDA but wrong on ROCm (or vice versa), suspect
  the format (fnuz vs OCP) and the activation/KV scales before the model code.
- Quantize/serve FP8 with the format native to the target GPU; rely on vLLM's
  conversion rather than reinterpreting bytes by hand.
- KV-cache FP8 (`--kv-cache-dtype fp8`) inherits the same platform format split;
  validate accuracy per platform.
- Record the GPU arch and FP8 format with any FP8 benchmark or accuracy result.

## Avoid

- Assuming an FP8 checkpoint is bit-portable across AMD and NVIDIA.
- Comparing FP8 numerics across platforms without noting the format difference.

## Related

- `knowledge/vllm/serving/quantization.md`
- `knowledge/vllm/rocm/rocm-setup-and-tuning.md`

## Source

- `vllm/model_executor/layers/quantization/` (fp8 utils)
- OCP FP8 spec; AMD CDNA FP8 (fnuz) docs
