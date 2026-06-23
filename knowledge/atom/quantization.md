# ATOM Quantization

## Use When

Use when fitting a model, raising throughput, or serving a quantized checkpoint on
ATOM. ATOM supports more aggressive AMD-oriented formats than the typical FP8/INT
set, plus on-the-fly re-quantization.

## Lesson

ATOM auto-detects quantization from the HuggingFace config and dispatches to AITER
quantized GEMMs. Formats:

- **FP8** — incl. FP8-block source checkpoints; **PTPC-FP8** (per-token per-channel)
  as an online target.
- **MXFP4** — microscaling FP4; strong size/throughput for MoE on MI300.
- **INT8 / INT4** — weight quantization.
- **KV cache FP8** — `--kv_cache_dtype fp8`.

### Online Quantization (distinctive)

- `--online_quant_config <...>` re-quantizes an **unquantized or FP8-block** source
  checkpoint to **PTPC-FP8 / MXFP4 mixed precision at load time** — no offline
  re-packing. Convenient for trying precisions quickly.

### Mixed Precision

- Different layers/experts can use different precisions (e.g. PTPC-FP8 + MXFP4),
  tuned for accuracy vs throughput.

## Rules

- Prefer a pre-quantized checkpoint when available; ATOM auto-detects it.
- Use `--online_quant_config` to experiment without re-packing, then validate
  accuracy (lm-eval gsm8k or similar) per precision.
- MXFP4 is attractive for large MoE on MI300 — measure accuracy, it is aggressive.
- Record format + AITER/ROCm version + GPU arch with any result; FP8 on ROCm uses
  the **fnuz** family (see the shared ROCm FP8 note).

## Avoid

- Stacking aggressive weight quant (MXFP4/INT4) + FP8 KV without an accuracy check.
- Assuming ATOM FP8 numerics match CUDA FP8 (format differs on ROCm).

## Related

- `knowledge/atom/aiter-model-ops.md`
- `knowledge/atom/configuration.md`
- `knowledge/vllm/rocm/rocm-fp8-formats.md` (fnuz vs OCP, shared)

## Source

- ATOM README (online quantization), `docs/configuration_guide.md`
- https://github.com/ROCm/ATOM
