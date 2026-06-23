# SGLang Performance Tuning

## Use When

Use when SGLang throughput is low, latency high, or you hit OOM, and you want a
principled order of knobs. Measure with `python -m sglang.bench_serving`.

## Lesson

Same physics as any engine (decode is bandwidth-bound, prefill compute-bound; see
the vLLM `performance-estimation.md`), but the knob names and a few defaults
differ. Tune one thing at a time and re-benchmark.

### Measure First

- `python -m sglang.bench_serving --backend sglang --host ... --port ...` with a
  realistic input/output length distribution and request rate; or
  `python -m sglang.bench_one_batch` for a single batch.
- Watch output tokens/s, TTFT, ITL, and the **RadixAttention cache hit rate** for
  prefix-heavy traffic.

### Throughput-First

1. Grow the KV pool: raise `--mem-fraction-static` (leave headroom).
2. Raise concurrency: `--max-running-requests`, and `--max-total-tokens`.
3. Keep the **overlap scheduler** and **CUDA graphs** on (do not disable).
4. Tune `--chunked-prefill-size` up for prefill-heavy throughput.
5. Keep **RadixAttention** on; use `--schedule-policy lpm` for prefix reuse.
6. Quantize (FP8/AWQ/GPTQ); enable DeepGEMM FP8 (`SGL_ENABLE_JIT_DEEPGEMM`) on
   Hopper/Blackwell; scale out with TP/EP.

### Latency-First

1. Keep `--chunked-prefill-size` moderate so prefill chunks don't delay decode.
2. Use speculative decoding (`--speculative-algorithm EAGLE3`, tune
   `--speculative-num-steps` / `--speculative-eagle-topk`).
3. Pick the fastest attention backend for the GPU (flashinfer / fa3).
4. Smaller/quantized model or more TP to cut per-token compute.

### Memory Fit (OOM)

1. Lower `--mem-fraction-static` (OOM at load) or raise it (KV pool too small).
2. Lower `--context-length`; use `--kv-cache-dtype fp8`.
3. Quantize weights; scale out (TP).

### Platform

- CUDA: flashinfer/fa3 backend, DeepGEMM FP8, CUDA graphs (BCG).
- ROCm: `SGLANG_USE_AITER=1`, `PYTORCH_TUNABLEOP_ENABLED=1`, hipBLASLt.

## Rules

- Change one knob per run; record GPU arch, SGLang version, backend, flags, and
  cache hit rate.
- Fix correctness before performance.
- For prefix-heavy workloads, RadixAttention + `lpm` is usually the biggest win.

## Avoid

- Disabling overlap scheduler / CUDA graph / radix cache to "simplify"; each is a
  default performance win.
- `--mem-fraction-static` so high it OOMs mid-run.

## Related

- `knowledge/sglang/radixattention.md`
- `knowledge/sglang/server-args.md`
- `knowledge/serving/performance-estimation.md` (engine-agnostic roofline)
- `knowledge/serving/performance-factors.md`

## Source

- `python/sglang/bench_serving.py`
- https://docs.sglang.ai/
