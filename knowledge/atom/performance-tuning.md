# ATOM Performance Tuning

## Use When

Use when ATOM throughput is low or latency high, on AMD GPUs. Measure with
`atom.benchmarks.benchmark_serving` and the trace tools.

## Lesson

Same physics as any engine (decode is bandwidth-bound, prefill compute-bound; see
`knowledge/shared/performance-estimation.md`), with ATOM/AMD specifics. Tune one
thing at a time and re-benchmark.

### Measure First

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=deepseek-ai/DeepSeek-R1 --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random --random-input-len=1024 --random-output-len=1024 \
  --num-prompts=1280 --max-concurrency=128 --request-rate=inf --ignore-eos \
  --percentile-metrics="ttft,tpot,itl,e2el"
```

For regressions, collect a trace (`--torch-profiler-dir --mark-trace` or the
benchmark `--profile` flag) and analyze with `tools/parse_trace.py` /
`tools/analyze_trace_summary.py` (per-kernel, per-layer breakdown).

### Throughput / Latency Order

1. Keep **piecewise torch.compile + CUDA graphs** on (default opt level 3); only
   lower for debugging.
2. Quantize: FP8 / MXFP4 (weights) + `--kv_cache_dtype fp8` to grow KV / cut
   bandwidth; try `--online_quant_config` to compare precisions fast.
3. MoE at scale: EP + **TBO** to hide all-to-all latency; DP attention.
4. Latency: **MTP speculative decoding** (`--method mtp`).
5. AMD GEMM tuning: `PYTORCH_TUNABLEOP_ENABLED=1` (hipBLASLt), ensure the right
   **AITER backend** (ASM/CK/Triton) is selected for your shapes.
6. Consider **P/D disaggregation** when prefill and decode contend.

### AITER Is The Lever

ATOM's speed is mostly AITER kernel selection. A slow op usually means a
suboptimal AITER backend/fallback — check the trace breakdown.

## Rules

- Change one knob per run; record GPU arch (gfx), ROCm + AITER + ATOM versions,
  quant format, TBO/EP layout, and opt level.
- Fix correctness (lm-eval) before tuning.
- Use the trace tools — ATOM ships per-kernel/per-layer analysis; use it.

## Avoid

- Lowering the compile/opt level in production (loses CUDA-graph decode speedup).
- Benchmarking the first run (model compilation can take ~10 min — warm up first).

## Related

- `knowledge/atom/aiter-model-ops.md`
- `knowledge/atom/distributed-tbo.md`
- `knowledge/shared/performance-estimation.md`
- `knowledge/shared/performance-factors.md`

## Source

- ATOM README (benchmarking/profiling), `docs/serving_benchmarking_guide.md`
- https://rocm.github.io/ATOM/benchmark-dashboard
