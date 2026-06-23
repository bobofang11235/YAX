# Performance Tuning

## Use When

Use when throughput is too low, latency (TTFT/ITL) too high, or you hit OOM /
preemption and need a principled order of knobs instead of guessing.

## Lesson

Tune in a fixed order, measuring with `vllm bench serve` between changes. Define
the target first: offline throughput, interactive latency, or memory fit. They
pull in different directions.

### Measure First

- Use `vllm bench serve` (or `benchmarks/benchmark_serving.py`) with a realistic
  input/output length distribution and request rate.
- Watch: output tokens/s (throughput), TTFT (time to first token), ITL/TPOT
  (inter-token latency), and preemption logs.

### Throughput-First (batch/offline)

1. Maximize concurrency: raise `--max-num-batched-tokens`, then `--max-num-seqs`
   until GPU memory or latency caps out.
2. Grow the KV pool: raise `--gpu-memory-utilization` (e.g. 0.90→0.95) leaving
   headroom; consider `--kv-cache-dtype fp8` to fit more sequences.
3. Keep CUDA graphs/compile on (no `--enforce-eager`); try a higher `-O` level.
4. Add GPUs with TP; use PP/EP for very large or MoE models.
5. Quantize weights (AWQ/GPTQ/FP8) to free memory for batch size.

### Latency-First (interactive)

1. Keep `--max-num-batched-tokens` moderate so prefill chunks don't delay decode.
2. Rely on chunked prefill + prefix caching to cut TTFT for shared prompts.
3. Consider speculative decoding (`--speculative-config`) to cut ITL for a single
   stream.
4. Pick the fastest decode backend for your GPU (e.g. FlashInfer on CUDA).
5. Use a smaller/quantized model or TP to drop per-token compute.

### Memory Fit (OOM / "can't fit max_model_len")

1. Lower `--gpu-memory-utilization` if OOM at load; raise it if KV pool too small.
2. Lower `--max-model-len` to what you actually need.
3. `--kv-cache-dtype fp8` to halve KV memory.
4. Quantize weights; or `--cpu-offload-gb` as a last resort (slow).
5. Add GPUs (TP/PP).

### Platform Boosters

- CUDA: FlashInfer backend, `VLLM_USE_DEEP_GEMM=1` for FP8/MoE on Hopper, CUDA
  graphs.
- ROCm: `VLLM_ROCM_USE_AITER=1` on MI300, `PYTORCH_TUNABLEOP_ENABLED=1` for GEMM
  autotuning, Triton FlashAttention.

## Rules

- Change one knob per run and re-benchmark; keep a small results table.
- Fix correctness before performance; never tune around a numerical bug.
- Record GPU arch, vLLM version, backend, and every non-default flag/env var with
  each result.

## Avoid

- `--gpu-memory-utilization` near 1.0 (mid-run OOM).
- Chasing throughput knobs while decode P99 latency silently blows past target.
- Comparing numbers across different backends/versions/GPUs.

## Related

- `knowledge/architecture/scheduler-batching.md`
- `knowledge/architecture/paged-attention-kv-cache.md`
- `knowledge/serving/engine-args-reference.md`
- `tools/serving/benchmark-serving.md`

## Source

- `benchmarks/` in vllm; `vllm bench serve`
- https://docs.vllm.ai/en/latest/performance/optimization.html
