# Scheduler And Continuous Batching

## Use When

Use when tuning throughput vs latency, reasoning about `--max-num-seqs`,
`--max-num-batched-tokens`, chunked prefill, preemption, or request priority.

## Lesson

vLLM does **continuous (iteration-level) batching**: every decode step the
scheduler re-decides which requests to run, instead of waiting for a whole batch
to finish. This keeps the GPU full as requests of different lengths start and
finish. The scheduler lives in `vllm/v1/core/sched/`.

### Per-Step Budget

Each step the scheduler admits work under two caps:

- `--max-num-seqs`: max concurrent sequences in a batch (concurrency ceiling).
- `--max-num-batched-tokens`: max tokens processed in one step (prefill tokens +
  decode tokens). This is the main throughput/latency knob.

Prefill (processing the prompt) is compute-bound and token-heavy; decode is one
token per sequence and memory-bandwidth-bound. Mixing them well is the core job.

### Chunked Prefill

- `--enable-chunked-prefill` (default on in V1) splits a long prompt's prefill
  into chunks that are interleaved with ongoing decodes, so a big prompt does not
  stall every other request's decode (avoids latency spikes / head-of-line
  blocking).
- Tune `--max-num-batched-tokens`: larger = more prefill throughput but larger
  decode latency jitter; smaller = smoother decode latency.

### Preemption

- When the KV pool is exhausted, the scheduler **preempts** lower-priority or
  newer sequences: it either **recomputes** them later (default) or **swaps** KV
  to CPU (`--swap-space`, V0-style). Frequent preemption shows as throughput
  collapse and "Sequence preempted" logs — reduce concurrency or add memory.

### Priority And Policy

- Default scheduling policy is FCFS. `--scheduling-policy priority` plus a
  per-request `priority` enables priority scheduling.

## Rules

- Throughput-first (batch/offline): raise `--max-num-batched-tokens` and
  `--max-num-seqs` until GPU memory or latency limits hit.
- Latency-first (interactive): keep `--max-num-batched-tokens` moderate so prefill
  chunks do not delay decodes; rely on chunked prefill.
- Seeing preemption logs: lower `--max-num-seqs`, lower `--max-model-len`, or add
  GPU memory; do not just raise utilization to 1.0.
- Measure with `vllm bench serve` while sweeping these knobs — do not guess.

## Avoid

- Treating `--max-num-seqs` as the throughput knob; batched-tokens usually
  dominates GPU-bound throughput.
- Setting batched-tokens so high that decode P99 latency becomes unacceptable.

## Related

- `knowledge/vllm/architecture/paged-attention-kv-cache.md`
- `knowledge/vllm/serving/performance-tuning.md`
- `knowledge/vllm/serving/engine-args-reference.md`

## Source

- `vllm/v1/core/sched/scheduler.py`
- https://docs.vllm.ai/en/latest/design/
