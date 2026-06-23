# Architecture Knowledge

How vLLM is structured and how a request flows through it. Read these to reason
about where a feature lives, why a config affects memory or throughput, and what
to edit.

## Notes

- `vllm-architecture-overview.md`: the V1 engine, component map, request
  lifecycle, process/executor model.
- `paged-attention-kv-cache.md`: blocks, the KV cache manager, prefix caching,
  memory accounting.
- `scheduler-batching.md`: continuous batching, chunked prefill, preemption,
  priority.
- `scheduler-model-size-impact.md`: how scheduler behavior interacts with model
  size (small vs medium vs large) and whether scheduling changes output.
- `attention-backends.md`: FlashAttention/FlashInfer/Triton/ROCm backends and how
  selection works.

## Source Of Truth

- Code: `vllm/v1/` (engine core, scheduler, KV cache, worker, attention).
- Upstream design docs: https://docs.vllm.ai/en/latest/design/
