# PagedAttention And KV Cache

## Use When

Use when reasoning about GPU memory, `--gpu-memory-utilization`,
`--max-model-len`, `--block-size`, prefix caching, or out-of-memory (OOM) and
"not enough KV cache" errors.

## Lesson

The KV cache stores per-token key/value tensors so past tokens are not
recomputed each step. vLLM stores it in **fixed-size blocks** (default 16 tokens
per block; some backends use other sizes) instead of one contiguous buffer per
sequence. This is **PagedAttention**: like OS paging, blocks are allocated on
demand and can be shared, which kills fragmentation and enables prefix sharing.

### Memory Accounting

- At startup vLLM runs a profiling forward pass, measures peak activation memory,
  then gives the rest (up to `--gpu-memory-utilization`, default ~0.9) to the KV
  cache as a fixed pool of blocks.
- Total KV bytes per token ≈ `2 (K and V) * num_layers * num_kv_heads *
  head_dim * dtype_bytes`. GQA/MQA shrink this via fewer KV heads. FP8 KV cache
  (`--kv-cache-dtype fp8`) roughly halves it vs fp16.
- KV cache capacity in tokens ≈ `pool_bytes / per_token_bytes`. If that is below
  `max_model_len`, vLLM warns or errors; lower `max_model_len`, raise
  `gpu_memory_utilization`, shrink the model/dtype, or add GPUs.

### Prefix Caching (Automatic)

- `--enable-prefix-caching` (default on in V1) hashes block contents so requests
  sharing a prefix (system prompt, few-shot examples, common documents) reuse
  already-computed KV blocks instead of recomputing the prefill.
- Biggest wins: many requests with a long shared system prompt, agent loops, and
  multi-turn chat. Negligible/overhead when prefixes never repeat.
- Blocks are reference-counted and evicted LRU when the pool is full.

### Block Sharing And Copy-on-Write

- Parallel sampling and beam search share prompt blocks; divergence triggers
  copy-on-write of just the diverging block.

## Rules

- OOM at load: lower `--gpu-memory-utilization`, lower `--max-model-len`, reduce
  `--max-num-seqs` / `--max-num-batched-tokens`, or use more GPUs / quantization.
- "Engine cannot fit max_model_len": KV pool is too small for the context; same
  levers as above, or `--kv-cache-dtype fp8` to halve KV memory.
- Long shared prompts: keep prefix caching on; it is the cheapest latency win.
- `--block-size` trades allocation granularity vs overhead; defaults are usually
  best. Some attention backends constrain valid block sizes.

## Avoid

- Setting `--gpu-memory-utilization` near 1.0: leaves no headroom for activation
  spikes / fragmentation and causes mid-run OOM.
- Assuming FP8 KV cache is free: it can cost a little accuracy; validate.

## Related

- `knowledge/architecture/scheduler-batching.md`
- `knowledge/serving/performance-tuning.md`
- `knowledge/serving/engine-args-reference.md`

## Source

- `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/block_pool.py`
- PagedAttention paper: https://arxiv.org/abs/2309.06180
- https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html
