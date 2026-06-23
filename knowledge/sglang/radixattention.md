# RadixAttention

## Use When

Use when reasoning about KV reuse, prefix caching, multi-turn/agent/fork
workloads, or the `--disable-radix-cache` / memory knobs in SGLang.

## Lesson

**RadixAttention** is SGLang's signature feature: automatic KV-cache **reuse**
across requests via a **radix tree** of token sequences. Instead of hashing fixed
blocks (vLLM's automatic prefix caching), SGLang keeps a tree whose paths are
token prefixes; any new request reuses the longest matching path's KV and only
computes the novel suffix.

Why a tree helps beyond simple prefix caching:

- **Tree-structured sharing**: many requests branching off a common prefix (system
  prompt, few-shot block, a document) share one copy of that KV.
- **`fork` in the frontend** creates branches that share the parent's KV cheaply —
  ideal for parallel sampling, tree-of-thought, multi-turn agents.
- **Cache-aware scheduling**: the scheduler can order requests to maximize cache
  hits (LPM — longest-prefix-match policy).

### Memory Model

- KV lives in pools: `ReqToTokenPool` (request → token slots) and `TokenToKVPool`
  (token → KV blocks), in `srt/mem_cache/memory_pool.py`.
- The `RadixCache` (`srt/mem_cache/radix_cache.py`) indexes cached prefixes;
  unreferenced nodes are evicted LRU when the pool is full.
- `--mem-fraction-static` controls how much VRAM is reserved for the KV pool
  (analogous to vLLM's `--gpu-memory-utilization`). Too high → OOM; too low →
  small pool, fewer concurrent/long requests.

### Controls

- On by default. `--disable-radix-cache` turns it off (useful to A-B its effect or
  debug a cache correctness issue).
- `--schedule-policy lpm` (default-ish) maximizes prefix reuse; `fcfs` is simplest.
- HiCache / hierarchical caching (newer) extends reuse to CPU/host tiers; UnifiedTree
  is on by default for hybrid SWA/Mamba models in recent releases.

## Rules

- Workloads with shared prefixes (system prompts, RAG with common context, agents,
  parallel sampling) benefit most — keep RadixAttention on.
- If memory is tight, lower `--mem-fraction-static` or context length before
  disabling the cache.
- When A-B-ing throughput, note whether radix cache was on; it can dominate
  results for prefix-heavy workloads.
- Suspected wrong output after heavy reuse → A-B with `--disable-radix-cache` to
  isolate a cache bug from a model bug.

## Avoid

- Treating RadixAttention as identical to vLLM APC; the tree enables sharing that
  block-hash caching does not (fork, branching).
- Benchmarking prefix-heavy traffic without reporting the cache hit rate.

## Related

- `knowledge/sglang/architecture.md`
- `knowledge/sglang/performance-tuning.md`
- `knowledge/sglang/vllm-vs-sglang.md`
- `knowledge/architecture/paged-attention-kv-cache.md` (vLLM comparison)

## Source

- `python/sglang/srt/mem_cache/radix_cache.py`, `memory_pool.py`
- RadixAttention paper: https://arxiv.org/abs/2312.07104
- https://docs.sglang.ai/
