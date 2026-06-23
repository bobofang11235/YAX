# Performance Estimation From Model Config

## Use When

Use to predict, before running anything, the order of magnitude of memory,
concurrency, decode throughput, and TTFT from a model's config plus a few
hardware numbers — and to know which regime (memory- vs compute-bound) you are
in. Pair with the calculator `scripts/perfcalc.py`.

## Lesson

Most first-order performance numbers fall out of the config. These are
back-of-the-envelope: right order of magnitude and regime, not exact. Real
numbers are lower (comms, kernel overhead, fragmentation, imperfect overlap);
always confirm with `vllm bench serve`.

Config fields used: `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
`num_key_value_heads` (GQA), `head_dim` (or `hidden/heads`), `intermediate_size`,
`vocab_size`. Hardware: VRAM, HBM **bandwidth** (BW), dense **TFLOPS**.

### 1. Weight Memory

```
weight_bytes = params * bytes_per_param      # bf16=2, fp8=1, int4=0.5
```

Per GPU under tensor parallel: `weight_bytes / TP`.

### 2. KV Cache Per Token (the concurrency driver)

```
kv_per_token = 2 (K,V) * num_layers * num_kv_heads * head_dim * kv_bytes
```

GQA shrinks this via `num_kv_heads`. FP8 KV halves `kv_bytes`.

### 3. KV Pool And Max Concurrency

```
kv_pool   = VRAM * util - weights - activation_headroom
max_tokens = kv_pool / kv_per_token
max_seqs  ~= max_tokens / avg_context_len
```

This is the **concurrency ceiling** the scheduler can fill (links directly to
`scheduler-model-size-impact.md`).

### 4. Decode Throughput (memory-bandwidth bound)

Each decode step reads the weights once plus the KV of every active token:

```
bytes_per_step = weight_bytes/TP + batch * (kv_per_token/TP) * context_len
step_time      = bytes_per_step / BW
tokens_per_s   = batch / step_time
```

At small batch / short context → **weight-bound** (batching is almost free).
At large batch / long context → **KV-bound**.

### 5. Roofline: Critical Batch Size

Decode flips from memory-bound to compute-bound around:

```
critical_batch ~= bytes_per_param * TFLOPS / (2 * BW)
```

e.g. H100 bf16: `2 * 989e12 / (2 * 3.35e12) ~= 295`. **Below ~295 concurrent
decodes, batching keeps improving throughput** — this is *why* continuous
batching matters and why small models (which seat huge batches) gain the most.

### 6. Prefill / TTFT (compute bound)

```
forward_flops = 2 * params * prompt_len      # 2 FLOPs/param/token
TTFT          = forward_flops / (TFLOPS * TP * MFU)   # MFU ~0.3-0.5
prefill_tps   = prompt_len / TTFT
```

## Example (worked by `scripts/perfcalc.py`)

Llama-3-8B (hidden 4096, 32 layers, 32 heads, 8 KV heads, inter 14336) on 1×H100
(80 GB, 3.35 TB/s, 989 TFLOPS), bf16, ctx 2048:

- Weights ~14 GB; KV **128 KB/token**; KV pool ~56 GB → **~459k tokens ~= 224 seqs**.
- Decode: batch 1 ~219 tok/s (weight-bound) → batch 128 ~8.7k tok/s (turning
  KV-bound). Critical batch ~295.
- Prefill 2048: ~31 TFLOP → **TTFT ~78 ms**, ~26k tok/s.

70B on 4×H100 (TP=4), ctx 4096: weights ~130 GB (32 GB/GPU); KV 320 KB/token;
~120 seqs; decode batch 1 ~95 tok/s (ideal upper bound; real lower due to TP
comms); TTFT(4096) ~360 ms.

### Run It

```bash
python3 scripts/perfcalc.py --config /path/to/config.json --vram-gb 80 --bw-tbs 3.35 --flops-tflops 989
python3 scripts/perfcalc.py --hidden 4096 --layers 32 --heads 32 --kv-heads 8 --inter 14336 --vocab 128256
python3 scripts/perfcalc.py --config cfg.json --gpus 4 --weight-bytes 1 --kv-bytes 1   # TP=4, fp8 weights+KV
```

## Rules

- Use these to size GPUs, pick TP, set `--max-model-len`, and sanity-check a
  benchmark (if measured throughput is far below estimate, look for a
  misconfiguration — eager mode, wrong backend, comms bound).
- Decode is memory-bound: estimate from **bandwidth**, not FLOPS. Prefill is
  compute-bound: estimate from **FLOPS**.
- Treat outputs as upper bounds; real systems lose 20-60% to overhead, more with
  TP/PP comms and low MFU.

## Avoid

- Using the FLOPS number to predict decode speed (it is bandwidth-bound).
- Trusting the estimate for MoE without adjusting params (use active+stored
  separately) or for models with unusual attention (MLA, sliding window).
- Forgetting activation/fragmentation headroom — leave a margin.

## Related

- `knowledge/serving/performance-factors.md`
- `knowledge/serving/performance-tuning.md`
- `knowledge/architecture/scheduler-model-size-impact.md`
- `knowledge/architecture/paged-attention-kv-cache.md`
- `scripts/perfcalc.py`

## Source

- `scripts/perfcalc.py`
- PagedAttention paper; vLLM `benchmarks/`
- Transformer FLOPs/memory roofline analysis (standard back-of-envelope)
