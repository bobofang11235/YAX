#!/usr/bin/env python3
"""Rough vLLM performance/memory estimator from a model config.

Back-of-the-envelope only: it predicts the right order of magnitude and which
regime (memory- vs compute-bound) a model is in, not exact benchmark numbers.
Always confirm with `vllm bench serve`.

Inputs: either a HuggingFace config.json (--config) or explicit model dims, plus
hardware specs. Outputs: weight memory, KV cache per token, KV pool / max
concurrency, decode step time + tokens/s vs batch, prefill TTFT, and the
critical batch size where decode flips from memory- to compute-bound.
"""

import argparse
import json
from pathlib import Path

GB = 1024 ** 3


def load_config_dims(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    # Some configs nest under text_config (multimodal) or similar.
    base = cfg.get("text_config", cfg)
    hidden = base.get("hidden_size")
    layers = base.get("num_hidden_layers")
    heads = base.get("num_attention_heads")
    kv_heads = base.get("num_key_value_heads", heads)
    head_dim = base.get("head_dim") or (hidden // heads if hidden and heads else None)
    inter = base.get("intermediate_size")
    vocab = base.get("vocab_size")
    tie = base.get("tie_word_embeddings", True)
    return {
        "hidden": hidden, "layers": layers, "heads": heads,
        "kv_heads": kv_heads, "head_dim": head_dim, "inter": inter,
        "vocab": vocab, "tie": tie,
    }


def estimate_params(d):
    """Estimate dense-transformer parameter count from dims (SwiGLU MLP)."""
    hidden, layers, heads = d["hidden"], d["layers"], d["heads"]
    kv_heads, head_dim, inter = d["kv_heads"], d["head_dim"], d["inter"]
    vocab, tie = d.get("vocab"), d.get("tie", True)
    if not all([hidden, layers, heads, head_dim, inter]):
        return None
    kv_dim = kv_heads * head_dim
    q_dim = heads * head_dim
    attn = hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden
    mlp = 3 * hidden * inter  # gate + up + down (SwiGLU); use 2 for non-gated
    per_layer = attn + mlp
    total = layers * per_layer
    if vocab:
        total += vocab * hidden * (1 if tie else 2)
    return total


def fmt_b(n):
    return "%.2fB" % (n / 1e9)


def main():
    ap = argparse.ArgumentParser(description="Rough vLLM perf/memory estimator")
    ap.add_argument("--config", help="path to HF config.json")
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--inter", type=int)
    ap.add_argument("--vocab", type=int)
    ap.add_argument("--params-b", type=float, help="param count in billions (overrides estimate)")
    # precision
    ap.add_argument("--weight-bytes", type=float, default=2.0, help="bytes/param (2=bf16, 1=fp8, 0.5=int4)")
    ap.add_argument("--kv-bytes", type=float, default=2.0, help="bytes per KV element (2=fp16, 1=fp8)")
    # hardware
    ap.add_argument("--vram-gb", type=float, default=80.0)
    ap.add_argument("--util", type=float, default=0.9)
    ap.add_argument("--bw-tbs", type=float, default=3.35, help="HBM bandwidth TB/s (H100~3.35, MI300X~5.3)")
    ap.add_argument("--flops-tflops", type=float, default=989.0, help="dense tensor TFLOPS (H100 bf16~989, MI300X~1307)")
    ap.add_argument("--gpus", type=int, default=1, help="tensor-parallel GPUs (splits weights+KV+compute)")
    ap.add_argument("--mfu", type=float, default=0.4, help="prefill model-FLOPs utilization (0.3-0.5 typical)")
    # workload
    ap.add_argument("--ctx", type=int, default=2048, help="avg context length for concurrency + decode KV reads")
    ap.add_argument("--prefill", type=int, default=2048, help="prompt length for TTFT estimate")
    ap.add_argument("--batches", default="1,8,32,128", help="batch sizes for decode table")
    args = ap.parse_args()

    if args.config:
        d = load_config_dims(args.config)
    else:
        d = {"hidden": args.hidden, "layers": args.layers, "heads": args.heads,
             "kv_heads": args.kv_heads or args.heads, "head_dim": args.head_dim,
             "inter": args.inter, "vocab": args.vocab, "tie": True}
    if d.get("head_dim") is None and d.get("hidden") and d.get("heads"):
        d["head_dim"] = d["hidden"] // d["heads"]

    params = args.params_b * 1e9 if args.params_b else estimate_params(d)
    if not params:
        raise SystemExit("Need --params-b or full dims (--config or --hidden/--layers/--heads/--head-dim/--inter).")
    if not (d.get("layers") and d.get("kv_heads") and d.get("head_dim")):
        raise SystemExit("Need layers, kv_heads, head_dim for KV math (--config or flags).")

    bw = args.bw_tbs * 1e12          # bytes/s
    flops = args.flops_tflops * 1e12  # FLOP/s
    gpus = max(1, args.gpus)

    # --- Memory ---
    weight_bytes = params * args.weight_bytes
    weight_bytes_pg = weight_bytes / gpus
    kv_per_token = 2 * d["layers"] * d["kv_heads"] * d["head_dim"] * args.kv_bytes  # bytes/token (all layers)
    kv_per_token_pg = kv_per_token / gpus
    vram_pg = args.vram_gb * GB * args.util
    activation_reserve = 2 * GB  # rough headroom per GPU
    kv_pool_pg = vram_pg - weight_bytes_pg - activation_reserve
    max_tokens = max(0.0, kv_pool_pg * gpus / kv_per_token)
    max_seqs = max_tokens / args.ctx if args.ctx else 0

    print("# Rough vLLM estimate (order-of-magnitude)")
    print("")
    print("Model: ~%s params | %d layers | kv_heads=%s head_dim=%s"
          % (fmt_b(params), d["layers"], d["kv_heads"], d["head_dim"]))
    print("Precision: %.2g B/param weights, %.2g B/KV-elem | GPUs(TP)=%d"
          % (args.weight_bytes, args.kv_bytes, gpus))
    print("HW/GPU: %.0fGB x %.2f util, BW %.2f TB/s, %.0f TFLOPS dense"
          % (args.vram_gb, args.util, args.bw_tbs, args.flops_tflops))
    print("")
    print("## Memory")
    print("Weights total:        %.1f GB (%.1f GB/GPU)" % (weight_bytes / GB, weight_bytes_pg / GB))
    print("KV cache per token:   %.1f KB  (%.1f KB/GPU)" % (kv_per_token / 1024, kv_per_token_pg / 1024))
    print("KV pool (free):       %.1f GB total" % (kv_pool_pg * gpus / GB))
    if kv_pool_pg <= 0:
        print("  !! Weights do not fit with headroom on %d GPU(s); add GPUs or quantize." % gpus)
    print("Max KV tokens:        %s tokens" % ("{:,}".format(int(max_tokens))))
    print("~Max concurrent seqs: %d  (at ctx=%d)" % (int(max_seqs), args.ctx))
    print("")

    # --- Decode: memory-bandwidth bound ---
    # Per step you read weights once + KV of every active token in every seq.
    print("## Decode (tokens/s, memory-bandwidth bound, ctx=%d)" % args.ctx)
    print("  batch   step_ms   tokens/s   note")
    for b in [int(x) for x in args.batches.split(",") if x.strip()]:
        bytes_step = weight_bytes_pg + (b * kv_per_token_pg * args.ctx)
        step_s = bytes_step / bw
        toks = b / step_s
        note = "weight-bound" if (b * kv_per_token_pg * args.ctx) < weight_bytes_pg else "KV-bound"
        print("  %5d   %7.2f   %8.0f   %s" % (b, step_s * 1e3, toks, note))
    print("")

    # --- Critical batch size: decode GEMM flips memory->compute bound ---
    crit = (args.weight_bytes * flops) / (2 * bw)
    print("## Roofline")
    print("Critical batch ~%.0f tokens: below this decode is memory-bound" % crit)
    print("(so batching keeps helping until ~%.0f concurrent decodes)." % crit)
    print("")

    # --- Prefill: compute bound ---
    fwd_flops = 2 * params * args.prefill  # 2 FLOPs/param/token forward
    ttft_s = fwd_flops / (flops * gpus * args.mfu)
    print("## Prefill / TTFT (compute bound, prompt=%d, MFU=%.2f)" % (args.prefill, args.mfu))
    print("Forward FLOPs:        %.1f TFLOP" % (fwd_flops / 1e12))
    print("~TTFT (1 prompt):     %.0f ms" % (ttft_s * 1e3))
    print("~Prefill throughput:  %.0f tokens/s" % (args.prefill / ttft_s))
    print("")
    print("Estimates only; validate with `vllm bench serve`. Ignores fragmentation,")
    print("non-attention activations, comms overhead, MoE routing, and overlap.")


if __name__ == "__main__":
    main()
