# SGLang Server Arguments

## Use When

Use when launching `python -m sglang.launch_server` or constructing `Engine` /
`ServerArgs`, or to find the flag that controls memory, parallelism, batching, or
a feature. Canonical list: `python/sglang/srt/server_args.py`; verify with
`python -m sglang.launch_server --help`.

## Lesson

Flags below are grouped by intent; defaults drift across versions.

### Model / Tokenizer

- `--model-path` (HF repo or local), `--tokenizer-path`, `--served-model-name`.
- `--dtype auto|bfloat16|float16`, `--context-length N` (max context; ~vLLM
  `--max-model-len`).
- `--trust-remote-code`, `--load-format`, `--revision`, `--random-seed`.
- `--device cuda|rocm|cpu|hpu|xpu`.

### Memory / KV / Cache

- `--mem-fraction-static` (fraction of VRAM for the static KV pool; ~vLLM
  `--gpu-memory-utilization`). Lower to avoid OOM, raise to grow KV.
- `--max-running-requests` (concurrency cap; ~vLLM `--max-num-seqs`).
- `--max-total-tokens` (KV pool token budget).
- `--chunked-prefill-size` (prefill chunk budget; ~vLLM `--max-num-batched-tokens`;
  `-1` disables chunked prefill).
- `--disable-radix-cache` (turn off RadixAttention).

### Parallelism / Distributed

- `--tp-size` / `--tensor-parallel-size`, `--dp-size`, `--ep-size`.
- `--enable-dp-attention` (DeepSeek MoE), `--enable-ep-moe`.
- Multi-node: `--dist-init-addr`, `--nnodes`, `--node-rank`, `--nccl-init-addr`.

### Scheduling / Performance

- `--schedule-policy lpm|fcfs|dfs-weight` (lpm maximizes prefix reuse).
- `--schedule-conservativeness` (lower = admit more aggressively).
- `--disable-overlap-schedule` (turn off the overlap/zero-overhead scheduler).
- `--disable-cuda-graph`, `--cuda-graph-max-bs`, `--cuda-graph-bs`.
- `--enable-torch-compile`, `--torch-compile-max-bs`.
- `--stream-interval`.

### Backends

- `--attention-backend flashinfer|triton|fa3|torch_native` (flashinfer default on
  CUDA).
- `--sampling-backend flashinfer|pytorch`.
- `--grammar-backend xgrammar|outlines|llguidance`.

### Quantization

- `--quantization fp8|awq|gptq|...`, `--kv-cache-dtype auto|fp8_e5m2|fp8_e4m3`.

### Features

- Speculative: `--speculative-algorithm EAGLE|EAGLE3`,
  `--speculative-num-steps`, `--speculative-num-draft-tokens`,
  `--speculative-eagle-topk`.
- LoRA: `--lora-paths`, `--max-loras-per-batch`.
- Tools/reasoning: `--tool-call-parser <name>`, `--reasoning-parser <name>`.

### Server

- `--host`, `--port`, `--api-key`, `--chat-template`, `--log-level`.

## Rules

- Start minimal: `python -m sglang.launch_server --model-path <m> --tp-size <n>`;
  add flags only to fix a measured problem.
- Map from vLLM intent: gpu-mem-util → `--mem-fraction-static`, max-num-seqs →
  `--max-running-requests`, max-num-batched-tokens → `--chunked-prefill-size`,
  max-model-len → `--context-length`.
- Confirm names against the installed version; SGLang flags evolve quickly.

## Avoid

- Copying flag sets across versions without `--help`.
- Disabling the overlap scheduler / CUDA graph / radix cache in production without
  a measured reason (each is a default perf win).

## Related

- `knowledge/sglang/environment-variables.md`
- `knowledge/sglang/performance-tuning.md`
- `knowledge/sglang/vllm-vs-sglang.md`

## Source

- `python/sglang/srt/server_args.py`
- https://docs.sglang.ai/backend/server_arguments.html
