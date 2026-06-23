# Engine Arguments Reference

## Use When

Use when launching `vllm serve` or constructing `LLM(...)` / `EngineArgs`, or when
you need to know which flag controls memory, parallelism, batching, or a feature.

## Lesson

All engine behavior is configured through `EngineArgs` (offline `LLM`) /
`AsyncEngineArgs` (server). The CLI `vllm serve <model> --flag value` maps 1:1 to
these. The canonical list is `vllm/engine/arg_utils.py`; always confirm with
`vllm serve --help` for the installed version. Flags below are grouped by intent;
defaults drift across versions, so treat them as guidance.

### Model And Tokenizer

- `model` (positional): HF repo id or local path.
- `--tokenizer`: override tokenizer path. `--tokenizer-mode auto|slow|mistral`.
- `--dtype auto|float16|bfloat16|float32`: compute dtype.
- `--max-model-len`: max context (prompt + output) tokens. Caps KV usage.
- `--trust-remote-code`: allow custom modeling code from the HF repo.
- `--revision` / `--code-revision`: pin model / code git revision.
- `--load-format auto|safetensors|pt|bitsandbytes|gguf|runai_streamer|...`.
- `--config-format`, `--hf-overrides`: override HF config fields.
- `--seed`: RNG seed for reproducibility.

### Memory And KV Cache

- `--gpu-memory-utilization` (default ~0.9): fraction of VRAM for weights +
  activations + KV pool. Lower to avoid OOM; raise to grow KV cache.
- `--kv-cache-dtype auto|fp8|fp8_e4m3|fp8_e5m2`: quantize KV cache (~half memory).
- `--block-size 8|16|32|...`: KV block granularity (backend-constrained).
- `--swap-space` (GiB): CPU swap space for preempted sequences (V0-style).
- `--cpu-offload-gb`: offload this many GiB of weights to CPU RAM.
- `--max-num-seqs`: max concurrent sequences per step.
- `--max-num-batched-tokens`: max tokens processed per step (key throughput knob).

### Parallelism / Distributed

- `--tensor-parallel-size` / `-tp`: shard each layer across N GPUs.
- `--pipeline-parallel-size` / `-pp`: split layers into N stages.
- `--data-parallel-size` / `-dp`: N engine replicas (pairs with EP for MoE).
- `--enable-expert-parallel`: shard MoE experts across the TP/DP group.
- `--distributed-executor-backend mp|ray|uni`: process vs Ray vs single.

### Scheduling / Batching

- `--enable-chunked-prefill` / `--no-enable-chunked-prefill`.
- `--enable-prefix-caching` / `--no-enable-prefix-caching`.
- `--scheduling-policy fcfs|priority`.
- `--max-num-partial-prefills`, `--long-prefill-token-threshold` (chunked-prefill
  fairness for long prompts).

### Performance / Compilation

- `--enforce-eager`: disable CUDA graphs / compilation (slower, less memory, easier
  debugging). Omit it to keep graphs for speed.
- `-O` / `--compilation-config`: torch.compile / piecewise CUDA-graph level (e.g.
  `-O3`). Higher = more fusion, longer warmup.
- `--quantization` / `-q`: see below and `quantization.md`.

### Features

- `--enable-lora`, `--max-loras`, `--max-lora-rank`, `--lora-modules`.
- `--enable-prompt-adapter` (prompt/prefix adapters).
- `--speculative-config '{...}'`: speculative decoding (draft model / ngram / EAGLE /
  MTP). (Older builds: `--speculative-model`, `--num-speculative-tokens`.)
- `--guided-decoding-backend xgrammar|outlines|guidance|auto`: structured outputs.
- `--limit-mm-per-prompt '{"image": N}'`: multimodal input caps.
- `--rope-scaling`, `--rope-theta`: context-length extension.

### Server (OpenAI API)

- `--host`, `--port`, `--api-key`, `--served-model-name`.
- `--max-logprobs`, `--chat-template`, `--response-role`.
- `--enable-auto-tool-choice`, `--tool-call-parser <name>`.
- `--disable-log-requests`, `--uvicorn-log-level`.

## Rules

- Start minimal: `vllm serve <model> -tp <gpus>`; add flags only to fix a measured
  problem.
- Change one flag per experiment and benchmark it.
- Memory problems → memory group; speed → batching + compilation; correctness →
  dtype / quantization / backend.

## Avoid

- Copying flag sets between vLLM versions without checking `--help`; names and
  defaults change (e.g. spec-decode moved to `--speculative-config`).
- Setting `--enforce-eager` in production to "be safe" — it leaves throughput on
  the table; use it only for debugging.

## Related

- `knowledge/vllm/serving/environment-variables.md`
- `knowledge/vllm/serving/performance-tuning.md`
- `knowledge/vllm/serving/quantization.md`
- `knowledge/vllm/architecture/scheduler-batching.md`

## Source

- `vllm/engine/arg_utils.py`
- https://docs.vllm.ai/en/latest/serving/engine_args.html
- https://docs.vllm.ai/en/latest/models/engine_args.html
