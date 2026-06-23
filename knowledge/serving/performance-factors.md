# Performance Factors (What Moves vLLM Throughput/Latency)

## Use When

Use to enumerate every major lever that affects vLLM performance, to find
candidate causes of a slowdown, or to know what to sweep in a tuning study.
`performance-tuning.md` gives the *order* to tune; this note is the *catalog*.

## Lesson

Performance = how well the GPU stays busy with useful FLOPs vs stalling on
memory, communication, host overhead, or waiting. The levers below are roughly
ordered by how often they dominate. Always measure with `vllm bench serve`.

### 1. Model, Precision, Quantization

- **Model size / architecture.** Params set compute and weight bandwidth. **GQA/MQA
  vs MHA** changes KV size dramatically (fewer KV heads → more concurrency). MoE:
  only active experts compute, but all experts occupy memory.
- **Compute dtype.** bf16/fp16 standard; fp32 is far slower. 
- **Weight quantization** (AWQ/GPTQ/FP8/INT8): less weight bandwidth → faster
  decode + frees VRAM for KV. Kernel matters: **Marlin/Machete** (Ampere+),
  **DeepGEMM** FP8 (Hopper), cutlass.
- **Activation FP8 (W8A8)** speeds compute on Hopper/MI300.
- **KV cache dtype** (`--kv-cache-dtype fp8`): ~half KV memory → more concurrency,
  small accuracy cost.

### 2. Memory / KV Cache

- **`--gpu-memory-utilization`**: bigger KV pool → more concurrent sequences →
  more throughput (leave headroom to avoid mid-run OOM).
- **`--max-model-len`**: caps per-seq KV; lower it to fit more sequences.
- **Automatic prefix caching**: reuses KV for shared prefixes (system prompts,
  few-shot, agent loops) → can cut TTFT/compute massively; near-zero benefit if
  prefixes never repeat.
- **`--block-size`**, swap/CPU-offload: offload is a last-resort and slow.

### 3. Attention & Sampling Kernels

- **Attention backend** (`VLLM_ATTENTION_BACKEND`): FlashAttention vs FlashInfer
  vs Triton vs xFormers; **FA2 vs FA3**; **MLA** backends for DeepSeek. Decode-heavy
  workloads often favor FlashInfer on CUDA.
- **Sampler kernels**: `VLLM_USE_FLASHINFER_SAMPLER`, fused penalties.
- **ROCm**: `VLLM_ROCM_USE_AITER`, custom paged attention.

### 4. Batching & Scheduling

- `--max-num-batched-tokens` (main throughput/latency knob), `--max-num-seqs`,
  chunked prefill, scheduling policy, preemption. See `scheduler-batching.md` and
  `scheduler-model-size-impact.md`.

### 5. Compilation & CUDA Graphs

- **CUDA graphs** (on unless `--enforce-eager`): remove per-kernel launch overhead;
  biggest win at small/medium batch and small models. `--enforce-eager` is a
  common accidental slowdown.
- **torch.compile level** (`-O`/`--compilation-config`): more fusion, longer
  warmup; piecewise CUDA-graph capture. Capture batch sizes affect coverage.

### 6. Parallelism & Communication

- **TP size**: shards layers but adds an all-reduce **every layer**; great on
  **NVLink**, much worse over PCIe. Don't over-shard a model that fits on fewer
  GPUs.
- **PP**: enables cross-node / large models but adds **pipeline bubbles**; needs
  enough in-flight requests to stay full.
- **DP / EP**: replicas; EP shards MoE experts.
- **Executor** (`mp` vs `ray`), **custom all-reduce**, NCCL/RCCL tuning, NVLink vs
  InfiniBand topology, P2P access.

### 7. Hardware & Platform

- GPU arch + **memory bandwidth** (decode is bandwidth-bound), NVLink/PCIe gen,
  number of GPUs.
- CUDA/ROCm/**driver** versions; clock/power caps and **thermal throttling**.
- CUDA: `VLLM_USE_DEEP_GEMM`. ROCm: `PYTORCH_TUNABLEOP_ENABLED`, hipBLASLt.

### 8. Workload Shape (often underestimated)

- **Input/output length distribution** and **request rate**: prefill-heavy vs
  decode-heavy changes the optimal config entirely.
- **Shared prefixes** → prefix-caching wins.
- **`n` / `best_of` / beam search**: more sequences per request → more KV + compute.
- **Sampling params**: large logprobs / top-k add cost.

### 9. Feature Overheads

- **Speculative decoding**: cuts single-stream latency when draft accept rate is
  high; can *reduce* aggregate throughput at high batch or low acceptance.
- **Structured/guided decoding**: grammar compile (first request spike) + per-step
  token-mask overhead.
- **LoRA / multi-LoRA**: punica kernels + adapter switching overhead.
- **Multimodal**: vision/audio encoder cost, input preprocessing, `--limit-mm-per-
  prompt`, mm input cache.

### 10. Serving / Host-Side

- **Python / GIL on the hot path**; V1 runs EngineCore in a separate process to
  reduce this — keep multiprocessing enabled.
- **Tokenization/detokenization**, chat-template application, **JSON
  serialization**, streaming, uvicorn/HTTP overhead.
- `--disable-log-requests`, avoid debug env (`VLLM_TRACE_FUNCTION`, `NCCL_DEBUG`).
- CPU threads / **NUMA binding**, PCIe transfer of inputs/outputs.

### 11. Cold Start / Warmup

- First requests pay weight load, **CUDA-graph capture**, torch.compile, and
  grammar compile. Warm up before benchmarking; exclude warmup from numbers.

## Rules

- Identify the bottleneck class first (memory / compute / comms / host / cold
  start) before turning knobs; the right lever depends on which one binds.
- Measure one change at a time with a realistic workload; record arch + versions +
  backend + flags.
- Decode bottlenecks → bandwidth/quantization/backend; prefill/TTFT → chunked
  prefill + prefix caching; scaling out → watch TP comms and PP bubbles.

## Avoid

- `--enforce-eager` in production; debug env vars left on; `gpu-memory-utilization`
  near 1.0.
- Over-sharding (TP) a model that fits on fewer GPUs, especially without NVLink.
- Benchmarking including cold-start/warmup, or across mismatched versions/backends.

## Related

- `knowledge/serving/performance-tuning.md`
- `knowledge/architecture/scheduler-model-size-impact.md`
- `knowledge/architecture/paged-attention-kv-cache.md`
- `knowledge/serving/quantization.md`
- `knowledge/rocm/rocm-setup-and-tuning.md`
- `knowledge/cuda/cuda-setup-and-tuning.md`

## Source

- `benchmarks/`, `vllm bench serve`
- https://docs.vllm.ai/en/latest/performance/optimization.html
