# Feature Overview (What vLLM Supports)

## Use When

Use to check whether vLLM can do something today and which flag turns it on,
before designing a workaround.

## Lesson

vLLM is a full serving stack, not just a kernel. The major feature areas and how
to enable them:

### Throughput / Memory Core

- **Continuous batching** — always on (iteration-level scheduling).
- **PagedAttention KV cache** — always on.
- **Automatic prefix caching** — `--enable-prefix-caching` (default on, V1).
- **Chunked prefill** — `--enable-chunked-prefill` (default on, V1).
- **CUDA graphs / torch.compile** — on unless `--enforce-eager`; tune with `-O`.

### Model Coverage

- **Decoder LLMs** — Llama, Qwen, Mistral/Mixtral, Gemma, Phi, DeepSeek, GPT-OSS,
  Granite, Command-R, and many more.
- **MoE** — Mixtral, DeepSeek-V2/V3, Qwen-MoE; expert parallel via
  `--enable-expert-parallel`.
- **Multimodal (vision/audio)** — LLaVA, Qwen-VL, Pixtral, InternVL, Phi-vision,
  Whisper, etc.; control with `--limit-mm-per-prompt`.
- **Embedding / pooling / reward / classification** — `--task embed|score|reward|
  classify` (a.k.a. `--runner pooling`).
- **Encoder-decoder** — partial (e.g. Whisper, some seq2seq).

### Decoding / Sampling

- **Sampling** — temperature, top-p/top-k/min-p, presence/frequency penalties,
  logprobs, seed, n / best_of.
- **Beam search** — supported (via the beam-search path).
- **Guided / structured output** — JSON-schema, regex, grammar, choice via
  `--guided-decoding-backend xgrammar|outlines|guidance`.
- **Tool / function calling** — `--enable-auto-tool-choice --tool-call-parser
  <name>` in the OpenAI server.
- **Speculative decoding** — draft model, n-gram, EAGLE/EAGLE3, Medusa, MTP via
  `--speculative-config`.

### Adapters / Fine-tune Serving

- **LoRA / multi-LoRA** — `--enable-lora` with `--max-loras`, `--max-lora-rank`;
  hot-swappable adapters per request.
- **Prompt adapters** — `--enable-prompt-adapter`.

### Quantization

- Weights/activations: FP8, AWQ, GPTQ, GGUF, bitsandbytes, INT8 (W8A8),
  Marlin/Machete kernels, AutoRound, compressed-tensors.
- KV cache: `--kv-cache-dtype fp8`.
- See `quantization.md`.

### Distributed / Scale

- TP, PP, DP, EP (see architecture overview).
- **Disaggregated prefill/decode (P/D)** and **KV transfer/connectors** for
  cross-instance KV movement.
- Ray or multiprocessing executors; multi-node serving.

### Serving Surface

- **OpenAI-compatible API**: `/v1/completions`, `/v1/chat/completions`,
  `/v1/embeddings`, `/v1/models`, plus metrics and health endpoints.
- **Prometheus metrics**, request logging, streaming, and a built-in benchmark
  CLI (`vllm bench serve` / `benchmarks/`).

## Rules

- Confirm a model is supported in `vllm/model_executor/models/registry.py` (or the
  supported-models docs) before debugging "unsupported architecture".
- Feature availability is version-dependent; check the release notes for the
  installed version.

## Avoid

- Assuming a feature flag exists with the same name across versions — verify with
  `vllm serve --help`.

## Related

- `knowledge/vllm/serving/engine-args-reference.md`
- `knowledge/vllm/serving/quantization.md`
- `knowledge/vllm/development/adding-a-model.md`

## Source

- `vllm/model_executor/models/registry.py`
- https://docs.vllm.ai/en/latest/features/
- https://docs.vllm.ai/en/latest/models/supported_models.html
