# vLLM vs SGLang

## Use When

Use when choosing between the engines, porting a config/knowledge from one to the
other, or mapping a concept you know in vLLM to its SGLang equivalent (or vice
versa).

## Lesson

Both are high-throughput LLM serving engines built on the same physics
(PagedAttention-style KV, continuous batching, CUDA graphs, FP8, EAGLE spec
decode, TP/PP/EP, OpenAI API). They differ in heritage and a few signature
features.

### Conceptual Differences

- **KV reuse**: vLLM uses **block-hash automatic prefix caching**; SGLang uses
  **RadixAttention** (a radix tree) which also enables tree/`fork` sharing.
- **Frontend**: SGLang has a **programmable DSL** (`sgl.gen/select/fork`); vLLM has
  no frontend language (just `LLM`/OpenAI API).
- **Scheduler**: SGLang emphasizes a **zero-overhead overlap scheduler** (CPU/GPU
  overlap); vLLM V1 also isolates EngineCore in a process for low overhead.
- **Engine versioning**: vLLM had a major V0→V1 internals relocation (~0.8);
  SGLang's big step was the v0.4 zero-overhead scheduler. Code maps differ
  accordingly (`devmap/areas.jsonl` vs `devmap/sglang-areas.jsonl`).
- **Structured output**: SGLang pioneered fast compressed-FSM; both now support
  xgrammar/outlines.

### Argument / Term Mapping

| Intent | vLLM | SGLang |
| --- | --- | --- |
| KV memory fraction | `--gpu-memory-utilization` | `--mem-fraction-static` |
| Max concurrent seqs | `--max-num-seqs` | `--max-running-requests` |
| Per-step token budget | `--max-num-batched-tokens` | `--chunked-prefill-size` |
| Max context | `--max-model-len` | `--context-length` |
| Tensor parallel | `--tensor-parallel-size` | `--tp-size` |
| Data parallel | `--data-parallel-size` | `--dp-size` |
| Quantization | `--quantization` | `--quantization` |
| KV cache dtype | `--kv-cache-dtype` | `--kv-cache-dtype` |
| Attention backend | `VLLM_ATTENTION_BACKEND` (env) | `--attention-backend` (flag) |
| Disable graphs | `--enforce-eager` | `--disable-cuda-graph` |
| Prefix caching | `--enable-prefix-caching` (default on) | RadixAttention (default; `--disable-radix-cache`) |
| Spec decode | `--speculative-config` | `--speculative-algorithm` + `--speculative-*` |
| Grammar backend | `--guided-decoding-backend` | `--grammar-backend` |
| Env namespace | `VLLM_*` | `SGLANG_*` / `SGL_*` |
| Launch | `vllm serve <model>` | `python -m sglang.launch_server --model-path <model>` |
| Code: runtime | `vllm/v1/` | `python/sglang/srt/` |
| Code: args | `vllm/engine/arg_utils.py` | `python/sglang/srt/server_args.py` |
| Code: env | `vllm/envs.py` | `python/sglang/srt/environ.py` |

### Choosing

- Prefer **SGLang** for heavy prefix sharing / branching programs (agents, tree
  search, many-shot), the frontend DSL, and DeepSeek-class MoE at scale.
- Prefer **vLLM** for its very broad model/hardware coverage and ecosystem; both
  are excellent general servers.
- They are close in raw throughput; pick by workload shape, feature needs, and
  the hardware/model you must support — then benchmark both.

## Rules

- Do not copy flags verbatim between engines; translate via the table and confirm
  with each engine's `--help`.
- When porting tuning, keep the *intent* (grow KV pool, raise concurrency, protect
  decode latency); the knob names change.
- Benchmark each engine with its own best defaults before declaring a winner.

## Avoid

- Assuming `VLLM_*` env vars or vLLM flag names work in SGLang (and vice versa).
- Comparing the two without matching workload, precision, backend, and hardware.

## Related

- `knowledge/sglang/server-args.md`
- `knowledge/sglang/radixattention.md`
- `knowledge/serving/engine-args-reference.md` (vLLM)
- `knowledge/serving/performance-estimation.md` (engine-agnostic)

## Source

- vLLM and SGLang docs/source; release blogs
- https://docs.sglang.ai/ , https://docs.vllm.ai/
