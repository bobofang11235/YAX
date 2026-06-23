# Adding A Model

## Use When

Use when vLLM reports an unsupported architecture, or you must implement/port a
new model into vLLM.

## Lesson

A vLLM model is a PyTorch module that consumes the engine's batched inputs and
KV cache and produces hidden states/logits. Most decoder LLMs reuse existing
attention/MLP/MoE layers, so a new model is often a config-mapping + wiring job,
not new kernels.

### Steps

1. **Confirm it is missing.** Check `vllm/model_executor/models/registry.py` and
   the supported-models docs. The `architectures` field in the HF `config.json`
   is the key vLLM matches.
2. **Create the model file** in `vllm/model_executor/models/<name>.py`. Start from
   the closest existing model (e.g. `llama.py`, `qwen2.py`, `mixtral.py`).
3. **Implement the module:** `__init__(vllm_config, prefix)`, the decoder layers
   using vLLM's `Attention`, RMSNorm, RoPE, and MLP/MoE layers, and a `forward`
   taking `input_ids`, `positions`, and KV cache; expose `compute_logits` and a
   `load_weights` that maps HF checkpoint names to module params.
4. **Register** the class in `registry.py` mapping the HF `architectures` name.
5. **Support interfaces** as needed: `SupportsLoRA`, `SupportsPP` (pipeline),
   `SupportsMultiModal`, quantization compatibility.
6. **Test:** load it, run a short generation, and A-B against HF Transformers
   logits/output on the same prompt to confirm correctness.

### Multimodal Additions

- Add an input processor / multimodal mapper and implement `SupportsMultiModal`;
  wire image/audio embeddings into the language model's input embeddings.

## Rules

- Reuse vLLM layers (`Attention`, `RowParallelLinear`/`ColumnParallelLinear`,
  fused MoE) so TP/quantization/backends work for free.
- Get the `load_weights` name mapping exactly right — silent mismaps produce
  garbage output, not errors.
- Validate first-token logits against HF Transformers (greedy) before benchmarking.
- Make tensor-parallel sharding explicit via the parallel linear layers.

## Avoid

- Writing custom attention when an existing backend covers the model.
- Skipping the HF-vs-vLLM logit A-B; "looks fluent" is not correctness.

## Related

- `knowledge/vllm/architecture/vllm-architecture-overview.md`
- `knowledge/vllm/development/custom-ops-and-platforms.md`
- `tools/vllm/development/add-new-model.md`

## Source

- `vllm/model_executor/models/` and `registry.py`
- https://docs.vllm.ai/en/latest/contributing/model/
