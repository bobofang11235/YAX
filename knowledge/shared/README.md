# Shared Knowledge (Engine-Agnostic)

Notes that apply to any LLM inference engine (vLLM, SGLang, ...). Engine-specific
detail lives under `knowledge/vllm/` and `knowledge/sglang/`.

## Notes

- `performance-estimation.md`: compute rough memory / max-concurrency / decode
  throughput / TTFT from a model config plus hardware specs (with
  `scripts/perfcalc.py`). Pure roofline math — engine-independent.
- `performance-factors.md`: the full catalog of levers that move throughput and
  latency (model/precision, memory/KV, kernels, batching, compilation,
  parallelism, hardware, workload shape, feature overheads, host, cold start).

## Maintenance

- Keep these free of engine-specific flag names; reference each engine's serving
  notes for the concrete knobs.
- Start new notes from `templates/knowledge-note.md`.
