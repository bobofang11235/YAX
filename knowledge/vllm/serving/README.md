# Serving Knowledge

Operating vLLM: the arguments you pass, the environment variables that change
behavior, the features available, and how to tune for throughput, latency, and
memory.

## Notes

- `engine-args-reference.md`: grouped `EngineArgs` / `vllm serve` flags.
- `environment-variables.md`: `VLLM_*` and external runtime env vars (CUDA + ROCm).
- `features-overview.md`: what vLLM supports today and the flag to enable each.
- `quantization.md`: weight/activation/KV quantization methods and when to use them.
- `performance-tuning.md`: a tuning order for throughput / latency / memory.
- `performance-factors.md`: the full catalog of levers that move performance.
- `performance-estimation.md`: compute rough memory/throughput/TTFT numbers from
  the model config (with `scripts/perfcalc.py`).

## Source Of Truth

- Args: `vllm/engine/arg_utils.py` and `vllm serve --help`.
- Env vars: `vllm/envs.py`.
- Always verify exact names against the installed version.
