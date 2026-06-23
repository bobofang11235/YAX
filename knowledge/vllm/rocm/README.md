# ROCm Knowledge

Running and tuning vLLM on AMD GPUs (ROCm/HIP), focused on MI200/MI300-class
hardware.

## Notes

- `rocm-setup-and-tuning.md`: build/install, AITER, TunableOp, hipBLASLt, RCCL,
  attention backend, key env vars.
- `rocm-fp8-formats.md`: fp8 fnuz (ROCm) vs OCP e4m3 (CUDA) and why it matters.

## Source Of Truth

- `vllm/platforms/rocm.py`, `Dockerfile.rocm`, ROCm sections of `vllm/envs.py`.
- https://docs.vllm.ai/en/latest/getting_started/installation/gpu/rocm.html
