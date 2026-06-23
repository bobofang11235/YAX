# CUDA Knowledge

Running and tuning vLLM on NVIDIA GPUs (CUDA), focused on Ampere/Ada/Hopper.

## Notes

- `cuda-setup-and-tuning.md`: install, FlashInfer, DeepGEMM, CUDA graphs,
  capability/dtype checks, NCCL, key env vars.

## Source Of Truth

- `vllm/platforms/cuda.py`, `Dockerfile`, CUDA sections of `vllm/envs.py`.
- https://docs.vllm.ai/en/latest/getting_started/installation/gpu/cuda.html
