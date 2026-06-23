# ATOM Environment Variables

## Use When

Use to change ATOM behavior without code edits, and for ROCm/AITER runtime
tuning. Authoritative list: `docs/environment_variables.md` (all `ATOM_*`).

## Lesson

ATOM exposes `ATOM_*` environment variables (documented in
`docs/environment_variables.md`). Because ATOM is young and the set changes, treat
that doc (or `grep ATOM_ atom/`) as the source of truth rather than memorizing
names here.

### ATOM Vars

- See `docs/environment_variables.md` for the full, current `ATOM_*` list (kernel
  selection, profiling, scheduling, and feature toggles). Verify before relying on
  any specific name.

### ROCm / AITER Runtime (apply to ATOM)

ATOM is ROCm-only and AITER-based, so the AMD runtime knobs matter:

- `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` — GPU selection (not
  `CUDA_VISIBLE_DEVICES`).
- `NCCL_*` / RCCL: `NCCL_DEBUG=INFO`, `NCCL_P2P_DISABLE`, `NCCL_IB_DISABLE`.
- `PYTORCH_TUNABLEOP_ENABLED=1` — GEMM autotuning (hipBLASLt) on MI300.
- AITER-specific toggles (kernel backend ASM/CK/Triton selection) — check AITER's
  env and ATOM's docs.
- `HSA_OVERRIDE_GFX_VERSION` — coax unsupported gfx (experimental; e.g. Navi4).
- `HF_TOKEN`, `HF_HOME`, `huggingface_hub` login for gated models.

## Rules

- Read `docs/environment_variables.md` for exact `ATOM_*` names; this note
  intentionally avoids guessing them.
- Record every non-default env var with benchmarks/bug reports — they change
  kernels and numerics.
- ROCm/AITER env (TunableOp, RCCL, gfx override) is part of ATOM's performance
  story.

## Avoid

- Assuming `VLLM_*` or `SGLANG_*` vars work in ATOM; the namespace is `ATOM_*`.
- Using CUDA device-selection vars on ROCm.

## Related

- `knowledge/atom/configuration.md`
- `knowledge/atom/performance-tuning.md`
- `knowledge/vllm/rocm/rocm-setup-and-tuning.md` (shared ROCm background)

## Source

- `docs/environment_variables.md`
- https://github.com/ROCm/ATOM
