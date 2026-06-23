# Index

Human entrypoint to YAX. Agents should use the runtime
(`python3 scripts/yax.py recommend "<task>"`) rather than this page.

## Start Here

- New to the repo? Read `../README.md` then `../AGENTS.md`.
- Need engine background? Browse `../knowledge/README.md` (vllm / sglang / shared).
- Have a concrete task? Run:
  ```bash
  python3 scripts/yax.py recommend "<your task>"
  ```
- Developing and need the right files for your version?
  ```bash
  python3 scripts/yax.py where "<your problem>" -V <version>            # vLLM
  python3 scripts/yax.py where "<your problem>" --engine sglang -V 0.5.13
  python3 scripts/yax.py where "<your problem>" --engine atom   -V 0.1.5
  ```

## Quick Map

### vLLM
- Serve a model → `workflows/vllm/serve-new-model.md`
- Make it faster → `workflows/vllm/optimize-throughput.md`
- Wrong output → `workflows/vllm/debug-accuracy.md`
- Edit/extend vLLM → `workflows/vllm/contribute-feature.md`
- AMD/ROCm → `workflows/vllm/rocm-bringup.md`
- All flags / env vars → `knowledge/vllm/serving/engine-args-reference.md`,
  `knowledge/vllm/serving/environment-variables.md`

### SGLang
- Serve a model → `workflows/sglang/serve-sglang-model.md`
- Hub / args / env → `knowledge/sglang/README.md`,
  `knowledge/sglang/server-args.md`, `knowledge/sglang/environment-variables.md`
- RadixAttention / DSL → `knowledge/sglang/radixattention.md`,
  `knowledge/sglang/frontend-language.md`

### ATOM (AMD ROCm / AITER)
- Serve a model → `workflows/atom/serve-atom-model.md`
- Hub / AITER ops → `knowledge/atom/README.md`, `knowledge/atom/aiter-model-ops.md`
- Config / env / quant → `knowledge/atom/configuration.md`,
  `knowledge/atom/environment-variables.md`, `knowledge/atom/quantization.md`
- Distributed / TBO / P-D → `knowledge/atom/distributed-tbo.md`

### Shared
- Estimate memory/throughput → `knowledge/shared/performance-estimation.md`
- Performance factors catalog → `knowledge/shared/performance-factors.md`

## Generated Registry

`registry/toolbox-index.json` (cards) plus per-engine code maps
(`registry/<engine>-codemap-by-version.json` for vllm / sglang / atom) are
generated. Rebuild all with `python3 scripts/yax.py index`; never edit by hand.
