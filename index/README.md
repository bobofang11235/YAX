# Index

Human entrypoint to YAX. Agents should use the runtime
(`python3 scripts/yax.py recommend "<task>"`) rather than this page.

## Start Here

- New to the repo? Read `../README.md` then `../AGENTS.md`.
- Need vLLM background? Browse `../knowledge/README.md`.
- Have a concrete task? Run:
  ```bash
  python3 scripts/yax.py recommend "<your task>"
  ```
- Developing vLLM and need the right files for your version?
  ```bash
  python3 scripts/yax.py where "<your problem>" -V <vllm-version>
  ```

## Quick Map

- Serve a model → `workflows/serve-new-model.md`
- Make it faster → `workflows/optimize-throughput.md`
- Wrong output → `workflows/debug-accuracy.md`
- Edit/extend vLLM → `workflows/contribute-feature.md`
- AMD/ROCm → `workflows/rocm-bringup.md`
- All engine flags → `knowledge/serving/engine-args-reference.md`
- All env vars → `knowledge/serving/environment-variables.md`

## Generated Registry

`registry/toolbox-index.json` (cards) and `registry/codemap-by-version.json`
(per-version code map) are generated. Rebuild both with
`python3 scripts/yax.py index`; never edit them by hand.
