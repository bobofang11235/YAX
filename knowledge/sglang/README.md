# SGLang Knowledge

Compact knowledge for developing and operating
[SGLang](https://github.com/sgl-project/sglang), a high-throughput LLM/VLM
serving framework. YAX began vLLM-focused; SGLang is the second engine and shares
the same harness shape (knowledge + version-tagged code map + sync state).

## Notes

- `architecture.md`: the SRT runtime, the multi-process design, scheduler, and
  request lifecycle.
- `radixattention.md`: SGLang's signature automatic KV reuse (radix tree).
- `server-args.md`: `ServerArgs` / `sglang.launch_server` flags.
- `environment-variables.md`: `SGLANG_*` / `SGL_*` and external runtime vars.
- `features-overview.md`: what SGLang supports today.
- `frontend-language.md`: the SGLang DSL (`sgl.gen/select/fork`).
- `performance-tuning.md`: tuning order for SGLang.
- `vllm-vs-sglang.md`: how the two engines differ (and arg/term mapping).

## Code Map

Use the version-aware indexer:

```bash
python3 scripts/yax.py where "<problem>" --engine sglang -V 0.5.13
python3 scripts/yax.py where --engine sglang --list-areas
python3 scripts/yax.py sync-status --engine sglang
```

## Source Of Truth

- Runtime: `python/sglang/srt/` (the "SRT"); frontend DSL: `python/sglang/lang/`.
- Args: `python/sglang/srt/server_args.py`; verify with
  `python -m sglang.launch_server --help`.
- https://docs.sglang.ai/
