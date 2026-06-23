# YAX

YAX is a retrieval-first **vLLM engineering harness**: a compact toolbox plus
knowledge base for working on [vLLM](https://github.com/vllm-project/vllm) as an
inference-engine developer and operator. It captures how to read and edit the
codebase, what features exist today, and which engine arguments and environment
variables matter on CUDA and ROCm.

It borrows its harness shape from the Zootopia toolbox: markdown is the source of
truth, a small Python runtime makes search cheap, and instructions, knowledge,
verification, scope, and lifecycle are kept in the repo instead of chat memory.

## Toolbox Loop

```text
task -> retrieve tools/workflows -> execute with validation -> save run -> update toolbox
```

```mermaid
flowchart TD
  task["vLLM task"] --> retrieve["python3 scripts/yax.py recommend '<task>'"]
  retrieve --> choose["Choose smallest matching workflow/tool"]
  choose --> context["Read linked knowledge only as needed"]
  context --> execute["Execute with the workflow validation gates"]
  execute --> validate{"Validation passed?"}
  validate -->|no| capture["Store raw notes / failed attempt in capture/ or runs/"]
  validate -->|yes| run["Save meaningful outcome in runs/"]
  run --> reusable{"Reusable improvement?"}
  reusable -->|yes| update["Update tool/workflow/knowledge"]
  update --> index["Regenerate index and validate"]
  reusable -->|no| done["Done"]
  index --> done
```

## How Agents Should Use It

1. Run `python3 scripts/yax.py recommend "<task>"` for any non-trivial vLLM task.
2. Read the top-ranked workflow first, then only the tools it references.
3. Follow linked `knowledge/` notes for deeper background (architecture, args,
   env vars, ROCm/CUDA specifics).
4. Execute with the workflow's validation gates (benchmark, accuracy check,
   `pytest`, smoke serve).
5. For meaningful tasks, record a run with
   `python3 scripts/yax.py new-run <slug>` and fill in the outcome.
6. If the run revealed a reusable improvement, update the narrowest artifact and
   regenerate the registry.

## Folders

```text
YAX/
├── AGENTS.md          # agent operating rules (read first)
├── README.md          # this file
├── scripts/yax.py     # toolbox runtime: recommend / search / validate / eval / index
├── templates/         # artifact templates
├── knowledge/         # compact vLLM knowledge library
│   ├── architecture/  # V1 engine internals, PagedAttention, scheduler, backends
│   ├── serving/       # engine args, env vars, features, quantization, perf
│   ├── rocm/          # ROCm/HIP build, AITER, FP8 fnuz, TunableOp
│   ├── cuda/          # CUDA build, FlashInfer, DeepGEMM, CUDA graphs, NCCL
│   └── development/   # repo layout, build, add a model, testing, codebase map
├── devmap/            # version-tagged code map (areas.jsonl + versions.json)
├── tools/             # canonical tool cards (compact capabilities)
├── workflows/         # reusable multi-tool task plans
├── runs/              # task lineage records
├── capture/           # raw source / task notes inbox
├── evals/             # golden routing cases for retrieval-quality eval
├── registry/          # generated machine-readable index
└── index/             # human entrypoint
```

## Runtime Commands

```bash
python3 scripts/yax.py recommend "serve a quantized model on 2 GPUs"
python3 scripts/yax.py where "preemption throughput collapse" -V 0.8.5  # version-aware code map
python3 scripts/yax.py where --list-areas                               # all code-map areas
python3 scripts/yax.py index          # rebuild registry (toolbox + codemap-by-version)
python3 scripts/yax.py validate       # check metadata + references
python3 scripts/yax.py eval           # retrieval/routing quality gate
python3 scripts/yax.py sync-status    # which vLLM version YAX reflects + how to refresh
python3 scripts/yax.py new-run <slug> # scaffold a run record
```

### Keeping YAX Current

YAX records which upstream vLLM point its knowledge reflects in
`devmap/sync-state.json` (`synced_to`), with a human log in `CHANGELOG.md`. To
update, you only review commits **newer than the last sync** rather than the whole
history:

```bash
python3 scripts/yax.py sync-status --vllm-path <vllm-clone>
# -> git log --oneline <synced_to>..origin/main
```

Then update the affected cards, bump `synced_to`, and add a CHANGELOG entry.

### Develop-By-Version Code Map

YAX's focus is developing vLLM. Because the tree moves between releases (notably
the V0→V1 engine relocation), `where` resolves the right paths **for your version
tag**:

```bash
python3 scripts/yax.py where "scheduler decides which requests run" -V 0.7.3
#   -> vllm/core/scheduler.py        (V0 layout)
python3 scripts/yax.py where "scheduler decides which requests run" -V latest
#   -> vllm/v1/core/sched/scheduler.py  (V1 layout)
```

Source of truth is `devmap/areas.jsonl`; the resolved per-version index is
generated to `registry/codemap-by-version.json`.

## Design Principles

- Markdown is the source of truth; the registry JSON is rebuildable.
- Correctness and reproducible benchmarks outrank novelty.
- Pin versions: vLLM, PyTorch, CUDA/ROCm, and driver move fast; record them.
- Knowledge cards stay compact; deep dives link to upstream source files.
- Failed attempts stay in `runs/` or `capture/` unless they teach a reusable
  lesson.
- Treat YAX as an agent harness: keep instructions, state, verification, scope,
  and lifecycle explicit and repo-local.

## Scope And Accuracy Note

vLLM evolves quickly. Cards describe the **V1 engine era** (default since vLLM
0.8.x) and flag where behavior is version-sensitive. Always confirm exact flag
names against the installed version (`vllm serve --help`) and the source of
truth in `vllm/engine/arg_utils.py` and `vllm/envs.py` before relying on a
detail.
