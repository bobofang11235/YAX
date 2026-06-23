# SGLang Frontend Language (DSL)

## Use When

Use when programming multi-call LLM workflows with SGLang's Python DSL, or when
deciding whether to use the frontend vs the plain OpenAI API.

## Lesson

SGLang ships a **frontend language**: a Python DSL for expressing LLM "programs"
with control flow, parallelism, and structured generation, executed against the
SGLang runtime (or other backends). It is what makes SGLang more than a server —
and it is **RadixAttention-aware**, so shared prefixes and `fork` branches reuse
KV automatically.

### Primitives

- `@sgl.function` — declare a program: `def f(s, ...): ...` where `s` is the prompt
  state.
- `s += "text"` / `s += sgl.user("...")` — append to the prompt.
- `sgl.gen("name", max_tokens=..., stop=...)` — generate into a variable.
- `sgl.select("name", choices=[...])` — constrained choice (efficient via the
  runtime).
- `s.fork(k)` — branch the state into k parallel continuations sharing the parent
  KV (great for parallel sampling / tree-of-thought).
- `sgl.image(...)`, regex/JSON constraints, and streaming.

### Execution

- Set a backend: `sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))`
  (a running `launch_server`) or an OpenAI backend.
- Run one: `state = f.run(...)`; batch: `f.run_batch([...])`.
- Code: `python/sglang/api.py` (surface), `python/sglang/lang/interpreter.py`
  (executor), `python/sglang/lang/ir.py` (IR).

### Why It Helps

- `fork` + RadixAttention make branching programs cheap (shared prefix KV).
- `select`/constraints are evaluated efficiently on the server.
- Encourages structured, reusable prompt programs vs ad-hoc string building.

## Rules

- Use the DSL when you have multi-step / branching / structured generation; use the
  plain OpenAI API for simple single completions.
- Point the DSL at a `RuntimeEndpoint` to get RadixAttention reuse across calls.
- Keep heavy logic on the server side (gen/select) rather than round-tripping.

## Avoid

- Confusing the frontend (`python/sglang/lang`) with the serving runtime
  (`python/sglang/srt`) — engine/serving work is in `srt`.

## Related

- `knowledge/sglang/radixattention.md`
- `knowledge/sglang/architecture.md`
- `knowledge/sglang/vllm-vs-sglang.md`

## Source

- `python/sglang/api.py`, `python/sglang/lang/`
- https://docs.sglang.ai/frontend/frontend.html
