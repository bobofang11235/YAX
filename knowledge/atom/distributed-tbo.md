# ATOM Distributed, TBO, And P/D Disaggregation

## Use When

Use when scaling ATOM across GPUs/nodes, tuning MoE expert parallelism, or
configuring prefill/decode disaggregation. Authoritative: `docs/distributed_guide.md`.

## Lesson

### Parallelism

- **TP** (`-tp N`): shard layers across GPUs.
- **DP**: replicas; pairs with EP for MoE.
- **EP**: shard MoE experts, using **MORI all-to-all** for expert routing comms.

### Two-Batch Overlap (TBO)

Following DeepSeek's system design, **TBO** splits each batch into **two
micro-batches** and pipelines them across compute and communication streams. This
**hides expert-parallel (all-to-all) communication latency** and lowers peak
memory. Best for MoE models with DP attention; see the TBO recipe.

### Prefill/Decode (P/D) Disaggregation

- Run **prefill and decode on separate GPU nodes**, moving KV across with RDMA:
  - **MORI-IO** transport;
  - **Mooncake** push-mode KV cache transfer.
- Supports large MoE (DeepSeek-R1, DeepSeek-V4-Pro). Lets you size prefill and
  decode pools independently for throughput.

## Rules

- For MoE at scale, combine EP + DP attention + TBO; measure comms overlap (TBO
  should hide most all-to-all latency).
- Use P/D disaggregation when prefill and decode have very different resource
  profiles or you need independent scaling; otherwise co-located is simpler.
- Record TP/DP/EP layout, TBO on/off, and the transport (MORI-IO/Mooncake) with
  any benchmark — they dominate MoE results.
- Start from a recipe (DeepSeek-R1, Qwen3-235B, TBO, PD) rather than from scratch.

## Avoid

- Enabling P/D disaggregation without the RDMA fabric/transport configured.
- Comparing MoE throughput without noting TBO and EP settings.

## Related

- `knowledge/atom/performance-tuning.md`
- `knowledge/atom/features-overview.md`
- `knowledge/atom/recipes` (see `devmap` area `recipes`)

## Source

- `docs/distributed_guide.md`; ATOM README (TBO, PD recipes)
- https://github.com/ROCm/ATOM
