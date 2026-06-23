---
id: oom-kv-triage
kind: tool
title: OOM And KV Cache Triage
status: active
tags: [oom, memory, kv-cache, preemption, max-model-len, gpu-memory-utilization, debug]
aliases: [out of memory, cannot fit max_model_len, preemption, kv cache too small, oom]
capabilities:
  - classify an OOM as load-time vs KV-pool vs runtime spike
  - pick the right memory lever
inputs:
  - the failing launch/run log and the model + GPU config
outputs:
  - a fix (flag/env change) and the reasoning
related:
  - workflow:debug-accuracy
  - tool:tune-serving-config
source:
  - knowledge/architecture/paged-attention-kv-cache.md
---

# OOM And KV Cache Triage

## Use When

Use on OOM at startup, "engine cannot fit max_model_len", or throughput collapse
from preemption.

## How To Use

1. Classify from the log:
   - OOM at load → lower `--gpu-memory-utilization`, quantize, or add GPUs.
   - "cannot fit max_model_len" → KV pool too small: lower `--max-model-len`,
     `--kv-cache-dtype fp8`, or raise utilization with headroom.
   - "Sequence preempted" repeatedly → lower `--max-num-seqs`/`--max-num-batched-
     tokens`, or add memory.
   - mid-run OOM → utilization too high; leave activation headroom.
2. Apply one lever; re-launch and re-check the log.

## Validation

- Startup shows a KV capacity >= max_model_len and no preemption under load.

## Constraints

- FP8 KV trades a little accuracy; validate.

## Related

- `knowledge/architecture/paged-attention-kv-cache.md`
- `tool:tune-serving-config`

## Source

- `knowledge/architecture/paged-attention-kv-cache.md`
