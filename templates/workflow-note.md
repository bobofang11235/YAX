---
id: <workflow-id>
kind: workflow
title: <Workflow Title>
status: draft
tags: [tag]
aliases: [alias]
tools:
  - tool:<tool-id>
triggers:
  - <task trigger>
validation:
  - <validation gate>
related:
  - tool:<tool-id>
source:
  - <url-or-repo-relative-path>
---

# <Workflow Title>

## Use When

Describe the recurring task shape.

## Workflow

1. Step one.
2. Step two.

## Tool Sequence

- `tool:<tool-id>`: why it is used.

## Validation Gates

- 

## Update Policy

Update this workflow when a run finds a better tool, ordering, validation gate,
or reusable failure pattern.

## Related Runs

- 

## Source

- 
