# 000: Harness Bootstrap

## Status

Accepted

## Date

2026-06-28

## Context

RallyClip needs a structured harness to constrain agent behavior across sessions.

## Decision

Adopt the Learn Harness Engineering template pack.

## Consequences

All agent sessions must follow the `AGENTS.md` startup/shutdown protocol. Features are tracked in `feature_list.json`. Progress is persisted in `claude-progress.md`.
