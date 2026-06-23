# Cursor Config For YAX

## Auto-commit hook

`hooks.json` registers a `stop` hook that runs `hooks/auto-commit.ps1` when an
agent turn finishes. The script:

- commits all changes in this repo with a message summarizing the changed
  top-level areas and file count, then `git push`es;
- no-ops when the tree is clean (no empty commits);
- **fails open** — it never blocks the agent if git/push fails.

### Credentials / security

The hook uses **only your existing cached git (HTTPS) credentials**. It does
**not** create, read, add, or upload any SSH or GitHub key, never runs
`gh auth`, and never edits `~/.ssh` or git credential config. "Don't add my
GitHub key into GitHub" is respected by design.

### Activation

Cursor loads project hooks from the **workspace root** `.cursor/hooks.json`.
- If you open **YAX** directly as the workspace, this file activates the hook.
- If you open the parent folder (e.g. `Desktop`) as the workspace, a matching
  `.cursor/hooks.json` there points at this same script.

The script self-locates the repo via its own path, so it commits the YAX repo
regardless of which workspace launched it.

### Turn it off

Delete or rename `hooks.json` (or remove the `stop` entry). Cursor reloads on
save; restart Cursor if it does not pick up the change.
