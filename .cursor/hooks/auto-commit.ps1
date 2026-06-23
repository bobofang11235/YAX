# Auto-commit and push the YAX repo at the end of an agent turn.
#
# Safety:
# - Fails OPEN: any problem -> exit 0, never blocks the agent.
# - Uses ONLY your existing git credentials (cached HTTPS). It never creates,
#   reads, adds, or uploads SSH/GitHub keys, never runs `gh auth`, and never
#   edits ~/.ssh or git credential config.
# - No empty commits: it no-ops when the working tree is clean.

$ErrorActionPreference = 'SilentlyContinue'

# Cursor's hook process may not have git on PATH; add it if needed.
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  foreach ($p in @('C:\Program Files\Git\cmd',
                   'C:\Program Files (x86)\Git\cmd',
                   "$env:LOCALAPPDATA\Programs\Git\cmd")) {
    if (Test-Path (Join-Path $p 'git.exe')) { $env:Path = "$p;$env:Path"; break }
  }
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { exit 0 }

# Repo root = two levels up from this script (.cursor/hooks -> repo root).
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repo

git rev-parse --is-inside-work-tree *> $null
if ($LASTEXITCODE -ne 0) { exit 0 }

$changes = git status --porcelain
if ([string]::IsNullOrWhiteSpace($changes)) { exit 0 }

git add -A *> $null

# Build a descriptive message from the staged files: top-level areas + count.
$files = @(git diff --cached --name-only)
if ($files.Count -eq 0) { exit 0 }
$areas = ($files | ForEach-Object { ($_ -split '/')[0] } | Sort-Object -Unique) -join ', '
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm'
$msg = "auto: update $areas ($($files.Count) files) [$stamp]"
git commit -m $msg *> $null

# Push current branch to its upstream. Fail open on any error (offline, etc.).
git push *> $null

exit 0
