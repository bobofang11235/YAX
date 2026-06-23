#!/usr/bin/env python3
"""YAX toolbox runtime.

YAX is a retrieval-first vLLM engineering harness. Markdown remains the source
of truth. This script builds a deterministic search index from tool, workflow,
and run cards; recommends relevant artifacts for a task; scaffolds new
artifacts; and validates references.

The design mirrors a compact agent harness: instructions route, knowledge
explains, the registry makes search cheap, and runs/handoffs keep durable state.
"""

import argparse
from collections import Counter, defaultdict
import datetime as _dt
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRS = ("tools", "workflows", "runs")
REGISTRY_PATH = ROOT / "registry" / "toolbox-index.json"
ABSTENTIONS_PATH = ROOT / "capture" / "abstentions.jsonl"
BASELINE_PATH = ROOT / "evals" / "baseline.json"
DEVMAP_DIR = ROOT / "devmap"

# YAX covers multiple inference engines symmetrically. Each engine has its own
# `<engine>-*` code-map files under devmap/ and its own generated registry index.
ENGINES = ("vllm", "sglang")


def engine_paths(engine):
    return {
        "areas": DEVMAP_DIR / ("%s-areas.jsonl" % engine),
        "versions": DEVMAP_DIR / ("%s-versions.json" % engine),
        "sync": DEVMAP_DIR / ("%s-sync-state.json" % engine),
        "codemap_index": ROOT / "registry" / ("%s-codemap-by-version.json" % engine),
    }


def available_engines():
    return [e for e in ENGINES if engine_paths(e)["areas"].exists()]
DRIFT_MARGIN_TOL = 3
KIND_DIR = {
    "tool": "tools/custom",
    "workflow": "workflows",
    "run": "runs",
}
TEMPLATE_FOR_KIND = {
    "tool": "tool-note.md",
    "workflow": "workflow-note.md",
    "run": "run-note.md",
}
REQUIRED = {
    "tool": ("id", "kind", "title", "status", "tags", "capabilities", "source"),
    "workflow": ("id", "kind", "title", "status", "tags", "tools", "triggers", "validation", "source"),
    "run": ("id", "kind", "title", "status", "task", "tools_considered", "tools_used", "workflow_update"),
}
TOKEN_RE = re.compile(r"[A-Za-z0-9_+.#/-]+")

STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "for", "with", "in", "on", "at",
    "by", "from", "as", "is", "are", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "into", "then", "than", "so", "do", "does",
    "did", "done", "before", "after", "over", "under", "your", "you", "my", "me",
    "we", "us", "our", "their", "them", "they", "but", "not", "no", "if", "when",
    "while", "which", "what", "how", "why", "where", "who", "will", "would",
    "can", "could", "should", "shall", "may", "might", "must", "one", "two",
    "up", "out", "about", "using", "use", "used", "per", "via", "also", "just",
    "only", "any", "some", "more", "most", "other", "there", "here", "i", "am",
    "was", "were", "has", "have", "had",
})

MIN_SCORE = 12


def rel(path):
    return path.relative_to(ROOT).as_posix()


def split_inline_list(value):
    inner = value.strip()[1:-1].strip()
    if not inner:
        return []
    parts = []
    current = []
    quote = None
    for char in inner:
        if char in ('"', "'"):
            quote = None if quote == char else char
            current.append(char)
        elif char == "," and quote is None:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return [parse_scalar(part) for part in parts if part]


def parse_scalar(value):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return split_inline_list(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none", "~"):
        return None
    return value


def parse_frontmatter(text):
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    block = text[4:end].strip("\n")
    body = text[end + len("\n---"):].lstrip("\n")
    data = {}
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        raw = raw.strip()
        if raw:
            data[key] = parse_scalar(raw)
            i += 1
            continue
        values = []
        i += 1
        while i < len(lines):
            child = lines[i]
            stripped = child.strip()
            if not stripped:
                i += 1
                continue
            if not child.startswith((" ", "\t")):
                break
            if stripped.startswith("- "):
                values.append(parse_scalar(stripped[2:]))
            i += 1
        data[key] = values
    return data, body


def parse_sections(body):
    sections = {}
    current = None
    buf = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(buf).strip()
            current = line[3:].strip()
            buf = []
        elif current:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    return sections


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 1]


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def load_artifact(path):
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    kind = meta.get("kind")
    if kind not in ("tool", "workflow", "run"):
        return None
    sections = parse_sections(body)
    title = meta.get("title") or body.splitlines()[0].lstrip("# ").strip() if body.splitlines() else meta.get("id", path.stem)
    artifact = {
        "id": str(meta.get("id", path.stem)),
        "kind": str(kind),
        "ref": "%s:%s" % (kind, meta.get("id", path.stem)),
        "title": str(title),
        "path": rel(path),
        "status": str(meta.get("status", "unknown")),
        "tags": as_list(meta.get("tags")),
        "aliases": as_list(meta.get("aliases")),
        "capabilities": as_list(meta.get("capabilities")),
        "inputs": as_list(meta.get("inputs")),
        "outputs": as_list(meta.get("outputs")),
        "tools": as_list(meta.get("tools")),
        "triggers": as_list(meta.get("triggers")),
        "validation": as_list(meta.get("validation")),
        "related": as_list(meta.get("related")),
        "source": as_list(meta.get("source")),
        "task": str(meta.get("task", "")),
        "workflow": str(meta.get("workflow", "")),
        "tools_considered": as_list(meta.get("tools_considered")),
        "tools_used": as_list(meta.get("tools_used")),
        "workflow_update": str(meta.get("workflow_update", "")),
        "sections": sections,
    }
    search_parts = [artifact["id"], artifact["title"], artifact["status"], artifact["task"], artifact["workflow"], artifact["workflow_update"]]
    for key in ("tags", "aliases", "capabilities", "inputs", "outputs", "tools", "triggers", "validation", "related", "source", "tools_considered", "tools_used"):
        search_parts.extend(artifact.get(key, []))
    search_parts.extend(sections.values())
    artifact["searchText"] = "\n".join(str(p) for p in search_parts if p)
    return artifact


def iter_markdown(root=ROOT):
    for dirname in ARTIFACT_DIRS:
        base = root / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.md")):
            if path.name == "README.md":
                continue
            yield path


def load_artifacts(root=ROOT):
    artifacts = []
    for path in iter_markdown(root):
        item = load_artifact(path)
        if item:
            artifacts.append(item)
    return artifacts


INDEX_SECTION_SYNOPSIS_CHARS = 240


def index_artifact(artifact):
    slim = {k: v for k, v in artifact.items() if k != "searchText"}
    sections = {}
    for name, body in artifact.get("sections", {}).items():
        synopsis = " ".join(str(body).split())
        if len(synopsis) > INDEX_SECTION_SYNOPSIS_CHARS:
            synopsis = synopsis[:INDEX_SECTION_SYNOPSIS_CHARS].rstrip() + "..."
        sections[name] = synopsis
    slim["sections"] = sections
    return slim


def build_index(root=ROOT):
    artifacts = [index_artifact(a) for a in load_artifacts(root)]
    by_kind = {"tool": [], "workflow": [], "run": []}
    for item in artifacts:
        by_kind[item["kind"]].append(item)
    for values in by_kind.values():
        values.sort(key=lambda a: a["id"])
    return {
        "version": "1.0.0",
        "source": "markdown",
        "counts": {k: len(v) for k, v in by_kind.items()},
        "artifacts": sorted(artifacts, key=lambda a: (a["kind"], a["id"])),
    }


def write_index(args):
    index = build_index(ROOT)
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Wrote %s" % rel(REGISTRY_PATH))
    print("Tools: {tool}, workflows: {workflow}, runs: {run}".format(**index["counts"]))
    if available_engines():
        write_codemap_index()
    return 0


def artifact_score(query, artifact):
    q = query.lower().strip()
    tokens = set(t for t in tokenize(query) if t not in STOPWORDS)
    if not tokens:
        return 0
    score = 0
    fields = {
        "id": artifact.get("id", ""),
        "title": artifact.get("title", ""),
        "tags": " ".join(artifact.get("tags", [])),
        "aliases": " ".join(artifact.get("aliases", [])),
        "capabilities": " ".join(artifact.get("capabilities", [])),
        "triggers": " ".join(artifact.get("triggers", [])),
        "task": artifact.get("task", ""),
        "sections": " ".join(artifact.get("sections", {}).values()),
        "related": " ".join(artifact.get("related", []) + artifact.get("tools", []) + artifact.get("tools_used", [])),
    }
    if q and q in artifact.get("searchText", "").lower():
        score += 10
    weights = {
        "id": 5,
        "title": 5,
        "tags": 6,
        "aliases": 6,
        "capabilities": 4,
        "triggers": 4,
        "task": 4,
        "sections": 1,
        "related": 2,
    }
    for token in tokens:
        for field, value in fields.items():
            if token in value.lower():
                score += weights[field]
    if artifact.get("kind") == "workflow":
        score += 1
    return score


def ranked(query, artifacts=None, limit=10, kinds=None):
    artifacts = artifacts if artifacts is not None else load_artifacts(ROOT)
    results = []
    for item in artifacts:
        if kinds and item["kind"] not in kinds:
            continue
        score = artifact_score(query, item)
        if score > 0:
            results.append((score, item))
    results.sort(key=lambda pair: (-pair[0], pair[1]["kind"], pair[1]["id"]))
    return results[:limit]


def print_result(score, item):
    print("[%s] %s  score=%s" % (item["ref"], item["title"], score))
    print("  path: %s" % item["path"])
    if item.get("tags"):
        print("  tags: %s" % ", ".join(item["tags"]))
    use_when = item.get("sections", {}).get("Use When", "")
    if use_when:
        first = " ".join(use_when.split())[:240]
        print("  use: %s" % first)


def cmd_search(args):
    results = ranked(args.query, limit=args.limit, kinds=set(args.kind) if args.kind else None)
    if not results:
        print("No matching tools, workflows, or runs found.")
        return 1
    for score, item in results:
        print_result(score, item)
    return 0


def log_abstention(query):
    try:
        rec = {"ts": _dt.datetime.now().isoformat(timespec="seconds"), "query": query}
        ABSTENTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ABSTENTIONS_PATH.open("a", encoding="utf-8") as fh:
            print(json.dumps(rec), file=fh)
    except OSError:
        pass


def cmd_gaps(args):
    if not ABSTENTIONS_PATH.exists():
        print("No abstentions logged yet.")
        return 0
    counts = Counter()
    for line in ABSTENTIONS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            counts[json.loads(line)["query"]] += 1
        except (ValueError, KeyError):
            continue
    if not counts:
        print("No abstentions logged yet.")
        return 0
    print("# Coverage gaps (abstained recommend queries)")
    for query, n in counts.most_common(args.limit):
        print("%3d  %s" % (n, query))
    return 0


def cmd_recommend(args):
    artifacts = load_artifacts(ROOT)
    floor = getattr(args, "min_score", MIN_SCORE)
    workflows = [p for p in ranked(args.query, artifacts, limit=5, kinds={"workflow"}) if p[0] >= floor]
    tools = [p for p in ranked(args.query, artifacts, limit=8, kinds={"tool"}) if p[0] >= floor]
    runs = [p for p in ranked(args.query, artifacts, limit=5, kinds={"run"}) if p[0] >= floor]
    print("# YAX Recommendation")
    print("")
    print("Query: %s" % args.query)
    print("")
    if workflows:
        print("## Workflows")
        for score, item in workflows:
            print("- `%s` (%s): %s" % (item["ref"], item["path"], item["title"]))
        print("")
    if tools:
        print("## Tools")
        for score, item in tools:
            print("- `%s` (%s): %s" % (item["ref"], item["path"], item["title"]))
        print("")
    if runs:
        print("## Similar Runs")
        for score, item in runs:
            print("- `%s` (%s): %s" % (item["ref"], item["path"], item["title"]))
        print("")
    if not (workflows or tools or runs):
        log_abstention(args.query)
        print("No reusable workflow or tool matched. Consider creating a run record after completing the task.")
        return 1
    print("## Suggested Next Step")
    if workflows:
        top = workflows[0][1]
        print("Read `%s`, then load only its referenced tools." % top["path"])
    elif tools:
        print("Read the top tool cards and compose a short task plan.")
    else:
        print("Read the similar run record, then decide whether to create a workflow.")
    return 0


def normalize_ref(ref, default_kind=None):
    if not ref:
        return ""
    if ":" in ref:
        return ref
    return "%s:%s" % (default_kind or "tool", ref)


def validate_artifacts(root=ROOT):
    artifacts = load_artifacts(root)
    errors = []
    warnings = []
    refs = {}
    by_kind = {"tool": set(), "workflow": set(), "run": set()}
    for item in artifacts:
        ref = item["ref"]
        if ref in refs:
            errors.append("Duplicate artifact ref `%s`: %s and %s" % (ref, refs[ref], item["path"]))
        refs[ref] = item["path"]
        by_kind[item["kind"]].add(ref)
        for key in REQUIRED[item["kind"]]:
            value = item.get(key)
            if value is None or value == "" or value == []:
                errors.append("%s missing required `%s`" % (item["path"], key))
    known = set(refs.keys())
    for item in artifacts:
        for refname in item.get("tools", []):
            ref = normalize_ref(refname, "tool")
            if ref not in by_kind["tool"]:
                errors.append("%s references unknown tool `%s`" % (item["path"], refname))
        for refname in item.get("tools_considered", []) + item.get("tools_used", []):
            ref = normalize_ref(refname, "tool")
            if ref not in by_kind["tool"]:
                errors.append("%s references unknown run tool `%s`" % (item["path"], refname))
        workflow = item.get("workflow")
        if item["kind"] == "run" and workflow:
            ref = normalize_ref(workflow, "workflow")
            if ref not in by_kind["workflow"]:
                errors.append("%s references unknown workflow `%s`" % (item["path"], workflow))
        update = item.get("workflow_update")
        if item["kind"] == "run" and update and update not in ("none", "no", "n/a"):
            ref = normalize_ref(update, "workflow")
            if ref not in known:
                errors.append("%s has unknown workflow_update `%s`" % (item["path"], update))
        for refname in item.get("related", []):
            if ":" in refname and refname.split(":", 1)[0] in ("tool", "workflow", "run") and refname not in known:
                warnings.append("%s related ref not found `%s`" % (item["path"], refname))
        if not item.get("source") and item["kind"] != "run":
            warnings.append("%s has no source" % item["path"])
    if REGISTRY_PATH.exists() and root == ROOT:
        try:
            current = build_index(root)
            stored = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            if stored.get("artifacts") != current.get("artifacts") or stored.get("counts") != current.get("counts"):
                warnings.append("registry/toolbox-index.json is stale; run `python3 scripts/yax.py index`")
        except Exception as exc:
            warnings.append("could not compare registry: %s" % exc)
    elif root == ROOT:
        warnings.append("registry/toolbox-index.json does not exist; run `python3 scripts/yax.py index`")
    if root == ROOT:
        for engine in available_engines():
            idx_path = engine_paths(engine)["codemap_index"]
            try:
                current_cm = build_codemap_index(engine)
                if not idx_path.exists():
                    warnings.append("%s does not exist; run `python3 scripts/yax.py index`" % rel(idx_path))
                else:
                    stored_cm = json.loads(idx_path.read_text(encoding="utf-8"))
                    if stored_cm.get("by_version") != current_cm.get("by_version"):
                        warnings.append("%s is stale; run `python3 scripts/yax.py index`" % rel(idx_path))
            except Exception as exc:
                warnings.append("could not compare %s codemap index: %s" % (engine, exc))
    return errors, warnings


def cmd_validate(args):
    errors, warnings = validate_artifacts(ROOT)
    for warning in warnings:
        print("WARNING: %s" % warning)
    for error in errors:
        print("ERROR: %s" % error)
    if errors:
        print("Validation failed: %s errors, %s warnings" % (len(errors), len(warnings)))
        return 1
    print("Validation passed: %s warnings" % len(warnings))
    return 0


def slugify(value):
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-") or "untitled"


def scaffold(kind, artifact_id):
    artifact_id = slugify(artifact_id)
    template = ROOT / "templates" / TEMPLATE_FOR_KIND[kind]
    if not template.exists():
        raise SystemExit("Missing template %s" % rel(template))
    if kind == "run":
        today = _dt.date.today().isoformat()
        out = ROOT / KIND_DIR[kind] / (today + "-" + artifact_id + ".md")
        run_id = today + "-" + artifact_id
    else:
        out = ROOT / KIND_DIR[kind] / (artifact_id + ".md")
        run_id = artifact_id
    if out.exists():
        raise SystemExit("Refusing to overwrite existing file %s" % rel(out))
    text = template.read_text(encoding="utf-8")
    title = artifact_id.replace("-", " ").replace("_", " ").title()
    text = text.replace("<tool-id>", artifact_id)
    text = text.replace("<workflow-id>", artifact_id)
    text = text.replace("<run-id>", run_id)
    text = text.replace("<Tool Title>", title)
    text = text.replace("<Workflow Title>", title)
    text = text.replace("<Run Title>", title)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("Created %s" % rel(out))
    return 0


def cmd_new_tool(args):
    return scaffold("tool", args.id)


def cmd_new_workflow(args):
    return scaffold("workflow", args.id)


def cmd_new_run(args):
    return scaffold("run", args.slug)


def scaffold_handoff(slug):
    slug = slugify(slug)
    template = ROOT / "templates" / "handoff-note.md"
    if not template.exists():
        raise SystemExit("Missing template %s" % rel(template))
    today = _dt.date.today().isoformat()
    out = ROOT / "runs" / (today + "-" + slug + "-handoff.md")
    if out.exists():
        raise SystemExit("Refusing to overwrite existing file %s" % rel(out))
    text = template.read_text(encoding="utf-8")
    title = slug.replace("-", " ").replace("_", " ").title()
    text = text.replace("<task title>", title)
    text = text.replace("runs/<slug>.md", "runs/" + today + "-" + slug + ".md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def cmd_new_handoff(args):
    out = scaffold_handoff(args.slug)
    print("Created %s" % rel(out))
    print("Handoff notes are not registry artifacts; no index regeneration needed.")
    return 0


# ---------------------------------------------------------------------------
# Evaluation: retrieval / routing quality of the recommender
# ---------------------------------------------------------------------------

RECOMMEND_LIMITS = {"workflow": 5, "tool": 8, "run": 5}


def ref_kind(ref):
    return ref.split(":", 1)[0] if ":" in ref else "tool"


def rank_of_ref(query, artifacts, ref):
    kind = ref_kind(ref)
    results = ranked(query, artifacts, limit=len(artifacts) + 1, kinds={kind})
    rank = None
    score = 0
    for idx, (sc, item) in enumerate(results, start=1):
        if item["ref"] == ref:
            rank, score = idx, sc
            break
    best_other = max([sc for sc, item in results if item["ref"] != ref] or [0])
    return rank, score, best_other


def eval_self_consistency(artifacts, top_k=None):
    checks = []
    for art in artifacts:
        kind = art["kind"]
        if kind not in ("tool", "workflow"):
            continue
        limit = top_k or RECOMMEND_LIMITS.get(kind, 5)
        terms = ([("trigger", t) for t in art.get("triggers", [])]
                 + [("alias", a) for a in art.get("aliases", [])])
        for term_kind, term in terms:
            if not term or not tokenize(term):
                continue
            rank, score, best_other = rank_of_ref(term, artifacts, art["ref"])
            checks.append({
                "ref": art["ref"], "term": term, "term_kind": term_kind,
                "rank": rank, "limit": limit, "score": score,
                "best_other": best_other,
                "passed": rank is not None and rank <= limit,
            })
    return checks


def load_eval_cases(evals_dir):
    cases = []
    base = Path(evals_dir)
    if not base.exists():
        return cases
    for path in sorted(base.glob("*.jsonl")):
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except ValueError as exc:
                raise SystemExit("Bad eval case %s line %d: %s" % (path.name, lineno, exc))
            obj.setdefault("id", "%s:%d" % (path.stem, lineno))
            cases.append(obj)
    return cases


def eval_golden(cases, artifacts, min_score=MIN_SCORE):
    results = []
    for case in cases:
        query = case.get("query", "")
        rec = {"id": case.get("id"), "query": query, "passed": True, "detail": "",
               "expect": case.get("expect"), "forbid": case.get("forbid", []),
               "expect_none": case.get("expect_none", False),
               "cluster": case.get("cluster", "unclustered")}
        if case.get("expect_none"):
            top = ranked(query, artifacts, limit=1)
            top_score = top[0][0] if top else 0
            top_ref = top[0][1]["ref"] if top else "-"
            rec["score"] = top_score
            if top_score >= min_score:
                rec["passed"] = False
                rec["detail"] = ("off-domain scored %d >= min_score %d (top=%s)"
                                 % (top_score, min_score, top_ref))
        if case.get("expect"):
            ref = case["expect"]
            kind = ref_kind(ref)
            max_rank = case.get("max_rank", RECOMMEND_LIMITS.get(kind, 5))
            rank, score, best_other = rank_of_ref(query, artifacts, ref)
            rec.update({"kind": kind, "rank": rank, "max_rank": max_rank,
                        "score": score, "best_other": best_other,
                        "rr": (1.0 / rank) if rank else 0.0,
                        "top1": rank == 1, "margin": score - best_other})
            if rank is None or rank > max_rank:
                rec["passed"] = False
                rec["detail"] = "expected %s rank=%s (max %s)" % (ref, rank, max_rank)
        forbid_rank = case.get("forbid_rank", 1)
        violations = []
        for fref in rec["forbid"]:
            frank, _, _ = rank_of_ref(query, artifacts, fref)
            if frank is not None and frank <= forbid_rank:
                violations.append("%s@%d" % (fref, frank))
        if violations:
            rec["passed"] = False
            extra = "forbidden in top-%d: %s" % (forbid_rank, ", ".join(violations))
            rec["detail"] = (rec["detail"] + "; " + extra) if rec["detail"] else extra
        results.append(rec)
    return results


def golden_snapshot(results):
    snap = {}
    for r in results:
        if not r.get("expect"):
            continue
        snap[r["id"]] = {
            "rank": r.get("rank"),
            "score": r.get("score", 0),
            "margin": r.get("margin", 0),
            "rr": round(r.get("rr", 0.0), 4),
        }
    return snap


def compare_drift(baseline, current, margin_tol=DRIFT_MARGIN_TOL):
    drifts = []
    for cid, cur in current.items():
        base = baseline.get(cid)
        if not base:
            continue
        base_rank, cur_rank = base.get("rank"), cur.get("rank")
        margin_delta = cur.get("margin", 0) - base.get("margin", 0)
        rank_worse = base_rank is not None and (cur_rank is None or cur_rank > base_rank)
        if margin_delta <= -margin_tol or rank_worse:
            drifts.append({
                "id": cid,
                "base_rank": base_rank, "cur_rank": cur_rank,
                "base_margin": base.get("margin", 0), "cur_margin": cur.get("margin", 0),
                "margin_delta": margin_delta,
            })
    return drifts


def cmd_eval(args):
    artifacts = load_artifacts(ROOT)
    hard_fail = 0
    print("# YAX Eval")
    print("")

    if not args.golden_only:
        checks = eval_self_consistency(artifacts, top_k=args.self_topk)
        total = len(checks)
        passed = sum(1 for c in checks if c["passed"])
        top1 = sum(1 for c in checks if c["rank"] == 1)
        print("## Self-consistency")
        print("Own triggers/aliases retrieving their card within kind limit: %d/%d"
              % (passed, total))
        if total:
            print("Own term ranks #1: %d/%d (%.0f%%)"
                  % (top1, total, 100.0 * top1 / total))
        for c in sorted(checks, key=lambda chk: chk["ref"]):
            if not c["passed"]:
                where = ("rank=%d" % c["rank"]) if c["rank"] else "absent"
                print("  FAIL %-38s %-7s term=%r %s"
                      % (c["ref"], c["term_kind"], c["term"], where))
        if not args.warn_only:
            hard_fail += total - passed
        print("")

    if not args.self_only:
        cases = load_eval_cases(args.evals)
        results = eval_golden(cases, artifacts, min_score=args.min_score)
        positives = [r for r in results if r.get("expect")]
        antis = [r for r in results if r.get("forbid")]
        negatives = [r for r in results if r.get("expect_none")]
        hits = sum(1 for r in positives if r.get("rank") and r["rank"] <= r["max_rank"])
        top1 = sum(1 for r in positives if r.get("top1"))
        mrr = (sum(r.get("rr", 0.0) for r in positives) / len(positives)) if positives else 0.0
        print("## Golden routing (%s)" % args.evals)
        print("Cases: %d positive, %d negative, %d with anti-targets"
              % (len(positives), len(negatives), len(antis)))
        if positives:
            denom = float(len(positives))
            print("Recall@k: %d/%d (%.3f)" % (hits, len(positives), hits / denom))
            print("Top-1:    %d/%d (%.3f)" % (top1, len(positives), top1 / denom))
            print("MRR:      %.3f" % mrr)
        if negatives:
            neg_ok = sum(1 for r in negatives if r["passed"])
            print("Negatives (abstain < %d): %d/%d" % (args.min_score, neg_ok, len(negatives)))
        for r in results:
            if not r["passed"]:
                print("  FAIL %-20s query=%r -> %s" % (r["id"], r["query"], r["detail"]))
        if not args.warn_only:
            hard_fail += sum(1 for r in results if not r["passed"])
        print("")

        clusters = defaultdict(list)
        for r in positives:
            clusters[r.get("cluster", "unclustered")].append(r)
        if len(clusters) > 1:
            print("## Per-cluster routing")
            for cname in sorted(clusters):
                crs = clusters[cname]
                chits = sum(1 for r in crs if r.get("rank") and r["rank"] <= r["max_rank"])
                ctop1 = sum(1 for r in crs if r.get("top1"))
                cmrr = sum(r.get("rr", 0.0) for r in crs) / len(crs)
                print("  %-14s recall %d/%-3d top1 %d/%-3d mrr %.3f"
                      % (cname, chits, len(crs), ctop1, len(crs), cmrr))
            print("")

        current = golden_snapshot(results)
        if args.save_baseline:
            BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            BASELINE_PATH.write_text(
                json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print("## Drift")
            print("Saved golden baseline: %s (%d cases)" % (rel(BASELINE_PATH), len(current)))
            print("")
        elif not args.no_baseline and BASELINE_PATH.exists():
            try:
                baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            except ValueError:
                baseline = {}
            drifts = compare_drift(baseline, current, margin_tol=args.drift_margin)
            print("## Drift vs %s (margin tol %d)" % (rel(BASELINE_PATH), args.drift_margin))
            if drifts:
                for d in sorted(drifts, key=lambda x: x["margin_delta"]):
                    print("  DRIFT %-20s rank %s->%s  margin %s->%s (%+d)"
                          % (d["id"], d["base_rank"], d["cur_rank"],
                             d["base_margin"], d["cur_margin"], d["margin_delta"]))
                if args.drift_fail and not args.warn_only:
                    hard_fail += len(drifts)
            else:
                print("No sub-threshold drift across %d shared cases." % len(current))
            print("")

    print("## Result")
    if hard_fail and not args.warn_only:
        print("FAIL: %d hard failure(s)" % hard_fail)
        return 1
    print("PASS")
    return 0


def cmd_mismatches(args):
    artifacts = load_artifacts(ROOT)
    runs = [a for a in artifacts if a["kind"] == "run"]
    candidates = []
    for run in runs:
        task = run.get("task") or run.get("title") or ""
        if not task or not tokenize(task):
            continue
        seen = set()
        for refname in run.get("tools_used", []):
            ref = normalize_ref(refname, "tool")
            if ref_kind(ref) != "tool" or ref in seen:
                continue
            seen.add(ref)
            rank, score, _ = rank_of_ref(task, artifacts, ref)
            if rank is None or rank > args.max_rank:
                candidates.append({"run": run["id"], "task": task, "ref": ref,
                                   "rank": rank, "score": score})
    print("# YAX Recommend Mismatches")
    print("")
    if not candidates:
        print("No mismatches: every run's used tools rank within top-%d for its task."
              % args.max_rank)
        return 0
    print("Runs whose used tool is not within top-%d for the run task:" % args.max_rank)
    for c in candidates:
        where = ("rank=%s" % c["rank"]) if c["rank"] else "absent"
        print("  %-44s used=%-34s (%s, score=%s)"
              % (c["run"], c["ref"], where, c["score"]))
    print("")
    print("## Candidate golden cases (review before adding to evals/recommend.jsonl)")
    for c in candidates:
        obj = {"id": "mined-%s" % c["ref"].split(":", 1)[1],
               "query": c["task"], "expect": c["ref"],
               "max_rank": args.max_rank, "src": "runs/%s.md" % c["run"]}
        print(json.dumps(obj))
    return 0


# ---------------------------------------------------------------------------
# Code map: version-tagged "where to look / what to edit" for vLLM development
# ---------------------------------------------------------------------------

# A very large tuple stands in for "latest" / unbounded so range checks are simple.
_VERSION_INF = (10 ** 6,)


def parse_version(tag):
    """Parse a version tag into a comparable tuple.

    'latest', 'main', 'head', '' -> +infinity (newest layout).
    '0.8.5' -> (0, 8, 5). Non-numeric suffixes (rc, post, +cu) are dropped.
    """
    if tag is None:
        return _VERSION_INF
    tag = str(tag).strip().lower().lstrip("v")
    if tag in ("", "latest", "main", "head", "nightly"):
        return _VERSION_INF
    parts = []
    for chunk in tag.split("."):
        m = re.match(r"\d+", chunk)
        parts.append(int(m.group()) if m else 0)
    return tuple(parts) or (0,)


def load_versions(engine="vllm"):
    path = engine_paths(engine)["versions"]
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def load_codemap(engine="vllm"):
    """Load code-map areas for an engine (skip blank/# lines)."""
    areas = []
    path = engine_paths(engine)["areas"]
    if not path.exists():
        return areas
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            areas.append(json.loads(line))
        except ValueError as exc:
            raise SystemExit("Bad %s code-map line %d: %s" % (engine, lineno, exc))
    return areas


def resolve_layout(area, version_tuple):
    """Pick the first layout whose [since, until] range contains version_tuple."""
    layouts = area.get("layouts", [])
    for layout in layouts:
        since = parse_version(layout.get("since", "0.0.0"))
        until = layout.get("until")
        until_t = parse_version(until) if until else _VERSION_INF
        if since <= version_tuple <= until_t:
            return layout
    return layouts[0] if layouts else None


def codemap_score(query, area):
    tokens = set(t for t in tokenize(query) if t not in STOPWORDS)
    if not tokens:
        return 0
    q = query.lower().strip()
    symptoms = " ".join(area.get("symptoms", [])).lower()
    area_name = area.get("area", "").lower()
    area_id = area.get("id", "").lower()
    blob = " ".join(layout_text(l) for l in area.get("layouts", [])).lower()
    score = 0
    if q and (q in symptoms or q in area_name):
        score += 8
    for token in tokens:
        if token in area_id:
            score += 6
        if token in symptoms:
            score += 5
        if token in area_name:
            score += 4
        if token in blob:
            score += 1
    return score


def layout_text(layout):
    parts = list(layout.get("folders", [])) + list(layout.get("files", [])) \
        + list(layout.get("entry", [])) + [layout.get("note", "")]
    return " ".join(str(p) for p in parts)


def print_area(area, version_tuple, version_tag):
    layout = resolve_layout(area, version_tuple)
    print("## %s" % area.get("area", area.get("id")))
    if not layout:
        print("  (no layout recorded)")
        return
    if layout.get("folders"):
        print("  check folders:")
        for f in layout["folders"]:
            print("    - %s" % f)
    if layout.get("files"):
        print("  edit files:")
        for f in layout["files"]:
            print("    - %s" % f)
    if layout.get("entry"):
        print("  entry points: %s" % ", ".join(layout["entry"]))
    if layout.get("note"):
        print("  note: %s" % layout["note"])
    print("")


def cmd_where(args):
    engine = getattr(args, "engine", "vllm")
    areas = load_codemap(engine)
    label = engine.upper()
    if not areas:
        print("No code map for engine '%s' (%s missing)."
              % (engine, rel(engine_paths(engine)["areas"])))
        return 1
    versions = load_versions(engine)
    tag = args.version or versions.get("default_tag", "latest")
    vt = parse_version(tag)
    if args.list_areas:
        print("# %s code-map areas" % label)
        for a in sorted(areas, key=lambda x: x["id"]):
            print("- %-20s %s" % (a["id"], a.get("area", "")))
        return 0
    scored = sorted(((codemap_score(args.query, a), a) for a in areas),
                    key=lambda p: (-p[0], p[1]["id"]))
    hits = [a for s, a in scored if s > 0][:args.limit]
    print("# Where to look in %s" % label)
    print("")
    print("Query: %s" % args.query)
    print("Engine: %s   Version: %s" % (engine, tag))
    print("")
    if not hits:
        print("No matching area. Try `--list-areas` or rephrase with code terms")
        print("(scheduler, kv cache, attention backend, quantization, model, ...).")
        return 1
    for area in hits:
        print_area(area, vt, tag)
    print("Confirm exact paths against the checkout: file layout is "
          "version-sensitive and may have moved.")
    return 0


def build_codemap_index(engine="vllm"):
    """Generate a per-version resolved view of an engine's code map.

    For each representative version tag, emit the resolved folders/files/entry
    for every area, so a developer on a given engine version gets the right
    paths.
    """
    areas = load_codemap(engine)
    versions = load_versions(engine)
    tags = versions.get("representative_tags", ["latest"])
    out = {"version": "1.0.0", "engine": engine,
           "source": engine_paths(engine)["areas"].name, "tags": tags,
           "by_version": {}}
    for tag in tags:
        vt = parse_version(tag)
        resolved = []
        for area in sorted(areas, key=lambda a: a["id"]):
            layout = resolve_layout(area, vt) or {}
            resolved.append({
                "id": area["id"],
                "area": area.get("area", ""),
                "folders": layout.get("folders", []),
                "files": layout.get("files", []),
                "entry": layout.get("entry", []),
                "note": layout.get("note", ""),
            })
        out["by_version"][tag] = resolved
    return out


def write_codemap_index():
    for engine in available_engines():
        index = build_codemap_index(engine)
        path = engine_paths(engine)["codemap_index"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Wrote %s (%s: %d areas, tags: %s)"
              % (rel(path), engine, len(load_codemap(engine)), ", ".join(index["tags"])))


def load_sync_state(engine="vllm"):
    path = engine_paths(engine)["sync"]
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def cmd_sync_status(args):
    """Show the upstream point YAX is synced to (per engine) and how to find newer commits."""
    engine = getattr(args, "engine", "vllm")
    state = load_sync_state(engine)
    if not state:
        print("No sync state for engine '%s' (%s missing)."
              % (engine, rel(engine_paths(engine)["sync"])))
        return 1
    ref = state.get("synced_to", "<unknown>")
    repo = state.get("repo") or state.get("vllm_repo") or ""
    branch = state.get("default_branch", "main")
    print("# YAX upstream sync status (%s)" % engine)
    print("")
    print("Repo:          %s" % repo)
    print("Synced to:     %s  (release %s)"
          % (ref, state.get("synced_to_release_date", "?")))
    print("Synced on:     %s" % state.get("synced_on", "?"))
    print("Engine era:    %s" % state.get("engine_era", "?"))
    print("")
    print("To review only what changed since the last sync:")
    print("  git -C <clone> fetch --tags")
    print("  git -C <clone> log --oneline %s..origin/%s" % (ref, branch))
    print("")
    print("After updating YAX, bump `synced_to`/`synced_on` in")
    print("%s and add a CHANGELOG.md entry." % rel(engine_paths(engine)["sync"]))
    repo_path = getattr(args, "repo_path", None)
    if repo_path:
        import subprocess
        p = Path(repo_path)
        if not (p / ".git").exists():
            print("")
            print("WARNING: %s is not a git checkout; skipping diff." % repo_path)
            return 0
        print("")
        print("## New commits in %s since %s" % (repo_path, ref))
        try:
            subprocess.run(["git", "-C", str(p), "fetch", "--tags", "--quiet"], check=False)
            res = subprocess.run(
                ["git", "-C", str(p), "log", "--oneline", "%s..origin/%s" % (ref, branch)],
                capture_output=True, text=True, check=False)
            out = (res.stdout or "").strip()
            if res.returncode != 0:
                print((res.stderr or "git log failed").strip())
                print("(Is %s a valid tag/ref in that clone?)" % ref)
                return 0
            print(out if out else "(none — YAX is up to date with origin/%s)" % branch)
        except FileNotFoundError:
            print("git not found on PATH; run the command above manually.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="YAX vLLM engineering toolbox runtime")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("index", help="build registry/toolbox-index.json + codemap index")
    p.set_defaults(func=write_index)

    p = sub.add_parser("where",
                       help="version-aware code map: which folders/files to check/edit for a problem")
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--engine", "-e", choices=ENGINES, default="vllm",
                   help="which engine's code map to use (default: vllm)")
    p.add_argument("--version", "-V", default=None,
                   help="engine version tag, e.g. 0.8.5 or latest (default: latest)")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--list-areas", action="store_true", help="list all code-map areas and exit")
    p.set_defaults(func=cmd_where)

    p = sub.add_parser("codemap-index", help="generate registry/<engine>-codemap-by-version.json")
    p.set_defaults(func=lambda args: (write_codemap_index(), 0)[1])

    p = sub.add_parser("sync-status",
                       help="show the engine version YAX is synced to + how to find newer commits")
    p.add_argument("--engine", "-e", choices=ENGINES, default="vllm",
                   help="which engine's sync state to show (default: vllm)")
    p.add_argument("--repo-path", "--vllm-path", dest="repo_path", default=None,
                   help="path to a local engine checkout; if given, runs the commit diff")
    p.set_defaults(func=cmd_sync_status)

    p = sub.add_parser("search", help="search tools, workflows, and runs")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--kind", action="append", choices=("tool", "workflow", "run"))
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("recommend", help="recommend workflows and tools for a task")
    p.add_argument("query")
    p.add_argument("--min-score", type=int, default=MIN_SCORE,
                   help="floor below which recommend abstains (no confident match)")
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("new-tool", help="scaffold a new tool card")
    p.add_argument("id")
    p.set_defaults(func=cmd_new_tool)

    p = sub.add_parser("new-workflow", help="scaffold a new workflow card")
    p.add_argument("id")
    p.set_defaults(func=cmd_new_workflow)

    p = sub.add_parser("new-run", help="scaffold a new run record")
    p.add_argument("slug")
    p.set_defaults(func=cmd_new_run)

    p = sub.add_parser("new-handoff", help="scaffold a handoff note in runs/")
    p.add_argument("slug")
    p.set_defaults(func=cmd_new_handoff)

    p = sub.add_parser("validate", help="validate artifact metadata and references")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("eval", help="evaluate retrieval/routing quality of recommend")
    p.add_argument("--evals", default=str(ROOT / "evals"),
                   help="directory of *.jsonl golden routing cases")
    p.add_argument("--self-topk", type=int, default=None,
                   help="override the kind recommend limit for self-consistency")
    p.add_argument("--self-only", action="store_true", help="run only self-consistency")
    p.add_argument("--golden-only", action="store_true", help="run only golden cases")
    p.add_argument("--warn-only", action="store_true", help="report but always exit 0")
    p.add_argument("--min-score", type=int, default=MIN_SCORE,
                   help="confident-match floor used for expect_none negative cases")
    p.add_argument("--save-baseline", action="store_true",
                   help="write current golden metrics to evals/baseline.json as the accepted state")
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the drift comparison even if a baseline exists")
    p.add_argument("--drift-fail", action="store_true",
                   help="treat sub-threshold drift as a hard failure")
    p.add_argument("--drift-margin", type=int, default=DRIFT_MARGIN_TOL,
                   help="margin drop on a still-passing case that counts as drift")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("gaps", help="show queries where recommend abstained (coverage gaps)")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_gaps)

    p = sub.add_parser("mismatches",
                       help="mine runs/ for used-but-not-recommended tools (candidate golden cases)")
    p.add_argument("--max-rank", type=int, default=3,
                   help="a used tool ranked worse than this for its run task is a mismatch")
    p.set_defaults(func=cmd_mismatches)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
