"""Normalize harness traces into one dict and embed it in a self-contained HTML page."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DEV_MARKERS = ("calib", "quick")
ARMS = ("A", "B", "C", "D")
PLATFORM_BY_ARM = {"A": "amd64", "B": "armv7", "C": "armv7", "D": "amd64"}
MEMORY_HEADER = "Memory from earlier sessions:\n"
PLACEHOLDER = "/*__DATA__*/null"
INTERN_MIN_CHARS = 256  # strings at least this long go into the shared table
TEXTS_KEY, DATA_KEY, REF_KEY = "$texts", "$data", "$t"
_STEP_FILE = re.compile(r"^(\d+)-([A-Za-z_]+)\.json$")
_REPEAT_DIR = re.compile(r"^r(\d+)$")

Step = dict[str, Any]


def _read_steps(directory: Path) -> list[Step]:
    """Read `NN-kind.json` files of one directory in step order; bad files are skipped."""
    steps: list[Step] = []
    for path in directory.glob("*.json"):
        match = _STEP_FILE.match(path.name)
        if not match:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skipping unreadable trace %s: %s", path, exc)
            continue
        if not isinstance(data, dict):
            logger.warning("skipping %s: step files must hold a JSON object", path)
            continue
        data["_file"] = path.name
        data.setdefault("step", int(match.group(1)))
        data.setdefault("kind", match.group(2))
        steps.append(data)
    return sorted(steps, key=lambda s: int(s["step"]))


def _result(step: Step) -> dict[str, Any]:
    res = step.get("result") or {}
    return {
        "stage": res.get("stage", "unknown"),
        "compiled": bool(res.get("compiled", False)),
        "correct": bool(res.get("correct", False)),
        "time_ms": res.get("time_ms"),
        "speedup": res.get("speedup"),
        "stderr": res.get("stderr", ""),
    }


def _llm_meta(step: Step | None) -> dict[str, Any] | None:
    if step is None:
        return None
    return {
        "prompt": step.get("prompt"),
        "text": step.get("text"),
        "cost_usd": step.get("cost_usd", 0.0),
        "duration_s": step.get("duration_s", 0.0),
        "file": step["_file"],
    }


def _arm_a(directory: Path) -> dict[str, Any] | None:
    """Arm A: attempts (generation + measured result + critique) and the stored memory."""
    if not directory.is_dir():
        return None
    steps = _read_steps(directory)
    if not steps:  # an aborted repeat leaves an empty directory behind
        return None
    sandbox = {s["round_no"]: s for s in steps if s["kind"] == "sandbox" and "round_no" in s}
    gens: dict[int, Step] = {}
    feedbacks: dict[int, Step] = {}
    for s in steps:
        if s["kind"] != "llm" or s.get("round_no") is None:
            continue
        if s.get("role") in ("generate", "refine"):
            gens[s["round_no"]] = s
        elif s.get("role") == "feedback":
            feedbacks[s["round_no"]] = s
    attempts = []
    for rnd in sorted(sandbox):
        sb, gen, crit = sandbox[rnd], gens.get(rnd), feedbacks.get(rnd + 1)
        attempts.append(
            {
                "round": rnd,
                "gen_prompt": gen.get("prompt") if gen else None,
                "gen_text": gen.get("text") if gen else None,
                "code": sb.get("code", ""),
                "flags": sb.get("flags", []),
                "platform": sb.get("platform") or PLATFORM_BY_ARM["A"],
                "result": _result(sb),
                "critique": crit.get("text") if crit else None,
                "critique_prompt": crit.get("prompt") if crit else None,
                "files": [f["_file"] for f in (gen, sb, crit) if f],
            }
        )
    memory_file = directory / "memory.md"
    by_role = {s.get("role"): s for s in steps if s["kind"] == "llm"}
    reused = next((s for s in steps if s["kind"] == "memory_reused"), None)
    return {
        "platform_inferred": any("platform" not in s for s in sandbox.values()),
        "reused_from": reused.get("source") if reused else None,
        "attempts": attempts,
        "memory_md": memory_file.read_text() if memory_file.exists() else None,
        "memory_reflection": _llm_meta(by_role.get("memory_reflection")),
        "memory_shorten": _llm_meta(by_role.get("memory_shorten")),
        "truncated": any(s["kind"] == "memory_truncated" for s in steps),
    }


def _arm_trials(directory: Path, arm: str) -> dict[str, Any] | None:
    """Arm B, C or D: trials with the memory in effect and the lesson written after a failure."""
    if not directory.is_dir():
        return None
    steps = _read_steps(directory)
    if not steps:
        return None
    by_number = {int(s["step"]): s for s in steps}
    gens = {s["trial"]: s for s in steps if s["kind"] == "llm" and s.get("role") == "generate"}
    sandbox = {s["trial"]: s for s in steps if s["kind"] == "sandbox" and "trial" in s}
    updates = {s["trial"]: s for s in steps if s["kind"] == "memory_update" and "trial" in s}
    trials = []
    for trial in sorted(sandbox):
        sb, gen, upd = sandbox[trial], gens.get(trial), updates.get(trial)
        lesson_llm = by_number.get(int(upd["step"]) - 1) if upd else None
        trials.append(
            {
                "trial": trial,
                "prompt": gen.get("prompt") if gen else None,
                "text": gen.get("text") if gen else None,
                "code": sb.get("code", ""),
                "flags": sb.get("flags", []),
                "platform": sb.get("platform") or PLATFORM_BY_ARM[arm],
                "result": _result(sb),
                "memory": sb.get("memory", []),
                "lesson": upd.get("lesson") if upd else None,
                "lesson_prompt": lesson_llm.get("prompt") if lesson_llm else None,
            }
        )
    first_prompt = trials[0]["prompt"] if trials else None
    seen = None
    if first_prompt and MEMORY_HEADER in first_prompt:
        seen = first_prompt.split(MEMORY_HEADER, 1)[1]
    return {
        "platform_inferred": any("platform" not in s for s in sandbox.values()),
        "trials": trials,
        "memory_seen": seen,
    }


def _transcript(directory: Path) -> list[Step]:
    """Every step of one arm directory exactly as traced, in order (unknown kinds included)."""
    if not directory.is_dir():
        return []
    entries = []
    for step in _read_steps(directory):
        entry = {k: v for k, v in step.items() if k != "_file"}
        entry["file"] = step["_file"]
        entries.append(entry)
    return entries


def _errors(directory: Path) -> list[dict[str, str]]:
    if not directory.is_dir():
        return []
    return [
        {"kind": s["kind"], "error": str(s.get("error", ""))}
        for s in _read_steps(directory)
        if s["kind"] in ("llm_error", "infra_error")
    ]


def _repeat_numbers(run_dir: Path) -> list[int]:
    numbers: set[int] = set()
    for arm in (*ARMS, "run"):
        arm_dir = run_dir / arm
        if arm_dir.is_dir():
            for child in arm_dir.iterdir():
                match = _REPEAT_DIR.match(child.name)
                if match:
                    numbers.add(int(match.group(1)))
    return sorted(numbers)


def _task_text(repeats: Sequence[dict[str, Any]]) -> str | None:
    for rep in repeats:
        if rep["A"] and rep["A"]["attempts"] and rep["A"]["attempts"][0]["gen_prompt"]:
            return str(rep["A"]["attempts"][0]["gen_prompt"])
    for rep in repeats:
        for arm in ("B", "C", "D"):
            if rep[arm] and rep[arm]["trials"] and rep[arm]["trials"][0]["prompt"]:
                return str(rep[arm]["trials"][0]["prompt"]).split(MEMORY_HEADER, 1)[0]
    return None


def load_run(run_dir: Path) -> dict[str, Any]:
    """Normalize one run directory into the structure the viewer renders."""
    meta_path = run_dir / "run_meta.json"
    meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    repeats: list[dict[str, Any]] = []
    for number in _repeat_numbers(run_dir):
        sub = f"r{number}"
        repeats.append(
            {
                "repeat": number,
                "A": _arm_a(run_dir / "A" / sub),
                "B": _arm_trials(run_dir / "B" / sub, "B"),
                "C": _arm_trials(run_dir / "C" / sub, "C"),
                "D": _arm_trials(run_dir / "D" / sub, "D"),
                "errors": _errors(run_dir / "run" / sub),
                "transcript": {arm: _transcript(run_dir / arm / sub) for arm in ARMS},
            }
        )
    cost, calls = 0.0, 0
    for arm in ARMS:
        for path in (run_dir / arm).glob("r*/*.json") if (run_dir / arm).is_dir() else []:
            if _STEP_FILE.match(path.name) and "llm" in path.name:
                data = json.loads(path.read_text())
                cost += float(data.get("cost_usd", 0.0))
                calls += 1
    task_text = _task_text(repeats)
    sha = meta.get("task_sha256") or (
        hashlib.sha256(task_text.encode()).hexdigest() if task_text else ""
    )
    return {
        "id": run_dir.name,
        "dev": any(marker in run_dir.name for marker in DEV_MARKERS),
        "meta": meta,
        "task_text": task_text,
        "task_sha256": sha,
        # traces from before the `platform` field existed (older harness: quota-only CPU limit,
        # no odd-size check); the viewer infers their platform from the arm and says so
        "platform_inferred": any(
            arm and arm["platform_inferred"] for rep in repeats for arm in (rep[name] for name in ARMS)
        ),
        "cost_usd": round(cost, 4),
        "llm_calls": calls,
        "repeats": repeats,
    }


def build_data(traces_root: Path) -> dict[str, Any]:
    """Load every run under `traces_root`, sorted by run id."""
    runs = [
        load_run(path)
        for path in sorted(traces_root.iterdir())
        if path.is_dir() and any((path / arm).is_dir() for arm in ARMS)
    ]
    return {"runs": runs}


def _safe_json(data: Any) -> str:
    """JSON that can sit inside a <script> element no matter what the traces contain."""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    for char, escaped in (
        ("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
        (" ", "\\u2028"), (" ", "\\u2029"),
    ):  # fmt: skip
        text = text.replace(char, escaped)
    return text


def intern_texts(data: Any) -> dict[str, Any]:
    """Move every long string into one shared table, leaving `{"$t": index}` references.

    The transcript repeats prompts that already sit in the normalized fields; interning stores
    each distinct long string once, so the page does not carry them twice.
    """
    index: dict[str, int] = {}
    texts: list[str] = []

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            if len(node) < INTERN_MIN_CHARS:
                return node
            if node not in index:
                index[node] = len(texts)
                texts.append(node)
            return {REF_KEY: index[node]}
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, dict):
            if REF_KEY in node:
                raise ValueError(f"data contains a {REF_KEY!r} key, which is reserved for references")
            return {key: walk(value) for key, value in node.items()}
        return node

    return {DATA_KEY: walk(data), TEXTS_KEY: texts}


def restore_texts(wrapped: dict[str, Any]) -> Any:
    """Inverse of `intern_texts`."""
    texts = wrapped[TEXTS_KEY]

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, dict):
            if set(node) == {REF_KEY}:
                return texts[node[REF_KEY]]
            return {key: walk(value) for key, value in node.items()}
        return node

    return walk(wrapped[DATA_KEY])


def render_html(data: dict[str, Any], template: str) -> str:
    """Inject `data` (long strings interned) into the template at the data placeholder."""
    if PLACEHOLDER not in template:
        raise ValueError(f"template has no {PLACEHOLDER!r} placeholder")
    return template.replace(PLACEHOLDER, _safe_json(intern_texts(data)))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: build `viewer/index.html` from `traces/`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=ROOT.parent / "traces")
    parser.add_argument("--out", type=Path, default=ROOT / "index.html")
    parser.add_argument("--template", type=Path, default=ROOT / "template.html")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    html = render_html(build_data(args.traces), args.template.read_text())
    args.out.write_text(html)
    logger.info("wrote %s (%.1f MB)", args.out, len(html.encode()) / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
