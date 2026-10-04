"""The task statement can be swapped (for ablations) and the choice is recorded."""

import json
import re
from pathlib import Path

from run import run_experiment
from src.config import Config
from src.prompts import TASK_PATH, load_task
from src.sandbox import AMD64, ARMV7
from tests.conftest import FakeLLM, code_reply, ok, scripted_run

GENERIC = TASK_PATH.parent / "prompt_task_generic.md"


def test_load_task_reads_a_custom_file(tmp_path: Path) -> None:
    path = tmp_path / "t.md"
    path.write_text("hello\n\n")
    assert load_task(path) == "hello\n"


def test_generic_task_keeps_the_contract_and_drops_environment_hints() -> None:
    text = load_task(GENERIC)
    lowered = text.lower()
    for needed in ("matmul", "FLAGS:", "row-major", "uninitialized", "every positive n"):
        assert needed in text, needed
    for dropped in ("container", "machine", "pthread", "openmp", "march=native", "cpu"):
        assert dropped not in lowered, dropped
    assert not re.search(r"\b(x86|amd64|arm|avx|neon|sse|32-bit|64-bit)\b", lowered)


def test_run_experiment_uses_the_given_task_and_records_it(tmp_path: Path) -> None:
    task_file = tmp_path / "task.md"
    task_file.write_text("CUSTOM TASK\n")
    cfg = Config(rounds=0, repeats=1, trials=1)
    llm = FakeLLM([code_reply("a0"), "memory", code_reply("b1"), code_reply("c1")])
    tables = {AMD64.name: {"a0": ok(2.0)}, ARMV7.name: {"b1": ok(1.0), "c1": ok(1.0)}}
    run_dir = run_experiment(
        cfg, llm, lambda p: scripted_run(tables[p.name]), {"amd64": 1.0, "armv7": 1.0},
        tmp_path, "t", task_path=task_file,
    )  # fmt: skip
    assert llm.prompts[0].startswith("CUSTOM TASK")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["task_file"] == str(task_file) and len(meta["task_sha256"]) == 64
