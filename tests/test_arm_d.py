"""Arm D: x86 plus arm A's memory, run on amd64 with no baseline arm."""

import json
from pathlib import Path

import pytest

from run import main, run_experiment
from src.config import Config
from src.sandbox import AMD64, ARMV7
from tests.conftest import FakeLLM, code_reply, ok, scripted_run
from tests.test_scalar_feedback import NAIVE, _source_run


def _amd64_only(table: dict):
    """Run fn that only knows amd64: an arm/v7 run would raise KeyError."""

    def for_platform(platform):
        return scripted_run({AMD64.name: table}[platform.name])

    return for_platform


def _arm_d_run(tmp_path: Path, memories: dict[int, str], replies: list[str], table: dict, **kw):
    source = _source_run(tmp_path, memories)
    llm = FakeLLM(replies)
    run_dir = run_experiment(
        Config(trials=3), llm, _amd64_only(table), NAIVE, tmp_path, "d",
        memory_source=source, arm_d=True, **kw,
    )  # fmt: skip
    return run_dir, llm


def test_arm_d_runs_only_the_memory_arm_on_amd64(tmp_path: Path) -> None:
    run_dir, llm = _arm_d_run(tmp_path, {1: "use avx2 intrinsics"}, [code_reply("d1")], {"d1": ok(2.0)})
    assert (run_dir / "D" / "r1").is_dir()
    assert not (run_dir / "B").exists() and not (run_dir / "C").exists()
    assert len(llm.prompts) == 1 and "use avx2 intrinsics" in llm.prompts[0]
    platforms = {json.loads(p.read_text())["platform"] for p in run_dir.glob("D/r1/*-sandbox.json")}
    assert platforms == {"amd64"}


def test_arm_d_retries_until_the_first_success_like_the_other_arms(tmp_path: Path) -> None:
    from tests.conftest import fail

    table = {"d1": fail("compile", "boom"), "d2": ok(1.5)}
    run_dir, _ = _arm_d_run(
        tmp_path, {1: "notes"}, [code_reply("d1"), "lesson", code_reply("d2")], table
    )
    data = json.loads((run_dir / "results.json").read_text())
    assert [(a["arm"], a["trials_to_success"]) for a in data["arms"]] == [("D", 2)]


def test_arm_d_is_recorded_in_run_meta_and_reports(tmp_path: Path) -> None:
    run_dir, _ = _arm_d_run(tmp_path, {1: "notes"}, [code_reply("d1")], {"d1": ok(1.0)})
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["target"] == "amd64" and meta["arms"] == ["D"]
    results = (run_dir / "results.md").read_text()
    assert "| D |" in results and "| B |" not in results and "| C |" not in results
    assert "`D`" in (run_dir / "README.md").read_text()


def test_default_run_still_targets_armv7_with_arms_b_and_c(tmp_path: Path) -> None:
    source = _source_run(tmp_path, {1: "notes"})
    llm = FakeLLM([code_reply("b1"), code_reply("c1")])

    def for_platform(platform):
        return scripted_run({ARMV7.name: {"b1": ok(1.0), "c1": ok(1.0)}}[platform.name])

    run_dir = run_experiment(
        Config(trials=1), llm, for_platform, NAIVE, tmp_path, "arm", memory_source=source
    )
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["target"] == "armv7" and meta["arms"] == ["B", "C"]


def test_only_repeats_selects_stored_memories(tmp_path: Path) -> None:
    run_dir, llm = _arm_d_run(
        tmp_path, {1: "memo-alpha", 2: "memo-beta", 3: "memo-gamma"},
        [code_reply("d1")], {"d1": ok(1.0)}, only_repeats=[2],
    )  # fmt: skip
    data = json.loads((run_dir / "results.json").read_text())
    assert sorted({a["repeat"] for a in data["arms"]}) == [2]
    assert "memo-beta" in llm.prompts[0] and "memo-alpha" not in llm.prompts[0]


def test_only_repeats_naming_a_missing_memory_is_an_error(tmp_path: Path) -> None:
    source = _source_run(tmp_path, {1: "one"})
    with pytest.raises(FileNotFoundError, match="repeat"):
        run_experiment(
            Config(), FakeLLM([]), _amd64_only({}), NAIVE, tmp_path, "bad",
            memory_source=source, arm_d=True, only_repeats=[5],
        )  # fmt: skip


def test_cli_exposes_arm_d_and_rejects_the_old_target_flag() -> None:
    with pytest.raises(SystemExit):
        main(["--target", "riscv"])
