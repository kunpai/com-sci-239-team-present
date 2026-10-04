import json
from pathlib import Path

from run import run_experiment
from src.config import Config
from src.llm import LLMError, LLMReply
from src.sandbox import AMD64, ARMV7, Platform
from tests.conftest import FakeLLM, code_reply, fail, ok, scripted_run

NAIVE = {"amd64": 100.0, "armv7": 900.0}


def _run_fn_for(amd: dict, arm: dict):
    tables = {AMD64.name: amd, ARMV7.name: arm}

    def for_platform(platform: Platform):
        return scripted_run(tables[platform.name])

    return for_platform


def test_full_repeat_with_fakes_writes_traces_memory_and_report(tmp_path: Path) -> None:
    cfg = Config(rounds=1, repeats=1, trials=2)
    llm = FakeLLM(
        [
            code_reply("a0"), "fb", code_reply("a1"),  # arm A: generate, feedback, refine
            "memory: use avx2 intrinsics",  # memory reflection
            code_reply("b1"), "lesson b", code_reply("b2"),  # arm B: fail, lesson, succeed
            code_reply("c1"), "lesson c", code_reply("c2"),  # arm C: fail, lesson, succeed
        ]
    )
    run_fn_for = _run_fn_for(
        {"a0": ok(1.0), "a1": ok(4.0)},
        {"b1": fail("compile"), "b2": ok(1.2), "c1": fail("compile"), "c2": ok(1.1)},
    )
    run_dir = run_experiment(cfg, llm, run_fn_for, NAIVE, tmp_path, "t1")

    assert (run_dir / "A" / "r1" / "memory.md").read_text().strip() == "memory: use avx2 intrinsics"
    assert "memory: use avx2 intrinsics" in llm.prompts[4 + 3]  # arm C's first prompt (after B)
    assert json.loads((run_dir / "run_meta.json").read_text())["naive_ms"] == NAIVE
    data = json.loads((run_dir / "results.json").read_text())
    assert [a["arm"] for a in data["arms"]] == ["B", "C"]
    assert (run_dir / "results.md").exists() and (run_dir / "README.md").exists()


def test_no_correct_candidate_in_arm_a_skips_arm_c(tmp_path: Path) -> None:
    cfg = Config(rounds=1, repeats=1, trials=1)
    llm = FakeLLM([code_reply("a0"), "fb", code_reply("a1"), code_reply("b1")])
    run_fn_for = _run_fn_for(
        {"a0": fail("compile"), "a1": fail("incorrect")}, {"b1": ok(1.0)}
    )
    run_dir = run_experiment(cfg, llm, run_fn_for, NAIVE, tmp_path, "t2")
    data = json.loads((run_dir / "results.json").read_text())
    arms = {a["arm"]: a for a in data["arms"]}
    assert arms["B"]["first_try_success"] and arms["C"]["skipped"]
    assert data["refine"][0]["memory"] is None


class _DyingLLM(FakeLLM):
    """Raises LLMError on the first call of the first repeat, then behaves normally."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies)
        self._died = False

    def __call__(self, prompt: str) -> LLMReply:
        if not self._died:
            self._died = True
            raise LLMError("boom")
        return super().__call__(prompt)


def test_llm_failure_in_one_repeat_does_not_lose_the_others(tmp_path: Path) -> None:
    cfg = Config(rounds=0, repeats=2, trials=1)
    llm = _DyingLLM([code_reply("a0"), "memory", code_reply("b1"), code_reply("c1")])
    run_fn_for = _run_fn_for({"a0": ok(2.0)}, {"b1": ok(1.0), "c1": ok(1.0)})
    run_dir = run_experiment(cfg, llm, run_fn_for, NAIVE, tmp_path, "t3")
    data = json.loads((run_dir / "results.json").read_text())
    assert [r["repeat"] for r in data["refine"]] == [2]
    assert (run_dir / "run" / "r1" / "01-llm_error.json").exists()


def test_run_meta_records_the_task_file_relative_to_the_repo(tmp_path: Path) -> None:
    from run import write_run_meta
    from src.prompts import TASK_PATH

    write_run_meta(tmp_path, Config(), NAIVE, TASK_PATH, "task")
    recorded = json.loads((tmp_path / "run_meta.json").read_text())["task_file"]
    assert recorded == "task/prompt_task.md"
