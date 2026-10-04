"""Scalar Reflexion feedback, memory reuse across runs, and LLM call retries."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from run import run_experiment
from src.candidate import Step
from src.config import Config
from src.llm import ClaudeCodeLLM, LLMError
from src.memory import EpisodicMemory, reflect_on_failure
from src.prompts import failure_lesson_prompt, scalar_observation
from src.sandbox import AMD64, ARMV7, Platform, RunResult
from src.traces import TraceWriter
from tests.conftest import FakeLLM, code_reply, fail, ok, scripted_run

NAIVE = {"amd64": 100.0, "armv7": 900.0}
STDERR = "fatal error: immintrin.h: No such file or directory"


def _failed_step() -> Step:
    result = RunResult("compile", False, False, None, None, STDERR)
    return Step("void matmul() {}", ("-O3",), result, None)


def test_config_defaults_to_diagnostic_feedback() -> None:
    assert Config().feedback == "diagnostic"


def test_config_rejects_unknown_feedback_mode() -> None:
    with pytest.raises(ValueError, match="feedback"):
        Config(feedback="bogus")


def test_diagnostic_lesson_prompt_shows_stage_and_compiler_output() -> None:
    prompt = failure_lesson_prompt("TASK", _failed_step(), (), 80)
    assert "Compilation failed" in prompt and "immintrin.h" in prompt


def test_scalar_lesson_prompt_shows_pass_fail_flags_and_no_diagnostics() -> None:
    prompt = failure_lesson_prompt("TASK", _failed_step(), (), 80, feedback="scalar")
    assert "void matmul() {}" in prompt and "-O3" in prompt  # the trajectory is still shown
    assert "Compiled: no. Correct: no. Speedup: n/a." in prompt
    assert "immintrin.h" not in prompt
    assert "Compilation failed" not in prompt and "Wrong result" not in prompt


def test_scalar_observation_distinguishes_wrong_answer_from_compile_failure() -> None:
    wrong = Step("c", (), RunResult("incorrect", True, False, None, None, "mismatch at 3"), None)
    prompt = failure_lesson_prompt("TASK", wrong, (), 80, feedback="scalar")
    assert "Compiled: yes. Correct: no. Speedup: n/a." in prompt and "mismatch" not in prompt


def test_scalar_observation_reports_speedup_for_a_passing_result() -> None:
    assert scalar_observation(RunResult("ok", True, True, 50.0, 2.5, "")) == (
        "Compiled: yes. Correct: yes. Speedup: 2.50x."
    )


def test_reflect_on_failure_uses_the_configured_feedback_mode(tmp_path: Path) -> None:
    llm = FakeLLM(["lesson"])
    scope = TraceWriter(tmp_path, "t").scope("B", 1)
    cfg = Config(feedback="scalar")
    reflect_on_failure(llm, "TASK", _failed_step(), EpisodicMemory(), cfg, scope)
    assert "immintrin.h" not in llm.prompts[0] and "Speedup: n/a" in llm.prompts[0]


def _run_fn_for(amd: dict, arm: dict):
    tables = {AMD64.name: amd, ARMV7.name: arm}

    def for_platform(platform: Platform):
        return scripted_run(tables[platform.name])

    return for_platform


def _source_run(root: Path, memories: dict[int, str]) -> Path:
    for repeat, text in memories.items():
        directory = root / "src_run" / "A" / f"r{repeat}"
        directory.mkdir(parents=True)
        (directory / "memory.md").write_text(text + "\n")
    return root / "src_run"


def test_memory_source_skips_arm_a_and_feeds_the_stored_memory_to_c(tmp_path: Path) -> None:
    source = _source_run(tmp_path, {1: "use avx2 intrinsics", 3: "block the loops"})
    cfg = Config(repeats=9, trials=2, feedback="scalar")  # repeats is ignored when reusing
    llm = FakeLLM(
        [
            code_reply("b1"), "lesson b", code_reply("b2"),  # repeat 1, arm B
            code_reply("c1"), "lesson c", code_reply("c2"),  # repeat 1, arm C
            code_reply("b3"), code_reply("c3"),  # repeat 3: both succeed first try
        ]
    )
    arm = {
        "b1": fail("compile", STDERR), "b2": ok(1.2), "c1": fail("compile", STDERR),
        "c2": ok(1.1), "b3": ok(1.0), "c3": ok(1.0),
    }  # fmt: skip
    run_dir = run_experiment(
        cfg, llm, _run_fn_for({}, arm), NAIVE, tmp_path, "reuse", memory_source=source
    )

    data = json.loads((run_dir / "results.json").read_text())
    assert sorted({a["repeat"] for a in data["arms"]}) == [1, 3]
    assert data["refine"] == []
    assert "use avx2 intrinsics" in llm.prompts[3]  # arm C, repeat 1, first prompt
    assert "use avx2 intrinsics" not in llm.prompts[0]  # arm B never sees memory
    assert all("immintrin.h" not in p for p in llm.prompts)  # scalar mode hides compiler output
    copied = (run_dir / "A" / "r1" / "memory.md").read_text().strip()
    assert copied == "use avx2 intrinsics"
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["memory_source"] == str(source)
    assert meta["config"]["feedback"] == "scalar"


def test_memory_source_without_any_memory_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty" / "A" / "r1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="memory"):
        run_experiment(
            Config(), FakeLLM([]), _run_fn_for({}, {}), NAIVE, tmp_path, "x",
            memory_source=tmp_path / "empty",
        )  # fmt: skip


def test_reuse_with_aborted_repeat_keeps_the_others(tmp_path: Path) -> None:
    source = _source_run(tmp_path, {2: "notes"})
    cfg = Config(trials=1)
    llm = FakeLLM([code_reply("b1"), code_reply("c1")])
    run_fn_for = _run_fn_for({}, {"b1": ok(1.0), "c1": ok(1.0)})
    run_dir = run_experiment(cfg, llm, run_fn_for, NAIVE, tmp_path, "r", memory_source=source)
    data = json.loads((run_dir / "results.json").read_text())
    assert {a["arm"] for a in data["arms"]} == {"B", "C"}


# --- LLM call retries with backoff -------------------------------------------------------------

RESULT = {"type": "result", "result": "OK", "session_id": "s", "total_cost_usd": 0.0,
          "duration_ms": 1, "usage": {}, "modelUsage": {}}  # fmt: skip


def _completed(stdout: str, rc: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], rc, stdout=stdout, stderr="")


def test_retries_back_off_between_attempts_then_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(1)
        if len(calls) < 3:
            raise subprocess.TimeoutExpired(cmd, 1)
        return _completed(json.dumps(RESULT))

    sleeps: list[float] = []
    monkeypatch.setattr("src.llm.subprocess.run", fake_run)
    llm = ClaudeCodeLLM(timeout_s=5, backoff_s=(30, 60, 120), sleep=sleeps.append)
    assert llm("hi").text == "OK"
    assert len(calls) == 3 and sleeps == [30, 60]


def test_gives_up_after_the_whole_backoff_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm.subprocess.run", lambda cmd, **kw: _completed("", rc=1))
    llm = ClaudeCodeLLM(timeout_s=5, backoff_s=(30, 60, 120), sleep=sleeps.append)
    with pytest.raises(LLMError, match="4 attempts"):
        llm("hi")
    assert sleeps == [30, 60, 120]  # no sleep after the final failure


def test_default_schedule_is_30_60_120(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.llm.subprocess.run", lambda cmd, **kw: _completed("", rc=1))
    monkeypatch.setattr("src.llm.time.sleep", sleeps.append)
    with pytest.raises(LLMError):
        ClaudeCodeLLM(timeout_s=5)("hi")
    assert sleeps == [30, 60, 120]


def test_config_carries_the_feedback_mode_replace() -> None:
    assert replace(Config(), feedback="scalar").feedback == "scalar"
