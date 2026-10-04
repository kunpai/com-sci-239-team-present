from pathlib import Path

from src.config import Config
from src.memory import EpisodicMemory
from src.reflexion import reflexion_trials
from src.traces import TraceScope
from tests.conftest import FakeLLM, code_reply, fail, ok, scripted_run


def _scope(tmp_path: Path) -> TraceScope:
    return TraceScope(tmp_path, "C", 1)


def test_first_try_success_makes_one_call(tmp_path: Path) -> None:
    llm = FakeLLM([code_reply("t1")])
    out = reflexion_trials(
        "TASK", llm, scripted_run({"t1": ok(2.0)}), EpisodicMemory(("x86 note",), 3),
        Config(), _scope(tmp_path),
    )  # fmt: skip
    assert out.first_success_trial == 1 and len(out.trials) == 1 and len(llm.prompts) == 1
    assert "x86 note" in llm.prompts[0]  # memory is appended to the first prompt


def test_failures_write_lessons_and_memory_grows_with_original_first(tmp_path: Path) -> None:
    llm = FakeLLM([code_reply("t1"), "lesson one", code_reply("t2"), "lesson two", code_reply("t3")])
    run_fn = scripted_run({"t1": fail("compile"), "t2": fail("compile"), "t3": ok(1.5)})
    out = reflexion_trials(
        "TASK", llm, run_fn, EpisodicMemory(("original",), 3), Config(), _scope(tmp_path)
    )
    assert out.first_success_trial == 3
    assert out.memory.entries == ("original", "lesson one", "lesson two")
    assert "lesson one" in llm.prompts[2] and "lesson two" in llm.prompts[4]


def test_all_trials_fail_and_no_lesson_after_the_last(tmp_path: Path) -> None:
    llm = FakeLLM([code_reply("t1"), "l1", code_reply("t2"), "l2", code_reply("t3")])
    run_fn = scripted_run({"t1": fail("compile"), "t2": fail("run"), "t3": fail("incorrect")})
    out = reflexion_trials("TASK", llm, run_fn, EpisodicMemory((), 3), Config(), _scope(tmp_path))
    assert out.first_success_trial is None and len(out.trials) == 3
    assert len(llm.prompts) == 5  # 3 generations + 2 lessons, none after the final failure
    assert out.memory.entries == ("l1", "l2")
