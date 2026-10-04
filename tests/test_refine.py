from pathlib import Path

from src.config import Config
from src.refine import self_refine
from src.traces import TraceScope
from tests.conftest import FakeLLM, code_reply, fail, ok, scripted_run


def test_self_refine_runs_rounds_and_keeps_history(tmp_path: Path) -> None:
    cfg = Config(rounds=2)
    llm = FakeLLM([code_reply("v0"), "fb0", code_reply("v1"), "fb1", code_reply("v2")])
    run_fn = scripted_run({"v0": ok(1.0), "v1": ok(3.0), "v2": ok(2.0)})
    out = self_refine("TASK", llm, run_fn, cfg, TraceScope(tmp_path, "A", 1))

    assert len(out.history) == 3
    assert [s.feedback for s in out.history] == ["fb0", "fb1", None]
    assert out.best == out.history[1]  # fastest correct, not the last
    assert "fb0" in llm.prompts[2]  # refine prompt carries the feedback
    assert "// v0" in llm.prompts[3]  # round-2 feedback prompt carries earlier attempts
    assert len(list(tmp_path.glob("*.json"))) == 8  # 5 llm calls + 3 sandbox runs


def test_unparseable_reply_is_a_failed_step_and_the_loop_continues(tmp_path: Path) -> None:
    cfg = Config(rounds=2)
    llm = FakeLLM([code_reply("v0"), "fb0", "I refuse to write code", "fb1", code_reply("v2")])
    run_fn = scripted_run({"v0": ok(1.0), "v2": ok(2.0)})
    out = self_refine("TASK", llm, run_fn, cfg, TraceScope(tmp_path, "A", 1))
    assert len(out.history) == 3
    assert out.history[1].result.stage == "parse"
    assert out.best == out.history[2]


def test_best_is_none_when_nothing_is_correct(tmp_path: Path) -> None:
    cfg = Config(rounds=1)
    llm = FakeLLM([code_reply("v0"), "fb0", code_reply("v1")])
    run_fn = scripted_run({"v0": fail("compile"), "v1": fail("incorrect")})
    out = self_refine("TASK", llm, run_fn, cfg, TraceScope(tmp_path, "A", 1))
    assert out.best is None and len(out.history) == 2
