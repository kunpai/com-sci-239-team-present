"""viewer.build turns harness traces into one normalized, safely embeddable data dict."""

import json
import re
from pathlib import Path

from run import run_experiment
from src.config import Config
from src.llm import LLMError, LLMReply
from src.prompts import load_task
from src.sandbox import AMD64, ARMV7
from tests.conftest import FakeLLM, code_reply, fail, ok, scripted_run
from viewer.build import build_data, load_run, render_html, restore_texts

NAIVE = {"amd64": 100.0, "armv7": 900.0}
MEM = "memory: use avx2 intrinsics"


def _make_run(tmp_path: Path, run_id: str = "matxfer_t") -> Path:
    cfg = Config(rounds=1, repeats=1, trials=2)
    llm = FakeLLM(
        [
            code_reply("a0"), "fb", code_reply("a1"), MEM,
            code_reply("b1"), "lesson b", code_reply("b2"),
            code_reply("c1"), "lesson c", code_reply("c2"),
        ]
    )  # fmt: skip
    tables = {
        AMD64.name: {"a0": ok(1.0), "a1": ok(4.0)},
        ARMV7.name: {
            "b1": fail("compile", "no such flag"), "b2": ok(1.2),
            "c1": fail("compile", "no such header"), "c2": ok(1.1),
        },
    }  # fmt: skip
    return run_experiment(
        cfg, llm, lambda p: scripted_run(tables[p.name]), NAIVE, tmp_path, run_id
    )


def test_arm_a_attempts_pair_generation_result_and_critique(tmp_path: Path) -> None:
    run = load_run(_make_run(tmp_path))
    attempts = run["repeats"][0]["A"]["attempts"]
    assert [a["round"] for a in attempts] == [0, 1]
    assert "// a0" in attempts[0]["code"] and attempts[0]["result"]["speedup"] == 1.0
    assert attempts[0]["critique"] == "fb"  # critique of attempt 0 is the round-1 feedback call
    assert attempts[1]["critique"] is None
    assert "fb" in attempts[1]["gen_prompt"]  # the refine prompt that produced attempt 1


def test_stored_memory_and_what_arm_c_was_shown_are_exact(tmp_path: Path) -> None:
    rep = load_run(_make_run(tmp_path))["repeats"][0]
    assert rep["A"]["memory_md"].strip() == MEM
    assert rep["C"]["memory_seen"] == f"- {MEM}\n"
    assert rep["B"]["memory_seen"] is None
    assert rep["A"]["truncated"] is False


def test_trials_carry_memory_in_effect_and_the_lesson_after_a_failure(tmp_path: Path) -> None:
    rep = load_run(_make_run(tmp_path))["repeats"][0]
    b, c = rep["B"]["trials"], rep["C"]["trials"]
    assert b[0]["result"]["stage"] == "compile" and b[0]["lesson"] == "lesson b"
    assert b[1]["memory"] == ["lesson b"] and b[1]["lesson"] is None
    assert c[0]["memory"] == [MEM] and c[1]["memory"] == [MEM, "lesson c"]
    assert b[0]["platform"] == "armv7" and rep["A"]["attempts"][0]["platform"] == "amd64"


def test_run_level_fields_task_text_cost_and_dev_flag(tmp_path: Path) -> None:
    quick = load_run(_make_run(tmp_path, "matxfer_quick"))
    assert quick["dev"] is True and quick["llm_calls"] == 10
    run = load_run(_make_run(tmp_path, "matxfer_full"))
    assert run["dev"] is False
    assert run["task_text"] == load_task()
    assert len(run["task_sha256"]) == 64 and run["meta"]["naive_ms"] == NAIVE


class _DyingLLM(FakeLLM):
    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies)
        self._died = False

    def __call__(self, prompt: str) -> LLMReply:
        if not self._died:
            self._died = True
            raise LLMError("boom")
        return super().__call__(prompt)


def test_aborted_repeat_is_listed_with_its_error_and_empty_arms(tmp_path: Path) -> None:
    cfg = Config(rounds=0, repeats=2, trials=1)
    llm = _DyingLLM([code_reply("a0"), "mem", code_reply("b1"), code_reply("c1")])
    tables = {AMD64.name: {"a0": ok(2.0)}, ARMV7.name: {"b1": ok(1.0), "c1": ok(1.0)}}
    run_dir = run_experiment(
        cfg, llm, lambda p: scripted_run(tables[p.name]), NAIVE, tmp_path, "matxfer_d"
    )
    reps = load_run(run_dir)["repeats"]
    assert [r["repeat"] for r in reps] == [1, 2]
    assert reps[0]["A"] is None and reps[0]["B"] is None and reps[0]["C"] is None
    assert reps[0]["errors"][0]["kind"] == "llm_error" and "boom" in reps[0]["errors"][0]["error"]
    assert reps[1]["A"] is not None and reps[1]["errors"] == []


def test_arm_c_is_none_when_it_was_skipped(tmp_path: Path) -> None:
    cfg = Config(rounds=0, repeats=1, trials=1)
    llm = FakeLLM([code_reply("a0"), code_reply("b1")])
    tables = {AMD64.name: {"a0": fail("compile")}, ARMV7.name: {"b1": ok(1.0)}}
    run_dir = run_experiment(
        cfg, llm, lambda p: scripted_run(tables[p.name]), NAIVE, tmp_path, "matxfer_s"
    )
    rep = load_run(run_dir)["repeats"][0]
    assert rep["C"] is None and rep["A"]["memory_md"] is None and rep["B"] is not None


def test_old_traces_without_platform_or_run_meta_still_load(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    for path in (run_dir / "B").rglob("*-sandbox.json"):
        data = json.loads(path.read_text())
        data.pop("platform", None)
        path.write_text(json.dumps(data))
    (run_dir / "run_meta.json").unlink()
    run = load_run(run_dir)
    assert run["meta"] == {} and len(run["task_sha256"]) == 64
    assert run["repeats"][0]["B"]["trials"][0]["platform"] == "armv7"  # inferred from the arm


def test_build_data_sorts_runs_and_is_deterministic(tmp_path: Path) -> None:
    _make_run(tmp_path, "matxfer_b")
    _make_run(tmp_path, "matxfer_a")
    first, second = build_data(tmp_path), build_data(tmp_path)
    assert [r["id"] for r in first["runs"]] == ["matxfer_a", "matxfer_b"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_render_html_cannot_be_broken_out_of_by_trace_text() -> None:
    template = '<script id="d" type="application/json">/*__DATA__*/null</script><script>app()</script>'
    data = {"runs": [{"text": "</script><img src=x onerror=alert(1)> <!-- &  "}]}
    html = render_html(data, template)
    assert html.count("</script>") == 2 and "<img" not in html and "<!--" not in html
    match = re.search(r'<script id="d" type="application/json">(.*?)</script>', html, re.DOTALL)
    assert match is not None and restore_texts(json.loads(match.group(1))) == data
