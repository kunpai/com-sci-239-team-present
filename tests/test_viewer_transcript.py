"""The transcript (every trace step in order) and the shared-string table that keeps it small."""

import json
import re
from pathlib import Path

import pytest

from run import run_experiment
from src.config import Config
from src.sandbox import AMD64, ARMV7
from tests.conftest import FakeLLM, code_reply, ok, scripted_run
from tests.test_viewer_build import MEM, NAIVE, _make_run
from viewer.build import intern_texts, load_run, render_html, restore_texts


def _files(run_dir: Path, arm: str) -> list[str]:
    names = (p.name for p in (run_dir / arm / "r1").glob("[0-9]*-*.json"))
    return sorted(names, key=lambda n: int(n.split("-", 1)[0]))  # numeric, as step order is


def test_transcript_lists_every_trace_file_in_step_order_per_arm(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    transcript = load_run(run_dir)["repeats"][0]["transcript"]
    for arm in ("A", "B", "C"):
        assert [t["file"] for t in transcript[arm]] == _files(run_dir, arm)
    assert [t["step"] for t in transcript["A"]] == list(range(1, len(transcript["A"]) + 1))


def test_transcript_keeps_the_raw_lesson_reply_and_the_memory_reflection(tmp_path: Path) -> None:
    transcript = load_run(_make_run(tmp_path))["repeats"][0]["transcript"]
    lessons = [t for t in transcript["B"] if t["kind"] == "llm" and t["role"] == "lesson"]
    assert [t["text"] for t in lessons] == ["lesson b"] and "Your attempt failed" in lessons[0]["prompt"]
    assert any(t["kind"] == "memory_update" and t["lesson"] == "lesson b" for t in transcript["B"])
    reflection = [t for t in transcript["A"] if t.get("role") == "memory_reflection"]
    assert reflection[0]["text"] == MEM


def test_transcript_keeps_sandbox_steps_with_code_and_result(tmp_path: Path) -> None:
    transcript = load_run(_make_run(tmp_path))["repeats"][0]["transcript"]
    sandbox = [t for t in transcript["C"] if t["kind"] == "sandbox"]
    assert [t["result"]["stage"] for t in sandbox] == ["compile", "ok"]
    assert "// c1" in sandbox[0]["code"] and sandbox[0]["platform"] == "armv7"
    assert sandbox[0]["result"]["stderr"] == "no such header"


def test_unknown_step_kinds_are_kept_verbatim(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    (run_dir / "B" / "r1" / "90-mystery.json").write_text(json.dumps({"foo": "bar", "n": 3}))
    transcript = load_run(run_dir)["repeats"][0]["transcript"]
    mystery = transcript["B"][-1]
    assert mystery["kind"] == "mystery" and mystery["step"] == 90
    assert mystery["foo"] == "bar" and mystery["n"] == 3 and mystery["file"] == "90-mystery.json"


def test_aborted_repeat_has_empty_transcripts(tmp_path: Path) -> None:
    from tests.test_viewer_build import _DyingLLM

    llm = _DyingLLM([code_reply("a0"), "mem", code_reply("b1"), code_reply("c1")])
    tables = {AMD64.name: {"a0": ok(2.0)}, ARMV7.name: {"b1": ok(1.0), "c1": ok(1.0)}}
    run_dir = run_experiment(
        Config(rounds=0, repeats=2, trials=1), llm, lambda p: scripted_run(tables[p.name]),
        NAIVE, tmp_path, "matxfer_d",
    )  # fmt: skip
    first = load_run(run_dir)["repeats"][0]
    assert first["transcript"] == {"A": [], "B": [], "C": [], "D": []}


def test_reused_memory_run_shows_the_reuse_in_arm_a_and_its_transcript(tmp_path: Path) -> None:
    source = tmp_path / "src_run" / "A" / "r1"
    source.mkdir(parents=True)
    (source / "memory.md").write_text("old notes\n")
    llm = FakeLLM([code_reply("b1"), code_reply("c1")])
    tables = {ARMV7.name: {"b1": ok(1.0), "c1": ok(1.0)}}
    run_dir = run_experiment(
        Config(trials=1), llm, lambda p: scripted_run(tables[p.name]), NAIVE, tmp_path, "matxfer_r",
        memory_source=tmp_path / "src_run",
    )  # fmt: skip
    rep = load_run(run_dir)["repeats"][0]
    assert rep["A"]["attempts"] == [] and rep["A"]["memory_md"].strip() == "old notes"
    assert rep["A"]["reused_from"].endswith("src_run")
    assert [t["kind"] for t in rep["transcript"]["A"]] == ["memory_reused"]


# --- shared-string table -----------------------------------------------------------------------

LONG = "x" * 400


def test_intern_and_restore_round_trip() -> None:
    data = {"runs": [{"a": LONG, "b": [LONG, "short", {"c": LONG + "y"}], "n": 3, "z": None}]}
    wrapped = intern_texts(data)
    assert restore_texts(wrapped) == data
    assert wrapped["$texts"].count(LONG) == 1 and len(wrapped["$texts"]) == 2  # LONG and LONG+"y"
    assert "short" in json.dumps(wrapped)  # short strings stay inline


def test_intern_replaces_each_long_string_with_a_reference() -> None:
    wrapped = intern_texts({"a": LONG, "b": LONG})
    assert wrapped["$data"] == {"a": {"$t": 0}, "b": {"$t": 0}}


def test_intern_refuses_data_that_looks_like_a_reference() -> None:
    with pytest.raises(ValueError, match=r"\$t"):
        intern_texts({"a": {"$t": 0}})


def test_render_html_stores_a_repeated_long_string_once_and_stays_safe() -> None:
    template = '<script id="d" type="application/json">/*__DATA__*/null</script>'
    evil = "</script><img src=x onerror=1>" + LONG
    html = render_html({"runs": [{"p": evil, "q": evil, "r": evil}]}, template)
    assert html.count(LONG) == 1 and "<img" not in html and html.count("</script>") == 1
    match = re.search(r'type="application/json">(.*?)</script>', html, re.DOTALL)
    assert match is not None
    assert restore_texts(json.loads(match.group(1)))["runs"][0]["q"] == evil


def test_a_normal_run_has_no_reused_from(tmp_path: Path) -> None:
    assert load_run(_make_run(tmp_path))["repeats"][0]["A"]["reused_from"] is None


def test_step_file_that_is_not_a_json_object_is_skipped_not_fatal(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    (run_dir / "B" / "r1" / "91-weird.json").write_text("[1, 2, 3]")
    transcript = load_run(run_dir)["repeats"][0]["transcript"]
    assert "91-weird.json" not in [t["file"] for t in transcript["B"]]


def test_intern_boundary_is_255_inline_256_interned() -> None:
    assert intern_texts({"a": "x" * 255})["$data"] == {"a": "x" * 255}
    assert intern_texts({"a": "x" * 256})["$data"] == {"a": {"$t": 0}}


def test_intern_refuses_a_reference_key_next_to_other_keys() -> None:
    with pytest.raises(ValueError, match=r"\$t"):
        intern_texts({"$t": 0, "other": 1})


# --- arm D (x86 + A's memory) ------------------------------------------------------------------


def _arm_d_run(tmp_path: Path) -> Path:
    source = tmp_path / "src_run" / "A" / "r1"
    source.mkdir(parents=True)
    (source / "memory.md").write_text("old notes\n")
    llm = FakeLLM([code_reply("d1")])
    tables = {AMD64.name: {"d1": ok(3.0)}}
    return run_experiment(
        Config(trials=1), llm, lambda p: scripted_run(tables[p.name]), NAIVE, tmp_path, "matxfer_d",
        memory_source=tmp_path / "src_run", arm_d=True,
    )  # fmt: skip


def test_arm_d_is_loaded_with_its_trial_memory_and_amd64_platform(tmp_path: Path) -> None:
    rep = load_run(_arm_d_run(tmp_path))["repeats"][0]
    assert rep["B"] is None and rep["C"] is None
    d = rep["D"]
    assert d["trials"][0]["platform"] == "amd64" and d["trials"][0]["memory"] == ["old notes"]
    assert d["memory_seen"] == "- old notes\n"


def test_arm_d_has_a_transcript_and_counts_toward_cost_and_calls(tmp_path: Path) -> None:
    run = load_run(_arm_d_run(tmp_path))
    assert [t["kind"] for t in run["repeats"][0]["transcript"]["D"]] == ["llm", "sandbox"]
    assert run["llm_calls"] == 1


def test_a_d_only_run_still_has_the_task_text_and_is_listed(tmp_path: Path) -> None:
    from viewer.build import build_data

    run_dir = _arm_d_run(tmp_path)
    assert load_run(run_dir)["task_text"]
    assert "matxfer_d" in [r["id"] for r in build_data(tmp_path)["runs"]]


def test_older_runs_have_a_null_d_arm(tmp_path: Path) -> None:
    assert load_run(_make_run(tmp_path))["repeats"][0]["D"] is None
