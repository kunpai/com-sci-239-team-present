import json
from pathlib import Path

from src.candidate import Step
from src.config import Config
from src.memory import EpisodicMemory
from src.refine import RefineOutcome
from src.reflexion import TrialOutcome
from src.report import (
    ArmRecord,
    RefineRecord,
    skipped_arm,
    summarize_refine,
    summarize_trials,
    write_report,
)
from tests.conftest import fail, ok


def test_summarize_refine_uses_none_for_failed_attempts() -> None:
    history = (Step("a", (), ok(1.0)), Step("b", (), fail("compile")), Step("c", (), ok(3.0)))
    rec = summarize_refine(1, RefineOutcome(history, history[2]), "memory text")
    assert rec.speedups == (1.0, None, 3.0) and rec.best_speedup == 3.0 and rec.memory


def test_summarize_refine_without_a_correct_candidate_has_a_note() -> None:
    history = (Step("a", (), fail("compile")),)
    rec = summarize_refine(2, RefineOutcome(history, None), None)
    assert rec.best_speedup is None and "no correct candidate" in rec.note


def test_summarize_trials_first_try_and_later_success() -> None:
    first = TrialOutcome((Step("a", (), ok(2.0)),), EpisodicMemory(), 1)
    later = TrialOutcome(
        (Step("a", (), fail("compile")), Step("b", (), ok(1.5))), EpisodicMemory(), 2
    )
    never = TrialOutcome((Step("a", (), fail("run")),), EpisodicMemory(), None)
    a, b, c = (summarize_trials("C", 1, o) for o in (first, later, never))
    assert a.first_try_success and a.trials_to_success == 1 and a.final_speedup == 2.0
    assert not b.first_try_success and b.trials_to_success == 2 and b.first_failure_stage == "compile"
    assert c.trials_to_success is None and c.final_speedup is None


def test_write_report_creates_all_outputs(tmp_path: Path) -> None:
    refine = [RefineRecord(1, (1.0, 2.0, None, 3.0), 3.0, "use blocking")]
    arms = [
        ArmRecord("B", 1, True, 1, 1.2, None, 1),
        ArmRecord("C", 1, False, 2, 1.1, "compile", 2),
        skipped_arm("C", 2, "no memory"),
    ]
    write_report(tmp_path, Config(), {"amd64": 100.0, "armv7": 900.0}, refine, arms)
    for name in ("results.json", "results.md", "speedup_by_round.png", "README.md"):
        assert (tmp_path / name).exists(), name
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["naive_ms"]["armv7"] == 900.0 and len(data["arms"]) == 3
    md = (tmp_path / "results.md").read_text()
    assert "First-try success" in md and "compile" in md
    assert "A/r1" in (tmp_path / "README.md").read_text()
