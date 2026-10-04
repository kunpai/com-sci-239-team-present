import json
from dataclasses import dataclass
from pathlib import Path

from src.traces import TraceWriter


@dataclass(frozen=True)
class _Reply:
    text: str
    cost_usd: float


@dataclass(frozen=True)
class _Step:
    code: str


def test_scope_writes_numbered_files_with_context(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path, "run1")
    scope = writer.scope("C", 2)
    scope.set_memory(["note one"])
    p1 = scope.log("llm", role="generate")
    p2 = scope.log("sandbox", compiled=False)
    assert p1.name == "01-llm.json" and p2.name == "02-sandbox.json"
    assert p1.parent == tmp_path / "run1" / "C" / "r2"
    data = json.loads(p2.read_text())
    assert data["arm"] == "C" and data["repeat"] == 2 and data["step"] == 2
    assert data["memory"] == ["note one"] and data["compiled"] is False


def test_log_llm_and_log_step_flatten_dataclasses(tmp_path: Path) -> None:
    scope = TraceWriter(tmp_path, "run1").scope("A", 1)
    llm_path = scope.log_llm("feedback", "PROMPT", _Reply("hi", 0.01), round_no=3)
    step_path = scope.log_step(_Step("int x;"), trial=1)
    llm = json.loads(llm_path.read_text())
    assert llm["role"] == "feedback" and llm["prompt"] == "PROMPT"
    assert llm["text"] == "hi" and llm["round_no"] == 3
    assert json.loads(step_path.read_text())["code"] == "int x;"
