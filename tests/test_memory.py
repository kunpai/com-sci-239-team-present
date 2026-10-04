from pathlib import Path

from src.candidate import Step
from src.config import Config
from src.memory import EpisodicMemory, clip_words, reflect_on_failure, write_memory
from src.traces import TraceScope
from tests.conftest import FakeLLM, fail, ok


def test_window_drops_oldest() -> None:
    mem = EpisodicMemory(window=3)
    for entry in ("a", "b", "c", "d"):
        mem = mem.append(entry)
    assert mem.entries == ("b", "c", "d")


def test_memory_is_immutable_value() -> None:
    base = EpisodicMemory(("x",), 3)
    assert base.append("y").entries == ("x", "y") and base.entries == ("x",)


def test_clip_words() -> None:
    assert clip_words("one two three four", 2) == "one two"
    assert clip_words("  short  ", 5) == "short"


def test_write_memory_retries_once_when_over_cap(tmp_path: Path) -> None:
    cfg = Config(memory_cap_words=5)
    llm = FakeLLM(["one two three four five six seven", "one two three"])
    text = write_memory(llm, "TASK", [Step("c", (), ok(2.0))], cfg, TraceScope(tmp_path, "A", 1))
    assert text == "one two three" and len(llm.prompts) == 2


def test_write_memory_hard_clips_when_retry_is_still_too_long(tmp_path: Path) -> None:
    cfg = Config(memory_cap_words=3)
    llm = FakeLLM(["a b c d e", "a b c d"])
    text = write_memory(llm, "TASK", [Step("c", (), ok(2.0))], cfg, TraceScope(tmp_path, "A", 1))
    assert text == "a b c"


def test_reflect_on_failure_clips_to_lesson_cap(tmp_path: Path) -> None:
    cfg = Config(lesson_cap_words=3)
    llm = FakeLLM(["w1 w2 w3 w4 w5"])
    step = Step("c", ("-O3",), fail("compile", "unknown option"))
    lesson = reflect_on_failure(
        llm, "TASK", step, EpisodicMemory(("old",), 3), cfg, TraceScope(tmp_path, "C", 1)
    )
    assert lesson == "w1 w2 w3" and "old" in llm.prompts[0]
