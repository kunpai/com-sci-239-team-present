from collections.abc import Mapping

from src.candidate import RunFn
from src.llm import LLMReply
from src.sandbox import RunResult


class FakeLLM:
    """Scripted LLM: returns queued replies and records every prompt."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> LLMReply:
        self.prompts.append(prompt)
        if not self._replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return LLMReply(self._replies.pop(0), "sid", 0.0, 0.0, {}, {})


def code_reply(tag: str, flags: str = "-O3") -> str:
    """A well-formed candidate reply; the first code line is `// <tag>`."""
    return (
        "Here you go.\n```c\n"
        f"// {tag}\n"
        "void matmul(const float *A, const float *B, float *C, int n) {}\n"
        f"```\nFLAGS: {flags}\n"
    )


def ok(speedup: float) -> RunResult:
    return RunResult("ok", True, True, 100.0 / speedup, speedup, "")


def fail(stage: str, msg: str = "boom") -> RunResult:
    return RunResult(stage, stage != "compile", False, None, None, msg)


def scripted_run(table: Mapping[str, RunResult]) -> RunFn:
    """RunFn that looks up the result by the candidate's `// <tag>` first line."""

    def run(code: str, flags: tuple[str, ...]) -> RunResult:
        return table[code.splitlines()[0].removeprefix("// ")]

    return run
