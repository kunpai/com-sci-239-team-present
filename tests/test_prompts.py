import re

from src.candidate import Step
from src.prompts import (
    failure_lesson_prompt,
    feedback_prompt,
    format_observation,
    generation_prompt,
    load_task,
    refine_prompt,
    reflection_prompt,
    shorten_prompt,
)
from tests.conftest import fail, ok

ARCH_WORDS = ("x86", "amd64", "arm", "avx", "neon", "sse", "32-bit", "64-bit")


def test_task_statement_never_names_the_architecture() -> None:
    text = load_task().lower()
    leaked = [w for w in ARCH_WORDS if re.search(rf"\b{re.escape(w)}\b", text)]
    assert leaked == []


def test_task_statement_says_the_code_runs_in_a_container_not_on_the_models_machine() -> None:
    # Claude Code always tells the model the host OS (macOS); this sentence corrects that
    # without naming any CPU architecture.
    assert "inside a Linux container" in load_task()


def test_generation_prompt_memory_block_is_the_only_difference() -> None:
    task = load_task()
    assert generation_prompt(task, ()) == task
    with_memory = generation_prompt(task, ["use blocking"])
    assert with_memory.startswith(task) and "use blocking" in with_memory


def test_format_observation_ok_and_failures() -> None:
    assert "3.00x" in format_observation(ok(3.0))
    assert "Compilation failed" in format_observation(fail("compile", "bad flag"))
    assert "bad flag" in format_observation(fail("compile", "bad flag"))


def test_history_prompts_carry_code_observations_and_feedback() -> None:
    history = [Step("// v0\nint a;", ("-O2",), ok(1.5), "tile the loops"), Step("// v1", (), ok(2.0))]
    fb = feedback_prompt("TASK", history)
    assert "// v0" in fb and "1.50x" in fb and "tile the loops" in fb
    rf = refine_prompt("TASK", history)
    assert "Feedback" in rf and "FLAGS" in rf


def test_reflection_and_lesson_prompts_state_the_word_cap() -> None:
    step = Step("// v0", ("-O3",), fail("compile", "unknown option"))
    assert "200 words" in reflection_prompt("TASK", [step], 200)
    lesson = failure_lesson_prompt("TASK", step, ["old note"], 80)
    assert "80 words" in lesson and "unknown option" in lesson and "old note" in lesson
    assert "50 words" in shorten_prompt("long text", 50)
