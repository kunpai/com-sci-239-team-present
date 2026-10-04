import pytest

from src.candidate import ParseError, Step, best_step, evaluate_reply, parse_candidate
from tests.conftest import code_reply, fail, ok, scripted_run


def test_parse_basic() -> None:
    cand = parse_candidate(code_reply("v0", "-O3 -fopenmp"))
    assert "matmul" in cand.code and cand.flags == ("-O3", "-fopenmp")


def test_parse_takes_last_matmul_block_and_last_flags_line() -> None:
    text = (
        "```c\nint helper(void) { return 1; }\n```\nthinking...\n"
        "```c\n// FLAGS: -O0\nvoid matmul(const float *A, const float *B, float *C, int n) {}\n```\n"
        "FLAGS: -O2\nFLAGS: -O3 -fopenmp\n"
    )
    cand = parse_candidate(text)
    assert "helper" not in cand.code
    assert cand.flags == ("-O3", "-fopenmp")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no code here\nFLAGS: -O3",
        "```c\nint helper(void) { return 1; }\n```\nFLAGS: -O3",
        "```c\nvoid matmul(){}\n```\nno flags line",
        '```c\nvoid matmul(){}\n```\nFLAGS: -O3 "unterminated',
    ],
)
def test_parse_rejects_bad_replies(text: str) -> None:
    with pytest.raises(ParseError):
        parse_candidate(text)


def test_evaluate_reply_turns_parse_errors_into_failed_steps() -> None:
    step = evaluate_reply("garbage", scripted_run({}))
    assert step.result.stage == "parse" and not step.result.success
    assert step.result.stderr


def test_evaluate_reply_runs_good_candidates() -> None:
    step = evaluate_reply(code_reply("v0"), scripted_run({"v0": ok(2.0)}))
    assert step.result.success and step.flags == ("-O3",)


def test_best_step_is_fastest_correct_not_last() -> None:
    steps = [
        Step("a", (), ok(1.0)),
        Step("b", (), ok(3.0)),
        Step("c", (), fail("incorrect")),
        Step("d", (), ok(2.0)),
    ]
    assert best_step(steps) is steps[1]
    assert best_step([Step("x", (), fail("compile"))]) is None
