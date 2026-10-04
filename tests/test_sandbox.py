import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from src.config import Config
from src.sandbox import (
    AMD64,
    ARMV7,
    FlagError,
    build_image,
    check_flags,
    docker_available,
    measure_naive,
    run_candidate,
)

CANNED = Path(__file__).parent / "canned"
NAIVE = Path(__file__).parent.parent / "task" / "naive.c"
CFG = Config(n=64, reps=1, cpus=2, compile_timeout_s=120, run_timeout_s=60)


def test_check_flags_accepts_normal_flags() -> None:
    check_flags(["-O3", "-march=native", "-fopenmp", "-ffast-math", "-funroll-loops"])


@pytest.mark.parametrize(
    "bad", [["-o", "x"], ["-wrapper", "sh"], ["-O3;rm"], ["-lfoo"], ["-I/etc"], ["-Wl,-z"], ["@f"]]
)
def test_check_flags_rejects_dangerous_flags(bad: list[str]) -> None:
    with pytest.raises(FlagError):
        check_flags(bad)


@pytest.fixture(scope="module")
def images() -> None:
    if not docker_available():
        pytest.skip("Docker daemon not available")
    build_image(AMD64)
    build_image(ARMV7)


@pytest.mark.docker
@pytest.mark.parametrize("platform", [AMD64, ARMV7])
def test_naive_is_correct_on_both_platforms(images: None, platform) -> None:
    result = run_candidate(platform, NAIVE.read_text(), ["-O2"], CFG, naive_ms=100.0)
    assert result.stage == "ok" and result.success
    assert result.time_ms is not None and result.speedup is not None


@pytest.mark.docker
def test_wrong_result_is_incorrect(images: None) -> None:
    result = run_candidate(AMD64, (CANNED / "wrong.c").read_text(), ["-O2"], CFG)
    assert result.stage == "incorrect" and result.compiled and not result.correct


@pytest.mark.docker
def test_accumulating_into_c_is_caught(images: None) -> None:
    result = run_candidate(AMD64, (CANNED / "accum.c").read_text(), ["-O2"], CFG)
    assert result.stage == "incorrect"


@pytest.mark.docker
def test_candidate_correct_only_for_multiples_of_four_is_caught(images: None) -> None:
    result = run_candidate(AMD64, (CANNED / "notail.c").read_text(), ["-O2"], CFG)
    assert result.stage == "incorrect"
    assert "37" in result.stderr


@pytest.mark.docker
def test_infinite_loop_times_out_and_leaves_no_container(images: None) -> None:
    cfg = replace(CFG, run_timeout_s=5)
    result = run_candidate(AMD64, (CANNED / "loop.c").read_text(), ["-O2"], cfg)
    assert result.stage == "timeout" and result.compiled
    leftover = subprocess.run(
        ["docker", "ps", "-q", "--filter", "name=matxfer-"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert leftover.stdout.strip() == ""


@pytest.mark.docker
def test_avx2_compiles_on_amd64_and_fails_to_compile_on_arm32(images: None) -> None:
    code = (CANNED / "avx2.c").read_text()
    flags = ["-O3", "-mavx2", "-mfma"]
    on_x86 = run_candidate(AMD64, code, flags, CFG)
    on_arm = run_candidate(ARMV7, code, flags, CFG)
    assert on_x86.compiled
    assert on_arm.stage == "compile" and not on_arm.compiled
    assert on_arm.stderr  # the compiler error is what the Reflexion step reads


@pytest.mark.docker
def test_rejected_flags_never_reach_docker(images: None) -> None:
    result = run_candidate(AMD64, NAIVE.read_text(), ["-o", "evil"], CFG)
    assert result.stage == "flags"


@pytest.mark.docker
def test_measure_naive_returns_a_time(images: None) -> None:
    result = measure_naive(AMD64, CFG)
    assert result.success and result.time_ms and result.time_ms > 0
