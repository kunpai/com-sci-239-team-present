"""Tests pinning the final-review fixes (docker errors, cpuset, odd sizes, trace platform)."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from run import run_experiment
from src.config import Config
from src.report import ArmRecord, write_report
from src.sandbox import (
    AMD64,
    ARMV7,
    WORK_ROOT,
    Platform,
    SandboxError,
    _docker_run,
    build_image,
    docker_available,
    run_candidate,
)
from src.traces import TraceWriter
from tests.conftest import FakeLLM, code_reply, ok, scripted_run

CANNED = Path(__file__).parent / "canned"
NAIVE = Path(__file__).parent.parent / "task" / "naive.c"
CFG = Config(n=64, reps=1, cpus=2, compile_timeout_s=120, run_timeout_s=60)


@pytest.fixture(scope="module")
def images() -> None:
    if not docker_available():
        pytest.skip("Docker daemon not available")
    build_image(AMD64)
    build_image(ARMV7)


@pytest.mark.docker
def test_docker_infrastructure_errors_raise_instead_of_blaming_the_candidate(images: None) -> None:
    missing = Platform("amd64", "linux/amd64", "matxfer:does-not-exist")
    with pytest.raises(SandboxError):
        run_candidate(missing, NAIVE.read_text(), ["-O2"], CFG)


@pytest.mark.docker
def test_container_sees_exactly_the_configured_cpus(images: None) -> None:
    WORK_ROOT.mkdir(exist_ok=True)
    rc, out, _err, timed_out = _docker_run(AMD64, WORK_ROOT, ["nproc"], CFG, 60)
    assert (rc, timed_out) == (0, False)
    assert out.strip() == str(CFG.cpus)


@pytest.mark.docker
def test_small_n_fallback_does_not_hide_a_broken_tail(images: None) -> None:
    code = (CANNED / "fallback_notail.c").read_text()
    result = run_candidate(AMD64, code, ["-O2"], CFG)
    assert result.stage == "incorrect"
    assert "odd" in result.stderr


@dataclass(frozen=True)
class _Step:
    code: str


def test_sandbox_trace_records_the_platform(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path, "r")
    path = writer.scope("B", 1, platform="armv7").log_step(_Step("x"), trial=1)
    assert json.loads(path.read_text())["platform"] == "armv7"
    path = writer.scope("A", 1).log_step(_Step("x"))
    assert json.loads(path.read_text())["platform"] is None


def test_experiment_traces_name_the_platform_each_step_ran_on(tmp_path: Path) -> None:
    cfg = Config(rounds=0, repeats=1, trials=1)
    llm = FakeLLM([code_reply("a0"), "memory", code_reply("b1"), code_reply("c1")])
    tables = {AMD64.name: {"a0": ok(2.0)}, ARMV7.name: {"b1": ok(1.0), "c1": ok(1.0)}}
    run_dir = run_experiment(
        cfg, llm, lambda p: scripted_run(tables[p.name]), {"amd64": 1.0, "armv7": 1.0}, tmp_path, "p"
    )
    a = json.loads((run_dir / "A" / "r1" / "02-sandbox.json").read_text())
    c = json.loads((run_dir / "C" / "r1" / "02-sandbox.json").read_text())
    assert (a["platform"], c["platform"]) == ("amd64", "armv7")


def test_sandbox_infrastructure_failure_aborts_only_that_repeat(tmp_path: Path) -> None:
    cfg = Config(rounds=0, repeats=2, trials=1)
    llm = FakeLLM(
        [code_reply("a0"), code_reply("a0"), "memory", code_reply("b1"), code_reply("c1")]
    )
    tables = {AMD64.name: {"a0": ok(2.0)}, ARMV7.name: {"b1": ok(1.0), "c1": ok(1.0)}}
    calls = {"n": 0}

    def run_fn_for(platform: Platform):
        inner = scripted_run(tables[platform.name])

        def run(code: str, flags: tuple[str, ...]):
            calls["n"] += 1
            if calls["n"] == 1:
                raise SandboxError("docker: Error response from daemon")
            return inner(code, flags)

        return run

    run_dir = run_experiment(cfg, llm, run_fn_for, {"amd64": 1.0, "armv7": 1.0}, tmp_path, "i")
    data = json.loads((run_dir / "results.json").read_text())
    assert [r["repeat"] for r in data["refine"]] == [2]
    assert (run_dir / "run" / "r1" / "01-infra_error.json").exists()


def test_mean_trials_does_not_hide_repeats_that_never_succeeded(tmp_path: Path) -> None:
    arms = [
        ArmRecord("C", 1, False, None, None, "compile", 3),
        ArmRecord("C", 2, True, 1, 1.0, None, 1),
        ArmRecord("C", 3, True, 1, 1.0, None, 1),
    ]
    write_report(tmp_path, Config(), {"amd64": 1.0, "armv7": 1.0}, [], arms)
    assert "1.00 (2/3 succeeded)" in (tmp_path / "results.md").read_text()
