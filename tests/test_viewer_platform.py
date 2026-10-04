"""Runs written before the `platform` trace field existed are flagged, not silently inferred."""

import json
from pathlib import Path

from tests.test_viewer_build import _make_run
from viewer.build import load_run


def test_current_traces_are_not_flagged(tmp_path: Path) -> None:
    assert load_run(_make_run(tmp_path))["platform_inferred"] is False


def test_traces_without_the_platform_field_are_flagged(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    for path in run_dir.rglob("*-sandbox.json"):
        data = json.loads(path.read_text())
        data.pop("platform", None)
        path.write_text(json.dumps(data))
    run = load_run(run_dir)
    assert run["platform_inferred"] is True
    assert run["repeats"][0]["B"]["trials"][0]["platform"] == "armv7"  # still inferred
