"""The viewer template and the real traces must keep working together."""

from pathlib import Path

import pytest

from viewer.build import PLACEHOLDER, build_data, render_html

ROOT = Path(__file__).parent.parent
TEMPLATE = ROOT / "viewer" / "template.html"


def test_template_has_exactly_one_data_placeholder_and_two_scripts() -> None:
    text = TEMPLATE.read_text()
    assert text.count(PLACEHOLDER) == 1
    assert text.count("<script") == 2 and text.count("</script>") == 2


def test_real_traces_build_under_the_size_budget() -> None:
    traces = ROOT / "traces"
    if not traces.is_dir() or not any(traces.iterdir()):
        pytest.skip("no real traces in this checkout")
    html = render_html(build_data(traces), TEMPLATE.read_text())
    assert len(html.encode()) < 25_000_000
    assert html.count("</script>") == 2  # trace text can never add a closing tag
