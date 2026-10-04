import json
import shutil
import subprocess

import pytest

from src.llm import ClaudeCodeLLM, LLMError, parse_cli_output

RESULT = {
    "type": "result",
    "result": "OK",
    "session_id": "s1",
    "total_cost_usd": 0.01,
    "duration_ms": 1200,
    "usage": {"input_tokens": 5},
    "modelUsage": {"some-model": {}},
}


def test_parse_single_object() -> None:
    reply = parse_cli_output(json.dumps(RESULT))
    assert reply.text == "OK" and reply.session_id == "s1"
    assert reply.cost_usd == 0.01 and reply.duration_s == 1.2


def test_parse_event_list_takes_last_result() -> None:
    events = [{"type": "system", "subtype": "init"}, RESULT]
    assert parse_cli_output(json.dumps(events)).text == "OK"


def test_parse_error_result_raises() -> None:
    with pytest.raises(LLMError):
        parse_cli_output(json.dumps({**RESULT, "is_error": True, "result": "Not logged in"}))


def test_parse_garbage_raises() -> None:
    with pytest.raises(LLMError):
        parse_cli_output("not json")


def _completed(stdout: str, rc: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], rc, stdout=stdout, stderr="")


def test_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd, 1)
        return _completed(json.dumps(RESULT))

    monkeypatch.setattr("src.llm.subprocess.run", fake_run)
    reply = ClaudeCodeLLM(timeout_s=5, sleep=lambda s: None)("hello")
    assert reply.text == "OK" and len(calls) == 2


def test_gives_up_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.llm.subprocess.run", lambda cmd, **kw: _completed("", rc=1))
    with pytest.raises(LLMError):
        ClaudeCodeLLM(timeout_s=5, sleep=lambda s: None)("hello")


def test_command_is_isolated_from_user_config(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return _completed(json.dumps(RESULT))

    monkeypatch.setattr("src.llm.subprocess.run", fake_run)
    ClaudeCodeLLM(timeout_s=5, model="some-model")("hello")
    cmd = seen[0]
    assert cmd[:3] == ["claude", "-p", "hello"]
    assert cmd[cmd.index("--tools") + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "some-model"


@pytest.mark.llm
@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not installed")
def test_real_claude_smoke() -> None:
    reply = ClaudeCodeLLM(timeout_s=120)("Reply with exactly: OK")
    assert "OK" in reply.text and reply.session_id
