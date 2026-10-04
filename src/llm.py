"""Claude Code (`claude -p`) as the LLM, isolated from the user's config."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Built-in tools off, no MCP servers, no skills, no user/project settings (CLAUDE.md, hooks),
# no session files. Verified on 2026-10-01; `--bare` is NOT used because it cannot authenticate.
_ISOLATION_ARGS = [
    "--output-format", "json",
    "--tools", "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--setting-sources", "",
    "--no-session-persistence",
]  # fmt: skip


class LLMError(RuntimeError):
    """Raised when a Claude Code call fails after retries."""


@dataclass(frozen=True)
class LLMReply:
    """One reply plus the metadata recorded in traces."""

    text: str
    session_id: str
    cost_usd: float
    duration_s: float
    usage: dict[str, Any]
    model_usage: dict[str, Any]


class LLM(Protocol):
    """Anything that turns a prompt into a reply (real or scripted)."""

    def __call__(self, prompt: str) -> LLMReply: ...


def parse_cli_output(stdout: str) -> LLMReply:
    """Parse `claude -p --output-format json` output (one object or a list of events)."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise LLMError(f"claude output is not JSON: {stdout[:200]!r}") from exc
    items = data if isinstance(data, list) else [data]
    results = [i for i in items if isinstance(i, dict) and i.get("type") == "result"]
    if not results:
        raise LLMError("claude output has no result object")
    result = results[-1]
    if result.get("is_error"):
        raise LLMError(f"claude reported an error: {result.get('result')!r}")
    return LLMReply(
        text=str(result.get("result", "")),
        session_id=str(result.get("session_id", "")),
        cost_usd=float(result.get("total_cost_usd") or 0.0),
        duration_s=float(result.get("duration_ms") or 0) / 1000.0,
        usage=dict(result.get("usage") or {}),
        model_usage=dict(result.get("modelUsage") or {}),
    )


class ClaudeCodeLLM:
    """Calls `claude -p` once per prompt; on failure waits `backoff_s[i]` seconds and retries."""

    def __init__(
        self,
        timeout_s: int,
        model: str | None = None,
        backoff_s: Sequence[float] = (30, 60, 120),
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._timeout_s = timeout_s
        self._model = model
        self._backoff_s = tuple(backoff_s)
        self._sleep = sleep
        self._cwd = Path(tempfile.mkdtemp(prefix="matxfer-llm-"))  # empty: no CLAUDE.md to find

    def _command(self, prompt: str) -> list[str]:
        cmd = ["claude", "-p", prompt, *_ISOLATION_ARGS]
        if self._model:
            cmd += ["--model", self._model]
        return cmd

    def __call__(self, prompt: str) -> LLMReply:
        last: Exception | None = None
        attempts = 1 + len(self._backoff_s)
        for attempt in range(attempts):
            try:
                proc = subprocess.run(
                    self._command(prompt),
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=self._timeout_s,
                    stdin=subprocess.DEVNULL,
                    cwd=self._cwd,
                    check=False,
                )
                if proc.returncode != 0:
                    raise LLMError(f"claude exited {proc.returncode}: {proc.stderr[:300]!r}")
                return parse_cli_output(proc.stdout)
            except (subprocess.TimeoutExpired, OSError, LLMError) as exc:
                last = exc
                logger.warning("claude call failed (attempt %d/%d): %s", attempt + 1, attempts, exc)
                if attempt < len(self._backoff_s):
                    (self._sleep or time.sleep)(self._backoff_s[attempt])
        raise LLMError(f"claude failed after {attempts} attempts: {last}") from last
