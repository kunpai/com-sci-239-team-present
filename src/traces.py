"""Per-step JSON trace files."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


class TraceScope:
    """Writes numbered JSON files into one (arm, repeat) directory."""

    def __init__(
        self, directory: Path, arm: str, repeat: int, platform: str | None = None
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._arm = arm
        self._repeat = repeat
        self._platform = platform
        self._step = 0
        self._memory: list[str] = []

    def set_memory(self, entries: Sequence[str]) -> None:
        """Record the memory entries in effect for subsequent steps."""
        self._memory = list(entries)

    def log(self, kind: str, **fields: Any) -> Path:
        """Write the next numbered trace file and return its path."""
        self._step += 1
        payload: dict[str, Any] = {
            "arm": self._arm,
            "repeat": self._repeat,
            "step": self._step,
            "kind": kind,
            "memory": list(self._memory),
            **fields,
        }
        path = self.directory / f"{self._step:02d}-{kind}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def log_llm(self, role: str, prompt: str, reply: Any, **ctx: Any) -> Path:
        """Log one LLM call; `reply` is a dataclass (LLMReply)."""
        return self.log("llm", role=role, prompt=prompt, **asdict(reply), **ctx)

    def log_step(self, step: Any, **ctx: Any) -> Path:
        """Log one sandbox evaluation; `step` is a dataclass (Step)."""
        return self.log("sandbox", **asdict(step), platform=self._platform, **ctx)


class TraceWriter:
    """Owns `<root>/<run_id>/` and hands out per-arm scopes."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def scope(self, arm: str, repeat: int, platform: str | None = None) -> TraceScope:
        """Return the scope for `<run_dir>/<arm>/r<repeat>/`; `platform` tags sandbox steps."""
        return TraceScope(self.run_dir / arm / f"r{repeat}", arm, repeat, platform)
