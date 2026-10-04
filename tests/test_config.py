from dataclasses import FrozenInstanceError

import pytest

from src.config import Config


def test_defaults_match_spec() -> None:
    cfg = Config()
    assert (cfg.rounds, cfg.repeats, cfg.trials, cfg.window) == (10, 3, 3, 3)
    assert cfg.memory_cap_words == 200
    assert cfg.n == 1024


def test_config_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        Config().rounds = 5  # type: ignore[misc]


def test_quick_shrinks_rounds_and_repeats_only() -> None:
    cfg = Config().quick()
    assert (cfg.rounds, cfg.repeats) == (3, 1)
    assert cfg.n == Config().n
