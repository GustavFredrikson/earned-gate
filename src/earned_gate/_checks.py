"""Small config validators shared by GateConfig and RolloutPolicy.

A NaN or infinity in a threshold silently disables the comparison it guards
(``nan > 0.15`` is never true), so every numeric setting is checked here and a
violation is a :class:`~earned_gate.errors.ConfigError`.
"""

from __future__ import annotations

import math

from .errors import ConfigError


def check_number(name: str, value: object) -> float:
    """A finite int or float (not bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, not {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError:
        raise ConfigError(f"{name} is too large") from None
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite, not {value!r}")
    return number


def check_int(name: str, value: object) -> int:
    """An int (not bool); floats such as 30.0 or 30.5 are rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an int, not {type(value).__name__}")
    return value
