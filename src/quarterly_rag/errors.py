"""Exceptions shared across layers."""

from __future__ import annotations


class ModelServerError(RuntimeError):
    """A model endpoint could not be reached or answered with an error.

    Every provider raises this one type so callers (doctor, generation, evaluation)
    handle failures the same way regardless of the wire protocol behind them.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
