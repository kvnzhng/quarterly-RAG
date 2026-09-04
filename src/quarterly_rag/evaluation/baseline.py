"""The regression gate: numbers that must not get worse (RAG-012).

`make eval` runs the evals and compares them with a committed baseline. The point is not
to freeze the numbers, which would block every improvement, but to make a drop a decision
somebody took rather than something that happened.

The tolerance is not cosmetic. The eval set holds 33 answerable questions, so one question
is three points; a tolerance tighter than that fails on noise, and a reader who sees a red
build for noise stops reading red builds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from quarterly_rag.config import Settings

BASELINE_FILE = "baseline.json"
DEFAULT_TOLERANCE = 0.05
"""Five points. One question in 33 is three, so anything tighter fails on noise."""


@dataclass(frozen=True)
class Comparison:
    metric: str
    baseline: float
    current: float
    tolerance: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    @property
    def regressed(self) -> bool:
        return self.delta < -self.tolerance

    @property
    def improved(self) -> bool:
        return self.delta > self.tolerance


@dataclass
class BaselineCheck:
    comparisons: list[Comparison] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    """Metrics the baseline names that this run did not produce. Treated as a failure:
    a gate that silently skips what it cannot find is not a gate."""

    @property
    def regressions(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.regressed]

    @property
    def passed(self) -> bool:
        return not self.regressions and not self.missing


def baseline_path(settings: Settings) -> Path:
    return settings.eval_dir / BASELINE_FILE


def save_baseline(
    settings: Settings,
    metrics: dict[str, float],
    run_record: dict[str, object],
    tolerance: float = DEFAULT_TOLERANCE,
) -> Path:
    """Overwrite the baseline. The deliberate 'I accept these numbers' action."""
    path = baseline_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tolerance": tolerance,
                "metrics": {k: round(v, 4) for k, v in sorted(metrics.items())},
                "run_record": run_record,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def load_baseline(settings: Settings) -> dict | None:
    path = baseline_path(settings)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def compare(baseline: dict, metrics: dict[str, float]) -> BaselineCheck:
    tolerance = float(baseline.get("tolerance", DEFAULT_TOLERANCE))
    check = BaselineCheck()
    for metric, previous in sorted(baseline.get("metrics", {}).items()):
        if metric not in metrics:
            check.missing.append(metric)
            continue
        check.comparisons.append(
            Comparison(metric, float(previous), float(metrics[metric]), tolerance)
        )
    return check
