"""Prediction calibration utilities for personal health models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ml_utils import clamp
from .models import jsonable, now_iso


@dataclass
class CalibrationBin:
    low: float
    high: float
    n: int
    predicted_mean: float | None
    observed_mean: float | None


@dataclass
class ReliabilityReport:
    model_name: str
    target_name: str
    bins: list[CalibrationBin]
    expected_calibration_error: float
    n: int
    generated_at: str = field(default_factory=now_iso)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class ProbabilityCalibrator:
    """Piecewise reliability calibrator for bounded predictions."""

    def __init__(self, model_name: str, target_name: str, n_bins: int = 10) -> None:
        self.model_name = model_name
        self.target_name = target_name
        self.n_bins = n_bins
        self._bin_observed: list[float | None] = [None for _ in range(n_bins)]
        self._report: ReliabilityReport | None = None

    def fit(self, predictions: list[float], outcomes: list[float]) -> ReliabilityReport:
        if len(predictions) != len(outcomes):
            raise ValueError("predictions and outcomes must have equal length")
        buckets: list[list[tuple[float, float]]] = [[] for _ in range(self.n_bins)]
        for pred, outcome in zip(predictions, outcomes):
            idx = min(self.n_bins - 1, int(clamp(pred) * self.n_bins))
            buckets[idx].append((clamp(pred), clamp(outcome)))

        bins = []
        weighted_error = 0.0
        total = max(len(predictions), 1)
        for idx, bucket in enumerate(buckets):
            low = idx / self.n_bins
            high = (idx + 1) / self.n_bins
            if bucket:
                pred_mean = sum(item[0] for item in bucket) / len(bucket)
                obs_mean = sum(item[1] for item in bucket) / len(bucket)
                self._bin_observed[idx] = obs_mean
                weighted_error += (len(bucket) / total) * abs(pred_mean - obs_mean)
                bins.append(
                    CalibrationBin(
                        low=round(low, 3),
                        high=round(high, 3),
                        n=len(bucket),
                        predicted_mean=round(pred_mean, 3),
                        observed_mean=round(obs_mean, 3),
                    )
                )
            else:
                bins.append(
                    CalibrationBin(
                        low=round(low, 3),
                        high=round(high, 3),
                        n=0,
                        predicted_mean=None,
                        observed_mean=None,
                    )
                )

        notes = []
        if len(predictions) < 30:
            notes.append("low_sample_calibration_report")
        self._report = ReliabilityReport(
            model_name=self.model_name,
            target_name=self.target_name,
            bins=bins,
            expected_calibration_error=round(weighted_error, 4),
            n=len(predictions),
            notes=notes,
        )
        return self._report

    def calibrate(self, probability: float) -> float:
        idx = min(self.n_bins - 1, int(clamp(probability) * self.n_bins))
        observed = self._bin_observed[idx]
        if observed is None:
            return clamp(probability)
        return round(0.65 * clamp(probability) + 0.35 * observed, 3)

    @property
    def report(self) -> ReliabilityReport | None:
        return self._report
