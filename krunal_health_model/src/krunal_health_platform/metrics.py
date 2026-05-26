"""Metrics registry and alignment monitor."""

from __future__ import annotations

from .models import MetricDefinition, MetricSnapshot, now_iso


class MetricsRegistry:
    def __init__(self) -> None:
        self.metrics: dict[str, MetricDefinition] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        defaults = [
            MetricDefinition(
                metric_id="engagement.daily_active",
                definition="count(app_open_events)",
                tier="engagement",
                ownership="product",
                alarm_thresholds={"min": 0.2},
                consumers=["product_dashboard"],
            ),
            MetricDefinition(
                metric_id="effective_engagement.plan_adherence",
                definition="share(recommended_actions_completed)",
                tier="effective_engagement",
                ownership="health_intelligence",
                alarm_thresholds={"min": 0.35},
                consumers=["research_dashboard", "experiment_readouts"],
            ),
            MetricDefinition(
                metric_id="objective.readiness_next_day",
                definition="next_day_readiness_score",
                tier="objective",
                ownership="health_intelligence",
                alarm_thresholds={"min": 0.5},
                consumers=["research_dashboard"],
            ),
            MetricDefinition(
                metric_id="meta.calibration.readiness_ece",
                definition="expected_calibration_error(readiness_probability)",
                tier="meta",
                ownership="ml_platform",
                alarm_thresholds={"max": 0.08},
                consumers=["model_quality_dashboard"],
            ),
            MetricDefinition(
                metric_id="meta.alignment_beta",
                definition="regression_beta(objective_outcome ~ engagement | baseline_covariates)",
                tier="meta",
                ownership="ml_platform",
                alarm_thresholds={"min": 0.02},
                consumers=["alignment_monitor"],
            ),
        ]
        self.metrics = {metric.metric_id: metric for metric in defaults}


class AlignmentMonitor:
    def evaluate(self, engagement_value: float, outcome_value: float) -> MetricSnapshot:
        alignment_beta = outcome_value - 0.8 * engagement_value
        return MetricSnapshot(
            metric_id="meta.alignment_beta",
            cohort="all_users",
            window="rolling_7d",
            value=round(alignment_beta, 3),
            computed_at=now_iso(),
            metadata={
                "engagement_value": engagement_value,
                "outcome_value": outcome_value,
                "alarm": alignment_beta < 0.02,
            },
        )

