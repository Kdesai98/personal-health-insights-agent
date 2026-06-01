"""Lab ingestion and non-diagnostic interpretation boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import jsonable, now_iso


LAB_REFERENCE_RANGES = {
    "apoB": {"low": None, "high": 90.0, "unit": "mg/dL", "category": "cardiometabolic"},
    "ldl": {"low": None, "high": 100.0, "unit": "mg/dL", "category": "cardiometabolic"},
    "hdl": {"low": 40.0, "high": None, "unit": "mg/dL", "category": "cardiometabolic"},
    "triglycerides": {"low": None, "high": 150.0, "unit": "mg/dL", "category": "cardiometabolic"},
    "hba1c": {"low": None, "high": 5.7, "unit": "%", "category": "glucose"},
    "vitamin_d": {"low": 30.0, "high": 100.0, "unit": "ng/mL", "category": "micronutrient"},
    "b12": {"low": 300.0, "high": None, "unit": "pg/mL", "category": "micronutrient"},
    "ferritin": {"low": 30.0, "high": 300.0, "unit": "ng/mL", "category": "micronutrient"},
    "testosterone_total": {"low": 300.0, "high": 1000.0, "unit": "ng/dL", "category": "hormone"},
    "hs_crp": {"low": None, "high": 1.0, "unit": "mg/L", "category": "inflammation"},
}


ALIASES = {
    "apob": "apoB",
    "apo b": "apoB",
    "ldl": "ldl",
    "hdl": "hdl",
    "triglycerides": "triglycerides",
    "a1c": "hba1c",
    "hba1c": "hba1c",
    "vitamin d": "vitamin_d",
    "25-oh vitamin d": "vitamin_d",
    "b12": "b12",
    "ferritin": "ferritin",
    "total testosterone": "testosterone_total",
    "testosterone": "testosterone_total",
    "hs-crp": "hs_crp",
    "crp": "hs_crp",
}


@dataclass
class LabObservation:
    marker: str
    value: float
    unit: str | None
    category: str
    reference_status: str
    source_text: str


@dataclass
class LabIngestionResult:
    observations: list[LabObservation]
    clinician_discussion_flags: list[str]
    unsupported_lines: list[str]
    boundary_notice: str
    parsed_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class LabIngestionBoundary:
    """Parses lab text while preventing diagnosis or treatment recommendations."""

    def parse_text(self, text: str) -> LabIngestionResult:
        observations = []
        unsupported_lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            marker = self._marker_from_line(line)
            value = self._value_from_line(line)
            if not marker or value is None:
                unsupported_lines.append(line)
                continue
            reference = LAB_REFERENCE_RANGES[marker]
            unit_match = re.search(r"(mg/dl|ng/ml|pg/ml|mg/l|%)", line, flags=re.IGNORECASE)
            unit = unit_match.group(1) if unit_match else reference["unit"]
            observations.append(
                LabObservation(
                    marker=marker,
                    value=value,
                    unit=unit,
                    category=str(reference["category"]),
                    reference_status=self._reference_status(marker, value),
                    source_text=line,
                )
            )

        flags = [
            f"{obs.marker} is {obs.reference_status}; discuss context with a clinician."
            for obs in observations
            if obs.reference_status in {"below_reference", "above_reference"}
        ]
        return LabIngestionResult(
            observations=observations,
            clinician_discussion_flags=flags,
            unsupported_lines=unsupported_lines,
            boundary_notice=(
                "Lab parsing is for organization and clinician-discussion prompts only; "
                "it does not diagnose, prescribe, or change treatment."
            ),
        )

    def _marker_from_line(self, line: str) -> str | None:
        normalized = line.lower().replace("_", " ")
        for alias, marker in ALIASES.items():
            if alias in normalized:
                return marker
        return None

    def _value_from_line(self, line: str) -> float | None:
        matches = re.findall(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)", line)
        if not matches:
            return None
        return float(matches[-1])

    def _reference_status(self, marker: str, value: float) -> str:
        reference = LAB_REFERENCE_RANGES[marker]
        low = reference["low"]
        high = reference["high"]
        if low is not None and value < low:
            return "below_reference"
        if high is not None and value > high:
            return "above_reference"
        return "within_reference"
