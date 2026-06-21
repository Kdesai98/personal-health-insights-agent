"""Voice-log nutrition parsing with uncertainty."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import jsonable, now_iso


FOOD_MACROS_PER_SERVING = {
    "tofu": {"calories": 180, "protein_g": 18, "carbs_g": 4, "fat_g": 11, "fiber_g": 2},
    "tempeh": {"calories": 195, "protein_g": 20, "carbs_g": 8, "fat_g": 11, "fiber_g": 5},
    "seitan": {"calories": 160, "protein_g": 25, "carbs_g": 8, "fat_g": 2, "fiber_g": 1},
    "lentils": {"calories": 230, "protein_g": 18, "carbs_g": 40, "fat_g": 1, "fiber_g": 16},
    "beans": {"calories": 225, "protein_g": 15, "carbs_g": 40, "fat_g": 1, "fiber_g": 15},
    "chickpeas": {"calories": 270, "protein_g": 15, "carbs_g": 45, "fat_g": 4, "fiber_g": 13},
    "rice": {"calories": 205, "protein_g": 4, "carbs_g": 45, "fat_g": 0, "fiber_g": 1},
    "quinoa": {"calories": 222, "protein_g": 8, "carbs_g": 39, "fat_g": 4, "fiber_g": 5},
    "oats": {"calories": 150, "protein_g": 5, "carbs_g": 27, "fat_g": 3, "fiber_g": 4},
    "avocado": {"calories": 240, "protein_g": 3, "carbs_g": 13, "fat_g": 22, "fiber_g": 10},
    "banana": {"calories": 105, "protein_g": 1, "carbs_g": 27, "fat_g": 0, "fiber_g": 3},
    "blueberries": {"calories": 85, "protein_g": 1, "carbs_g": 21, "fat_g": 0, "fiber_g": 4},
    "granola": {"calories": 220, "protein_g": 5, "carbs_g": 35, "fat_g": 7, "fiber_g": 4},
    "yogurt": {"calories": 160, "protein_g": 17, "carbs_g": 9, "fat_g": 6, "fiber_g": 0},
    "protein powder": {"calories": 120, "protein_g": 24, "carbs_g": 3, "fat_g": 2, "fiber_g": 1},
    "protein shake": {"calories": 220, "protein_g": 30, "carbs_g": 12, "fat_g": 5, "fiber_g": 2},
    "eggs": {"calories": 70, "protein_g": 6, "carbs_g": 0, "fat_g": 5, "fiber_g": 0},
}


QUANTITY_WORDS = {
    "one": 1.0,
    "a": 1.0,
    "an": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "half": 0.5,
}


@dataclass
class ParsedFoodItem:
    name: str
    quantity: float
    matched_keyword: str
    confidence: float
    estimated_nutrition: dict[str, float]


@dataclass
class NutritionParseResult:
    raw_text: str
    parsed_foods: list[ParsedFoodItem]
    estimated_nutrition: dict[str, float]
    confidence: float
    clarifying_questions: list[str]
    parsed_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class VoiceNutritionParser:
    """A deterministic parser that becomes the boundary for future LLM parsing."""

    def parse(self, raw_text: str) -> NutritionParseResult:
        normalized = raw_text.lower()
        items = []
        for keyword, macros in FOOD_MACROS_PER_SERVING.items():
            if keyword not in normalized:
                continue
            quantity = self._quantity_before_keyword(normalized, keyword)
            estimated = {
                key: round(float(value) * quantity, 1)
                for key, value in macros.items()
            }
            items.append(
                ParsedFoodItem(
                    name=keyword,
                    quantity=quantity,
                    matched_keyword=keyword,
                    confidence=0.78 if quantity != 1.0 else 0.68,
                    estimated_nutrition=estimated,
                )
            )

        totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
        for item in items:
            for key in totals:
                totals[key] += item.estimated_nutrition.get(key, 0.0)
        totals = {key: round(value, 1) for key, value in totals.items()}

        confidence = 0.0
        if items:
            confidence = round(sum(item.confidence for item in items) / len(items), 2)
            if len(items) == 1 and len(normalized.split()) > 10:
                confidence = min(confidence, 0.58)

        questions = []
        if not items:
            questions.append("What were the main protein and carb sources in that meal?")
        elif confidence < 0.72:
            questions.append("Was that portion small, typical, or large?")
        if "bowl" in normalized and confidence < 0.80:
            questions.append("Was the bowl mostly grains, vegetables, or protein?")

        return NutritionParseResult(
            raw_text=raw_text,
            parsed_foods=items,
            estimated_nutrition=totals,
            confidence=confidence,
            clarifying_questions=questions,
        )

    def _quantity_before_keyword(self, text: str, keyword: str) -> float:
        pattern = rf"(?:(\d+(?:\.\d+)?)|({'|'.join(QUANTITY_WORDS)}))\s+(?:servings? of |cups? of |pieces? of |scoops? of |large |small )?{re.escape(keyword)}"
        match = re.search(pattern, text)
        if not match:
            return 1.0
        if match.group(1):
            return float(match.group(1))
        return QUANTITY_WORDS.get(match.group(2), 1.0)
