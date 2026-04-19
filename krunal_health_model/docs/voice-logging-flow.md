# Voice Logging Flow

## Purpose

Voice logging should make the system low maintenance. Krunal should be able to speak naturally during the day, and the system should convert that into structured food and exercise data.

Wispr Flow or another voice-to-text tool can provide the transcription layer. The health app should handle parsing, clarification, storage, and feedback.

## Food Logging

Example voice input:

```text
For breakfast I had a tofu scramble, two slices of toast, avocado, berries, and water.
```

Parsed output:

```json
{
  "type": "food_log",
  "meal_label": "breakfast",
  "raw_text": "For breakfast I had a tofu scramble, two slices of toast, avocado, berries, and water.",
  "parsed_foods": ["tofu scramble", "2 slices toast", "avocado", "berries", "water"],
  "estimated_nutrition": {
    "calories": null,
    "protein_g": null,
    "carbs_g": null,
    "fat_g": null,
    "fiber_g": null
  },
  "confidence": 0.7,
  "needs_clarification": true,
  "clarifying_question": "Roughly how much tofu was in the scramble?"
}
```

## Exercise Logging

Example voice input:

```text
I rowed two miles at moderate intensity and then lifted chest and triceps for about 50 minutes.
```

Parsed output:

```json
{
  "type": "exercise_log",
  "raw_text": "I rowed two miles at moderate intensity and then lifted chest and triceps for about 50 minutes.",
  "activities": [
    {
      "activity": "rowing",
      "distance_miles": 2,
      "intensity": "moderate"
    },
    {
      "activity": "strength_training",
      "focus": "chest and triceps",
      "duration_minutes": 50
    }
  ],
  "confidence": 0.82
}
```

## Clarification Policy

Ask a clarifying question only when:

- Protein estimate would change meaningfully.
- Calorie estimate would change meaningfully.
- Food violates or may violate the vegan-vegetarian baseline.
- Exercise intensity or volume is unclear enough to affect recovery planning.
- A symptom or injury red flag appears.

Otherwise, store the log with an uncertainty score.

## Daily Check-In

The minimal check-in should be less than one minute.

Suggested prompt:

```text
Energy 1-10, mood 1-10, focus 1-10, digestion 1-10, soreness 1-10. Anything unusual?
```

