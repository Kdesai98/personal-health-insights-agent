# RL Environment v0

## Purpose

This document defines the first reinforcement-learning-style environment for Krunal's health system.

The initial product should not start with full RL. It should begin as a safe recommendation loop using rules, retrieval, personal history, and simple scoring. Once enough personal data exists, the same structure can support contextual bandits and later RL-style policy learning.

## Environment Summary

```text
state       = Krunal's daily health state
action      = recommended behavior or daily plan
reward      = next-day and longer-term health outcomes
policy      = the strategy that chooses actions
constraints = safety, preferences, medical boundaries, schedule, recovery
```

## State

Core state categories:

- Sleep.
- Readiness and recovery.
- Heart rate and HRV.
- Activity.
- Manual exercise logs.
- Nutrition.
- Meditation.
- Supplements baseline.
- Subjective energy, mood, stress, focus, digestion, and soreness.
- Travel/environment context.
- Labs when available.

## Actions

Possible action categories:

- Training recommendation.
- Recovery recommendation.
- Meal recommendation.
- Protein recommendation.
- Sleep timing recommendation.
- Focus block recommendation.
- Digestion experiment.
- Lab test suggestion to discuss with a clinician.
- Clarifying question.

## Rewards

Same-day reward:

- Adherence.
- Subjective energy.
- Focus.
- Mood.
- Digestion.
- Workout completion.

Next-day reward:

- Sleep quality.
- Readiness.
- Resting heart rate.
- HRV.
- Soreness.
- Energy.
- Digestion.

Longer-term reward:

- Muscle gain.
- Body fat reduction.
- Strength progress.
- Rowing performance.
- Stable high mood.
- Strong focus.
- Lab marker improvements.
- Sustainable adherence.

## Constraints

Hard constraints:

- No diagnosis.
- No medication changes.
- No unverified supplement recommendations.
- No extreme calorie restriction.
- No unsafe training through injury.
- No certainty from weak data.
- Vegan-vegetarian compatibility.

Soft constraints:

- Low maintenance.
- Respect meditation practice.
- Avoid unnecessary logging burden.
- Avoid recommendations that interfere with work or travel.
- Prefer sustainable training over heroic intensity.

## First Algorithm

```text
1. Pull Oura data.
2. Parse voice logs for food and exercise.
3. Build daily health state.
4. Apply safety rules.
5. Generate daily plan.
6. Ask for simple feedback.
7. Score adherence and outcomes.
8. Save the result for future personalization.
```

## Later Algorithm

After 30 to 90 days of clean data:

- Add baseline models for sleep, readiness, energy, digestion, and training response.
- Add contextual bandits for low-risk choices, such as meal timing, workout timing, and bedtime nudges.
- Compare recommendations against outcomes.
- Run small N-of-1 experiments.

After 6 to 12 months of high-quality data:

- Train a personal response model.
- Simulate likely next-day outcomes from candidate actions.
- Use RL-style planning with safety constraints.
- Keep human-readable explanations and audit logs.

