# Krunal Health Data Contract

## Purpose

This file defines the first personal data contract for Krunal's health model.

The system should be low maintenance. Oura should provide automatic wearable data. Manual logging should focus on what Oura does not capture well: diet, rowing, lifting context, unusual recovery factors, and subjective state.

## System Role

The system is a personal health thought partner. It may help Krunal reason about health, plan the day, notice patterns, and run low-risk personal experiments.

It is not a doctor, dietician, therapist, or emergency service. It must not diagnose, prescribe, change medication, or override licensed clinicians.

## Daily Knowledge

For each day, the system should know or estimate:

- Sleep duration, timing, quality, and regularity.
- Resting heart rate.
- Heart rate trends.
- HRV and readiness-style recovery signals when available.
- Calories burned or activity energy from Oura.
- Steps from Oura.
- Exercise manually logged by Krunal, especially rowing and lifting.
- Food manually logged through voice-to-text.
- Approximate protein, calories, carbs, fat, fiber, and meal timing.
- Meditation completion in morning and night.
- Supplements that are part of the declared baseline.
- Subjective energy, mood, stress, soreness, digestion, focus, and mental stability when logged.
- Travel or environment changes when relevant.
- Latest lab results once connected or manually entered.

## Data Sources

### Oura

Oura is the primary wearable source.

Expected Oura data:

- Sleep data.
- Resting heart rate.
- Heart rate data.
- HRV and readiness metrics when available.
- Activity data.
- Steps.
- Calories burned.
- Daily summary metrics available through the Oura API.

### Voice Logs

Manual logs should be captured with voice-to-text, likely through Wispr Flow or a similar dictation workflow.

Manual logs should include:

- What Krunal ate.
- Rowing sessions.
- Lifting workouts, including major movements, sets, reps, weight, and perceived effort when convenient.
- Subjective energy, mood, stress, digestion, focus, soreness, and recovery.
- Travel, illness, poor sleep causes, or unusual schedule disruptions.

### Labs

Labs are not the first integration, but the data model should support them.

Useful lab categories:

- Lipids, including LDL, HDL, triglycerides, and ApoB if available.
- Glucose health, including fasting glucose, fasting insulin, and HbA1c if available.
- Inflammation markers, including hs-CRP if available.
- Vitamin and mineral markers, including vitamin D, B12, ferritin, and magnesium if available.
- Hormones, including testosterone if available.
- Thyroid markers if available.

The system may suggest labs to discuss with a clinician, but must not diagnose based on labs.

## Declared Baseline Assumptions

These are user-declared defaults, not model-discovered facts.

- Krunal uses Oura as the primary wearable.
- Krunal follows a vegan-vegetarian diet. Clarify later whether dairy, eggs, honey, or other vegetarian-but-not-vegan foods are included.
- Krunal generally lives in high air quality environments.
- Krunal may travel to India once or twice per year, where air quality may materially change.
- Krunal exercises six to seven days per week depending on schedule and recovery.
- Krunal lifts weights.
- Krunal rows, and rowing may be misrepresented or underrepresented by wearable step-based activity metrics.
- Krunal meditates twice daily, morning and night.
- Krunal practices religious meditation/Vipassana.
- Krunal does not consume caffeine.
- Hydration is generally good.
- Krunal takes a daily vegan multivitamin.
- Krunal takes ashwagandha at night.
- Krunal takes saffron in the morning.
- Krunal does not generally buy new supplements and does not want unverified supplement recommendations.

## Inference Policy

Inference is useful in health, but it must be transparent, confidence-scored, and reversible.

Allowed inference examples:

- Recovery status from sleep, HRV, resting heart rate, soreness, and training load.
- Likely under-recovery after poor sleep plus elevated resting heart rate.
- Training load mismatch when Oura shows many steps but the manual log says rowing.
- Approximate calories and macros from a voice food log.
- Possible digestion triggers from repeated food and symptom patterns.
- Whether today should be training-biased, recovery-biased, or maintenance-biased.

Inference rules:

- Store source and confidence for every inference.
- Prefer "may be", "likely", and "worth testing" over certainty.
- Ask for clarification when the recommendation would materially change.
- Never infer a diagnosis.
- Never infer that a medication or supplement should be started, stopped, or changed.
- Never use one bad day of data to make a strong conclusion.

## Off-Limits Behavior

No health topic is inherently off-limits for discussion, but some actions are off-limits.

The system must not:

- Diagnose medical or psychiatric conditions.
- Prescribe or change medication.
- Tell Krunal to stop medication or ignore a clinician.
- Provide emergency or crisis mental health handling as if it were a professional service.
- Recommend unverified supplements.
- Recommend extreme diets, dehydration, starvation, or unsafe cutting.
- Recommend training through injury red flags.
- Make strong claims from weak wearable signals.
- Hide uncertainty.
- Optimize a single metric at the expense of overall health.

## Allowed Recommendation Areas

The system may recommend:

- Workouts.
- Recovery days.
- Rowing, lifting, cardio, mobility, and walking adjustments.
- Meals and meal timing.
- Vegan-vegetarian protein strategies.
- Bedtime and sleep routine adjustments.
- Hydration reminders if needed.
- Lab tests to discuss with a clinician.
- Meditation schedule protection.
- Low-risk experiments, such as meal timing, workout timing, sleep schedule, protein target, or evening routine.

## Outcomes To Improve

- Gain more muscle.
- Reduce body fat while preserving or increasing muscle.
- Improve sleep further.
- Maintain or improve mood.
- Improve digestion.
- Build very strong focus.
- Maintain strong daily energy.
- Improve athletic performance.
- Support healthy testosterone levels.
- Increase peace and stability of mind.

