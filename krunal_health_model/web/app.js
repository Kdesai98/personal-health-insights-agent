const sampleInputDir =
  "/Users/krunal_desai/Documents/Playground/personal-health-insights-agent/krunal_health_model/data/sample";

const $ = (id) => document.getElementById(id);

let apiToken = localStorage.getItem("krunalHealthApiToken") || "";

function setJson(id, value) {
  $(id).textContent = JSON.stringify(value, null, 2);
}

async function api(path, options = {}) {
  const authHeaders = apiToken ? { Authorization: `Bearer ${apiToken}` } : {};
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...authHeaders, ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

async function refreshLearning() {
  const learning = await api("/learning-summary?user_id_hash=krunal_demo_user");
  $("feedback-count").textContent = `${learning.feedback_events} feedback events`;
  setJson("learning-json", learning.advanced_learning || learning);
}

async function runDailyCycle() {
  const result = await api("/daily-cycle", {
    method: "POST",
    body: JSON.stringify({
      user_id_hash: "krunal_demo_user",
      input_dir: sampleInputDir,
    }),
  });
  $("day-type").textContent = result.plan.day_type;
  $("summary").textContent = result.plan.summary;
  $("workout").textContent = result.plan.workout;
  $("nutrition").textContent = result.plan.nutrition;
  $("question").textContent = result.plan.question || "-";
  setJson("learning-json", result.learning_summary.advanced_learning || result.learning_summary);
  $("feedback-count").textContent = `${result.learning_summary.feedback_events} feedback events`;
}

async function parseFood() {
  const result = await api("/voice/food/parse", {
    method: "POST",
    body: JSON.stringify({ raw_text: $("food-text").value }),
  });
  setJson("food-json", result);
}

async function recordFeedback() {
  const result = await api("/feedback", {
    method: "POST",
    body: JSON.stringify({
      user_id_hash: "krunal_demo_user",
      date: "2026-04-19",
      action_id: $("feedback-action").value,
      adhered: true,
      reward: Number($("feedback-reward").value),
      outcomes: {
        energy_next_day_1_to_10: 8,
        digestion_next_day_1_to_10: 7,
        readiness_next_day: 84,
      },
    }),
  });
  setJson("feedback-json", result);
  await refreshLearning();
}

async function parseLabs() {
  const result = await api("/labs/parse", {
    method: "POST",
    body: JSON.stringify({ text: $("lab-text").value }),
  });
  setJson("labs-json", result);
}

async function evaluateEscalation() {
  const result = await api("/escalation/evaluate", {
    method: "POST",
    body: JSON.stringify({ free_text: $("escalation-text").value }),
  });
  setJson("escalation-json", result);
}

async function boot() {
  try {
    const status = await api("/health");
    $("api-status").textContent = `Backend ${status.status}`;
    await refreshLearning();
  } catch (error) {
    $("api-status").textContent = error.message;
  }
}

$("run-cycle").addEventListener("click", runDailyCycle);
$("save-token").addEventListener("click", () => {
  apiToken = $("api-token").value.trim();
  localStorage.setItem("krunalHealthApiToken", apiToken);
  $("api-status").textContent = apiToken ? "Token saved" : "Token cleared";
});
$("parse-food").addEventListener("click", parseFood);
$("record-feedback").addEventListener("click", recordFeedback);
$("parse-labs").addEventListener("click", parseLabs);
$("evaluate-escalation").addEventListener("click", evaluateEscalation);

boot();
$("api-token").value = apiToken;
