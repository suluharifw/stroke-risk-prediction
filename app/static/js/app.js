/* Stroke Risk Assessment — frontend interactions.
 * - Live BMI / glucose category chips as the user types.
 * - Async prediction with an animated gauge + advice reveal.
 * The server owns all feature logic; the chip bands below are UI hints only.
 */
(() => {
  "use strict";

  const form = document.getElementById("assessment-form");
  const btn = document.getElementById("assess-btn");
  const errorEl = document.getElementById("form-error");
  const card = document.getElementById("result-card");
  const idle = document.getElementById("result-idle");
  const outcome = document.getElementById("result-outcome");

  // ---- Live category chips (mirror the model's clinical bands) ----
  const bmiChip = document.getElementById("bmi-chip");
  const glucoseChip = document.getElementById("glucose-chip");

  const setChip = (el, text, tone) => {
    if (!text) { el.hidden = true; return; }
    el.textContent = text;
    el.dataset.tone = tone;
    el.hidden = false;
  };

  const bmiBand = (v) => {
    if (!v) return null;
    if (v < 18.5) return ["Underweight", "warn"];
    if (v <= 24.9) return ["Normal", "ok"];
    if (v <= 29.9) return ["Overweight", "warn"];
    return ["Obesity", "high"];
  };
  const glucoseBand = (v) => {
    if (!v) return null;
    if (v < 70) return ["Hypoglycemia", "warn"];
    if (v <= 140) return ["Normal", "ok"];
    if (v <= 199) return ["Prediabetic", "warn"];
    return ["Hyperglycemia", "high"];
  };

  document.getElementById("bmi").addEventListener("input", (e) => {
    const band = bmiBand(parseFloat(e.target.value));
    setChip(bmiChip, band && band[0], band && band[1]);
  });
  document.getElementById("avg_glucose_level").addEventListener("input", (e) => {
    const band = glucoseBand(parseFloat(e.target.value));
    setChip(glucoseChip, band && band[0], band && band[1]);
  });

  // ---- Gauge animation ----
  const ARC_LENGTH = 258; // must match stroke-dasharray in CSS
  const gaugeFill = document.getElementById("gauge-fill");
  const needle = document.getElementById("needle");
  const gaugePct = document.getElementById("gauge-pct");

  const animateGauge = (probability) => {
    const pct = Math.max(0, Math.min(1, probability));
    gaugeFill.style.strokeDashoffset = String(ARC_LENGTH * (1 - pct));
    needle.style.transform = `rotate(${-90 + pct * 180}deg)`;

    // Count-up the percentage readout.
    const target = Math.round(pct * 100);
    let current = 0;
    const step = () => {
      current += Math.max(1, Math.ceil((target - current) / 6));
      if (current >= target) current = target;
      gaugePct.textContent = `${current}%`;
      if (current < target) requestAnimationFrame(step);
    };
    step();
  };

  // ---- Render an outcome ----
  const renderOutcome = (data) => {
    const higher = data.band === "higher";

    document.getElementById("risk-eyebrow").textContent = "Estimated stroke risk";

    const verdict = document.getElementById("verdict");
    verdict.dataset.band = data.band;
    verdict.textContent = higher ? "Higher risk — follow up advised" : "Lower risk";

    const list = document.getElementById("advice-list");
    list.innerHTML = "";
    data.advice.forEach((item, i) => {
      const li = document.createElement("li");
      li.textContent = item;
      li.style.animationDelay = `${0.15 + i * 0.06}s`;
      list.appendChild(li);
    });

    idle.hidden = true;
    outcome.hidden = false;
    card.dataset.state = "done";
    requestAnimationFrame(() => animateGauge(data.probability));
  };

  // ---- Submit ----
  const collect = () => ({
    age: document.getElementById("age").value,
    gender: document.getElementById("gender").value,
    ever_married: document.getElementById("ever_married").value,
    work_type: document.getElementById("work_type").value,
    avg_glucose_level: document.getElementById("avg_glucose_level").value,
    bmi: document.getElementById("bmi").value,
    hypertension: document.getElementById("hypertension").value,
    heart_disease: document.getElementById("heart_disease").value,
    smoking_status: document.getElementById("smoking_status").value,
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorEl.hidden = true;

    const payload = collect();
    if (!payload.age || !payload.avg_glucose_level || !payload.bmi) {
      errorEl.textContent = "Please fill in age, glucose, and BMI.";
      errorEl.hidden = false;
      return;
    }

    btn.disabled = true;
    btn.textContent = "Assessing…";
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");
      renderOutcome(data);
    } catch (err) {
      errorEl.textContent = err.message;
      errorEl.hidden = false;
    } finally {
      btn.disabled = false;
      btn.textContent = "Assess risk";
    }
  });

  form.addEventListener("reset", () => {
    bmiChip.hidden = true;
    glucoseChip.hidden = true;
    outcome.hidden = true;
    idle.hidden = false;
    card.dataset.state = "idle";
    errorEl.hidden = true;
  });
})();
