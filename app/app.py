"""Flask serving layer for the stroke-risk assessment tool.

Thin by design: it validates the incoming record, delegates to
:class:`stroke_prediction.StrokePredictor`, and returns JSON. All feature
engineering lives in the package, so the web layer never touches encodings.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# Run without installing the package: add ../src to the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stroke_prediction import PatientRecord, StrokePredictor  # noqa: E402

app = Flask(__name__)
CORS(app)

predictor = StrokePredictor()

# Advice is owned by the server so there is a single source of truth.
ADVICE: dict[str, list[str]] = {
    "higher": [
        "Share this result with a doctor — it is a screening signal, not a diagnosis.",
        "Keep blood pressure in range with regular monitoring and prescribed medication.",
        "Prioritise a low-saturated-fat, high-fibre diet with plenty of fruit and vegetables.",
        "Stop smoking and limit alcohol to protect your blood vessels.",
        "Build up light, regular physical activity as your doctor advises.",
        "Learn the FAST stroke warning signs and seek emergency care if they appear.",
    ],
    "lower": [
        "Keep up the habits that protect you — this is a lower-risk result, not a guarantee.",
        "Stay active with regular light exercise through the week.",
        "Maintain a balanced diet and a healthy weight.",
        "Keep blood pressure, glucose, and cholesterol within normal ranges.",
        "Avoid smoking and drink alcohol in moderation.",
        "Schedule routine check-ups to catch any change early.",
    ],
}


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    """Validate the payload, run inference, and return the assessment."""
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON."}), 400
    try:
        record = PatientRecord.from_dict(payload)
    except (KeyError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 422

    result = predictor.predict(record)
    band = "higher" if result.label == 1 else "lower"
    return jsonify(
        {
            "label": result.label,
            "probability": result.probability,
            "band": band,
            "advice": ADVICE[band],
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=5001, debug=True)
