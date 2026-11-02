from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Allows requests from your React development server.

MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_risk_model.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]

@app.get("/")
def home():
    return jsonify({
        "message": "Bank Loan Risk Prediction API is running",
        "endpoints": {
            "health": "GET /api/health",
            "model_info": "GET /api/model-info",
            "predict": "POST /api/predict"
        }
    })

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None}), 200

@app.get("/api/model-info")
def model_info():
    return jsonify({
        "model": "RandomForestClassifier",
        "n_estimators": model.n_estimators,
        "features": FEATURES,
        "target": "Risk_Flag",
        "classes": {"0": "LOW RISK", "1": "HIGH RISK"}
    }), 200

@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "success": False,
            "error": "Request body must be a JSON object."
        }), 400

    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        return jsonify({
            "success": False,
            "error": "Missing required fields.",
            "missing_fields": missing,
            "required_fields": FEATURES
        }), 400

    try:
        values = {feature: float(payload[feature]) for feature in FEATURES}
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "All input fields must contain numeric values."
        }), 400

    if not all(np.isfinite(value) for value in values.values()):
        return jsonify({"success": False, "error": "Values must be finite numbers."}), 400

    # Basic input validation; adjust limits to your dataset/business rules if needed.
    if not 18 <= values["Age"] <= 100:
        return jsonify({"success": False, "error": "Age must be between 18 and 100."}), 400
    if values["Annual_Income"] < 0 or values["Loan_Amount"] < 0:
        return jsonify({"success": False, "error": "Income and loan amount cannot be negative."}), 400
    if not 300 <= values["Credit_Score"] <= 900:
        return jsonify({"success": False, "error": "Credit score must be between 300 and 900."}), 400
    if not 0 <= values["Debt_to_Income_Ratio"] <= 1:
        return jsonify({"success": False, "error": "Debt-to-income ratio must be between 0 and 1."}), 400
    if values["Loan_Term_Months"] <= 0:
        return jsonify({"success": False, "error": "Loan term must be greater than zero."}), 400
    if values["Past_Loans"] < 0 or values["Defaults"] < 0:
        return jsonify({"success": False, "error": "Past loans and defaults cannot be negative."}), 400

    input_df = pd.DataFrame([values], columns=FEATURES)
    prediction = int(model.predict(input_df)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        class_index = list(model.classes_).index(1) if 1 in model.classes_ else None
        if class_index is not None:
            probability = round(float(probabilities[class_index]), 4)

    result = "HIGH RISK" if prediction == 1 else "LOW RISK"

    return jsonify({
        "success": True,
        "prediction": prediction,
        "risk": result,
        "high_risk_probability": probability,
        "message": f"Customer is classified as {result}."
    }), 200

@app.errorhandler(404)
def not_found(_error):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404

if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
