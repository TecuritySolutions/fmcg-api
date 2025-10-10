"""
FMCG Intelligent Forecasting API
--------------------------------
Production-ready Flask backend for FMCG model predictions.
- Accepts JSON input from frontend forms or Postman.
- Individually extracts and validates input fields.
- Serves multiple pre-trained ML models.
- Follows PEP-8 coding standards and clean API architecture.
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------------------
# Application Setup
# ------------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------------------------
# Model Registry
# ------------------------------------------------------------------------------

MODELS_DIR = "models"

MODELS = {
    "fmcg_darknet": joblib.load(os.path.join(MODELS_DIR, "fmcg-1.joblib")),
    "fmcg_hashlock": joblib.load(os.path.join(MODELS_DIR, "fmcg-2.joblib")),
    "fmcg_infinity": joblib.load(os.path.join(MODELS_DIR, "fmcg-5.joblib")),
}


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes string or categorical columns using LabelEncoder.
    Returns a numeric dataframe ready for prediction.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            encoder = LabelEncoder()
            try:
                df[col] = encoder.fit_transform(df[col])
            except Exception:
                df[col] = 0
    return df


def validate_fields(payload: dict) -> list:
    """
    Validates that all required input fields are present in the payload.
    Returns a list of missing field names.
    """
    required_fields = [
        "Location_type", "WH_capacity_size", "zone", "WH_regional_zone",
        "num_refill_req_l3m", "transport_issue_l1y", "Competitor_in_mkt",
        "retail_shop_num", "wh_owner_type", "distributor_num",
        "flood_impacted", "flood_proof", "electric_supply", "dist_from_hub",
        "workers_num", "wh_est_year", "storage_issue_reported_l3m",
        "temp_reg_mach", "approved_wh_govt_certificate", "wh_breakdown_l3m",
        "govt_check_l3m", "product_wg_ton"
    ]
    return [field for field in required_fields if payload.get(field) is None]


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Root API route for metadata."""
    return jsonify({
        "service": "FMCG Prediction API",
        "status": "active",
        "available_models": list(MODELS.keys()),
        "usage": "POST /predict with JSON body"
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Handles FMCG model prediction requests."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid or empty JSON payload"}), 400

        model_key = data.get("model")
        if not model_key or model_key not in MODELS:
            return jsonify({
                "error": "Invalid or missing 'model' key",
                "available_models": list(MODELS.keys())
            }), 400

        # Validate all fields
        missing_fields = validate_fields(data)
        if missing_fields:
            return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

        # Create DataFrame
        input_df = pd.DataFrame([{
            "Location_type": data.get("Location_type"),
            "WH_capacity_size": data.get("WH_capacity_size"),
            "zone": data.get("zone"),
            "WH_regional_zone": data.get("WH_regional_zone"),
            "num_refill_req_l3m": data.get("num_refill_req_l3m"),
            "transport_issue_l1y": data.get("transport_issue_l1y"),
            "Competitor_in_mkt": data.get("Competitor_in_mkt"),
            "retail_shop_num": data.get("retail_shop_num"),
            "wh_owner_type": data.get("wh_owner_type"),
            "distributor_num": data.get("distributor_num"),
            "flood_impacted": data.get("flood_impacted"),
            "flood_proof": data.get("flood_proof"),
            "electric_supply": data.get("electric_supply"),
            "dist_from_hub": data.get("dist_from_hub"),
            "workers_num": data.get("workers_num"),
            "wh_est_year": data.get("wh_est_year"),
            "storage_issue_reported_l3m": data.get("storage_issue_reported_l3m"),
            "temp_reg_mach": data.get("temp_reg_mach"),
            "approved_wh_govt_certificate": data.get("approved_wh_govt_certificate"),
            "wh_breakdown_l3m": data.get("wh_breakdown_l3m"),
            "govt_check_l3m": data.get("govt_check_l3m"),
            "product_wg_ton": data.get("product_wg_ton")
        }])

        # Encode categorical data
        input_df = encode_dataframe(input_df)

        # Match features with model input
        model = MODELS[model_key]
        if hasattr(model, "feature_names_in_"):
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        # Generate prediction
        prediction = model.predict(input_df)[0]

        try:
            score = float(prediction)
            # score = max(0, min(100, score))
        except Exception:
            score = str(prediction)

        return jsonify({
            "status": "success",
            "model_used": model_key,
            "prediction_score": round(float(prediction), 2)
        })

    except Exception as err:
        return jsonify({"error": f"Server Error: {str(err)}"}), 500


# ------------------------------------------------------------------------------
# Health Check Route
# ------------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Returns the health status of the API."""
    return jsonify({"status": "OK", "service": "FMCG Prediction API"})


# ------------------------------------------------------------------------------
# Main Entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
