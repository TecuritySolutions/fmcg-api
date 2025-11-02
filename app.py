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

MODELS = {}

# Load models with error handling
model_files = {
    "fmcg_darknet": "fmcg-1.joblib",
    "fmcg_hashlock": "fmcg-2.joblib",
    "fmcg_infinity": "fmcg-5.joblib"
}

for model_name, model_file in model_files.items():
    try:
        MODELS[model_name] = joblib.load(os.path.join(MODELS_DIR, model_file))
        print(f"✓ Loaded model: {model_name}")
    except Exception as e:
        print(f"⚠ Warning: Could not load model {model_name}: {e}")
        # Add placeholder for demo purposes
        MODELS[model_name] = "model_placeholder"

# Ensure at least one model is available for demo
if not any(model != "model_placeholder" for model in MODELS.values()):
    print("⚠ Running in demo mode - no models loaded successfully")


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
        print("Received data ====> ", data)
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
        print("prediction score ====> ", round(float(prediction), 2))
        return jsonify({
            "status": "success",
            "model_used": model_key,
            "prediction_score": round(float(prediction), 2)
        })

    except Exception as err:
        return jsonify({"error": f"Server Error: {str(err)}"}), 500

# --------------------------------------------------------------------------
# Pincode Autofill Route (NEW)
# --------------------------------------------------------------------------

@app.route("/pincode", methods=["POST"])
def get_pincode_details():
    """
    Fetches location details (population, landmarks, etc.)
    based on the given pincode using local CSV data.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        # Accept either 'pincode' or 'Pincode'
        pincode = str(data.get("pincode") or data.get("Pincode") or "").strip()
        if not pincode:
            return jsonify({"error": "Missing 'pincode' field"}), 400

        # Load CSV (in same folder as app.py)
        df = pd.read_csv("pincode_data.csv", dtype=str)

        # Normalize column names (remove case sensitivity)
        df.columns = df.columns.str.strip().str.lower()

        # Match pincode
        if "pincode" not in df.columns:
            return jsonify({"error": "CSV missing 'pincode' column"}), 500

        matched_row = df[df["pincode"].astype(str) == pincode]
        if matched_row.empty:
            return jsonify({"error": f"No data found for pincode {pincode}"}), 404

        record = matched_row.iloc[0].to_dict()

        # Safely extract available details
        details = {
            "pincode": record.get("Pincode")or record.get("pincode"),
            "population": record.get("population") or record.get("Population"),
            "landmark": record.get("Place_name") or record.get("place_name") or record.get("landmark"),
        }

        return jsonify({"status": "success", "details": details})

    except Exception as e:
        return jsonify({"error": f"Server Error: {str(e)}"}), 500



# ------------------------------------------------------------------------------
# Health Check Route
# ------------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Returns the health status of the API."""
    return jsonify({"status": "OK", "service": "FMCG Prediction API"})


@app.route("/docs", methods=["GET"])
def get_docs():
    """Returns comprehensive API documentation."""
    docs = {
        "api_info": {
            "name": "FMCG Intelligent Forecasting API",
            "version": "1.0.0",
            "description": "Production-ready Flask backend for FMCG model predictions. Accepts JSON input from frontend forms or Postman, validates input fields, and serves multiple pre-trained ML models.",
            "base_url": request.url_root.rstrip('/')
        },
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Root API route for metadata and service status",
                "parameters": [],
                "response_example": {
                    "service": "FMCG Prediction API",
                    "status": "active",
                    "available_models": list(MODELS.keys()),
                    "usage": "POST /predict with JSON body"
                }
            },
            {
                "path": "/predict",
                "method": "POST",
                "description": "Main prediction endpoint for FMCG warehouse forecasting",
                "content_type": "application/json",
                "required_fields": [
                    {"name": "model", "type": "string", "description": "Model to use for prediction", "allowed_values": list(MODELS.keys())},
                    {"name": "Location_type", "type": "string", "description": "Type of warehouse location (Urban/Rural/Semi-urban)"},
                    {"name": "WH_capacity_size", "type": "numeric", "description": "Warehouse capacity size in cubic units"},
                    {"name": "zone", "type": "string", "description": "Geographical zone (North/South/East/West/Central)"},
                    {"name": "WH_regional_zone", "type": "string", "description": "Regional administrative zone"},
                    {"name": "num_refill_req_l3m", "type": "numeric", "description": "Number of refill requests in last 3 months"},
                    {"name": "transport_issue_l1y", "type": "numeric", "description": "Transportation issues reported in last 1 year"},
                    {"name": "Competitor_in_mkt", "type": "numeric", "description": "Number of competitors in market area"},
                    {"name": "retail_shop_num", "type": "numeric", "description": "Number of retail shops served"},
                    {"name": "wh_owner_type", "type": "string", "description": "Warehouse ownership type (Private/Government/Leased)"},
                    {"name": "distributor_num", "type": "numeric", "description": "Number of distributors connected"},
                    {"name": "flood_impacted", "type": "numeric", "description": "Flood impact history (0=No, 1=Yes)"},
                    {"name": "flood_proof", "type": "numeric", "description": "Flood protection measures (0=No, 1=Yes)"},
                    {"name": "electric_supply", "type": "numeric", "description": "Electricity supply reliability score"},
                    {"name": "dist_from_hub", "type": "numeric", "description": "Distance from distribution hub in kilometers"},
                    {"name": "workers_num", "type": "numeric", "description": "Number of workers employed"},
                    {"name": "wh_est_year", "type": "numeric", "description": "Year warehouse was established"},
                    {"name": "storage_issue_reported_l3m", "type": "numeric", "description": "Storage issues reported in last 3 months"},
                    {"name": "temp_reg_mach", "type": "numeric", "description": "Temperature regulation machinery (0=No, 1=Yes)"},
                    {"name": "approved_wh_govt_certificate", "type": "numeric", "description": "Government certification status (0=No, 1=Yes)"},
                    {"name": "wh_breakdown_l3m", "type": "numeric", "description": "Warehouse breakdowns in last 3 months"},
                    {"name": "govt_check_l3m", "type": "numeric", "description": "Government inspections in last 3 months"},
                    {"name": "product_wg_ton", "type": "numeric", "description": "Product weight capacity in tons"}
                ],
                "request_example": {
                    "model": "fmcg_darknet",
                    "Location_type": "Urban",
                    "WH_capacity_size": 1000,
                    "zone": "North",
                    "WH_regional_zone": "Zone-A",
                    "num_refill_req_l3m": 5,
                    "transport_issue_l1y": 2,
                    "Competitor_in_mkt": 3,
                    "retail_shop_num": 15,
                    "wh_owner_type": "Private",
                    "distributor_num": 8,
                    "flood_impacted": 0,
                    "flood_proof": 1,
                    "electric_supply": 85,
                    "dist_from_hub": 25,
                    "workers_num": 12,
                    "wh_est_year": 2018,
                    "storage_issue_reported_l3m": 1,
                    "temp_reg_mach": 1,
                    "approved_wh_govt_certificate": 1,
                    "wh_breakdown_l3m": 0,
                    "govt_check_l3m": 2,
                    "product_wg_ton": 500
                },
                "response_example": {
                    "status": "success",
                    "model_used": "fmcg_darknet",
                    "prediction_score": 85.32
                },
                "error_responses": [
                    {"code": 400, "description": "Invalid or missing fields", "example": {"error": "Missing fields: ['Location_type', 'zone']"}},
                    {"code": 400, "description": "Invalid model selection", "example": {"error": "Invalid or missing 'model' key", "available_models": list(MODELS.keys())}},
                    {"code": 500, "description": "Server error during prediction", "example": {"error": "Server Error: [error details]"}}
                ]
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint to verify API status",
                "response_example": {
                    "status": "OK",
                    "service": "FMCG Prediction API"
                }
            },
            {
                "path": "/docs",
                "method": "GET",
                "description": "This comprehensive API documentation endpoint"
            }
        ],
        "models": {
            "fmcg_darknet": {
                "file": "models/fmcg-1.joblib",
                "description": "Primary FMCG forecasting model for warehouse performance prediction"
            },
            "fmcg_hashlock": {
                "file": "models/fmcg-2.joblib",
                "description": "Alternative FMCG forecasting model with different training parameters"
            },
            "fmcg_infinity": {
                "file": "models/fmcg-5.joblib",
                "description": "Advanced FMCG forecasting model optimized for complex scenarios"
            }
        },
        "usage_examples": {
            "curl": f"curl -X POST {request.url_root}predict -H 'Content-Type: application/json' -d '{{\"model\": \"fmcg_darknet\", \"Location_type\": \"Urban\", \"WH_capacity_size\": 1000, \"zone\": \"North\", ...}}'",
            "python": f"""import requests
import json

url = "{request.url_root}predict"
data = {{
    "model": "fmcg_darknet",
    "Location_type": "Urban",
    "WH_capacity_size": 1000,
    "zone": "North",
    # ... include all 22 required fields
}}

response = requests.post(url, json=data)
result = response.json()
print(result)""",
            "javascript": f"""fetch('{request.url_root}predict', {{
  method: 'POST',
  headers: {{
    'Content-Type': 'application/json',
  }},
  body: JSON.stringify({{
    model: 'fmcg_darknet',
    Location_type: 'Urban',
    WH_capacity_size: 1000,
    zone: 'North',
    // ... include all 22 required fields
  }})
}})
.then(response => response.json())
.then(data => console.log(data));"""
        }
    }

    return jsonify(docs)


# ------------------------------------------------------------------------------
# Main Entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
