"""Flask API server for Boston housing price predictions."""

import logging
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from .model import create_model
from .utils import format_prediction, get_feature_names, validate_input

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global model variable
model = None
MODEL_PATH = os.environ.get(
    "MODEL_PATH", str(Path(__file__).parent.parent / "models" / "model.h5")
)


def load_model():
    """Load the pre-trained model."""
    global model
    try:
        model = create_model()
        model.load_weights(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.route("/")
def index() -> str:
    """
    Root endpoint returning basic API information.

    Returns:
        Welcome message and API information
    """
    return jsonify(
        {
            "name": "Boston Housing Price Predictor API",
            "version": "2.0.0",
            "description": "MLP regression model for predicting Boston area home values",
            "endpoints": {
                "/": "This help message",
                "/health": "Health check endpoint",
                "/api/predict": "POST endpoint for predictions",
                "/api/features": "GET endpoint for feature information",
            },
            "status": "running",
        }
    )


@app.route("/health")
def health_check() -> tuple:
    """
    Health check endpoint for container orchestration.

    Returns:
        JSON response with health status and HTTP status code
    """
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "Model not loaded"}), 503

    return jsonify({"status": "healthy", "model_loaded": True}), 200


@app.route("/api/features", methods=["GET"])
def get_features() -> dict[str, Any]:
    """
    Get information about expected input features.

    Returns:
        JSON with feature names and descriptions
    """
    from .utils import get_feature_descriptions

    return jsonify(
        {
            "features": get_feature_names(),
            "descriptions": get_feature_descriptions(),
            "count": 13,
            "format": "Comma-separated values or JSON array",
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict() -> tuple:
    """
    Prediction endpoint for home value estimation.

    Expected input format (JSON):
        {"input": "val1,val2,...,val13"} or {"input": [val1, val2, ..., val13]}

    Returns:
        JSON response with prediction and HTTP status code

    Example:
        POST /api/predict
        {"input": "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"}

        Response:
        {"prediction": 18.4, "unit": "thousands of dollars"}
    """
    global model

    if model is None:
        logger.error("Prediction attempted with no model loaded")
        return jsonify({"error": "Model not loaded"}), 503

    try:
        # Parse input
        if request.is_json:
            req_data = request.get_json()
            if "input" not in req_data:
                return jsonify({"error": "Missing 'input' field in JSON"}), 400
            input_data = req_data["input"]
        else:
            # Support form data for backward compatibility
            if "input" not in request.form:
                return jsonify({"error": "Missing 'input' field"}), 400
            input_data = request.form["input"]

        # Strip whitespace if string
        if isinstance(input_data, str):
            input_data = input_data.strip()

        # Validate and preprocess input
        processed_input = validate_input(input_data)

        # Make prediction
        raw_prediction = model.predict(processed_input, verbose=0)
        prediction = format_prediction(raw_prediction)

        logger.info(f"Prediction made: {prediction:.2f}")

        return jsonify(
            {
                "prediction": round(prediction, 2),
                "unit": "thousands of dollars",
                "success": True,
            }
        ), 200

    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        return jsonify(
            {
                "error": "Invalid input",
                "message": str(e),
                "expected_format": "13 comma-separated numeric values or array",
            }
        ), 400

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500


@app.errorhandler(404)
def not_found(error) -> tuple:
    """Handle 404 errors."""
    return jsonify(
        {"error": "Not found", "message": "The requested endpoint does not exist"}
    ), 404


@app.errorhandler(500)
def internal_error(error) -> tuple:
    """Handle 500 errors."""
    return jsonify(
        {"error": "Internal server error", "message": "An unexpected error occurred"}
    ), 500


def create_app() -> Flask:
    """
    Application factory for creating Flask app instances.

    Returns:
        Configured Flask application
    """
    load_model()
    return app


def main():
    """Main entry point for the server."""
    # Load model on startup
    load_model()

    # Get configuration from environment
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5002"))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
