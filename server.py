"""Flask API serving predictions from the regression model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

from model import def_model

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow import keras


app = Flask(__name__)
# Limit request bodies to a small size to reduce the risk of denial-of-service
# attacks that attempt to stream arbitrarily large payloads to the API. The
# model only needs 13 floating point features, so a 16 KiB ceiling is more than
# sufficient for legitimate requests.
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
MODEL_FEATURES = 13


def _load_model_weights(model) -> None:
    """Load the persisted model weights from disk."""

    weights_path = Path(__file__).with_name("model.h5")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    model.load_weights(weights_path)


def _coerce_payload(payload: Any) -> np.ndarray:
    """Validate and coerce the input payload into a numeric tensor."""

    if isinstance(payload, str):
        tokens: Iterable[str] = (item.strip() for item in payload.split(","))
    elif isinstance(payload, (bytes, bytearray)):
        try:
            decoded = payload.decode()
        except UnicodeDecodeError as exc:  # pragma: no cover - defensive
            raise BadRequest("Input payload must be UTF-8 encoded.") from exc
        tokens = (item.strip() for item in decoded.split(","))
    elif isinstance(payload, Iterable) and not isinstance(payload, Mapping):
        tokens = payload
    else:
        raise BadRequest("Input must be a comma separated string or iterable of values.")

    values = []
    for index, item in enumerate(tokens):
        if item == "":
            raise BadRequest(f"Feature at position {index} is empty.")
        try:
            values.append(float(item))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise BadRequest(f"Feature at position {index} must be numeric.") from exc
        if len(values) > MODEL_FEATURES:
            raise BadRequest(
                f"Exactly {MODEL_FEATURES} features are required, received more than {MODEL_FEATURES}."
            )

    if len(values) != MODEL_FEATURES:
        raise BadRequest(
            f"Exactly {MODEL_FEATURES} features are required, received {len(values)}."
        )

    return np.asarray(values, dtype=np.float32).reshape(1, MODEL_FEATURES)


@app.errorhandler(BadRequest)
def handle_bad_request(exc: BadRequest):  # type: ignore[override]
    """Return JSON responses for validation errors."""

    return jsonify({"error": exc.description}), exc.code


@app.get("/")
def healthcheck() -> dict[str, str]:
    """Simple health-check endpoint."""

    return {"status": "ok"}


@app.post("/api")
def api() -> dict[str, float]:
    """Return the model prediction for the provided feature vector."""

    if request.is_json:
        req_data = request.get_json(silent=True)
        if req_data is None:
            raise BadRequest("Request body contained invalid JSON.")
        if not isinstance(req_data, Mapping):
            raise BadRequest("JSON payload must be an object with an 'input' field.")
        payload = req_data.get("input")
    else:
        payload = request.form.get("input")

    if payload is None:
        raise BadRequest("The 'input' field is required.")

    clean = _coerce_payload(payload)
    prediction = float(model.predict(clean, verbose=0).squeeze())
    return {"prediction": prediction}


model: "keras.Model" = def_model()
_load_model_weights(model)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
