"""Flask API serving predictions from the regression model.

The module performs three tasks when imported:

* builds the Keras model architecture by calling :func:`model.def_model`
* loads the persisted ``model.h5`` weights so predictions are available
* exposes a small Flask application with routes for health checks and inference

The goal is to keep the example compact while still showcasing good production
hygiene such as input validation and explicit error messages.
"""

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
MODEL_FEATURES = 13


def _load_model_weights(model: "keras.Model") -> None:
    """Load the persisted model weights from disk.

    Parameters
    ----------
    model:
        An instance of the architecture returned by :func:`model.def_model`.

    Raises
    ------
    FileNotFoundError
        If ``model.h5`` is missing from the repository root. This is surfaced to
        help new contributors diagnose environment issues quickly.
    """

    weights_path = Path(__file__).with_name("model.h5")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    model.load_weights(weights_path)


def _coerce_payload(payload: Any) -> np.ndarray:
    """Validate and coerce the input payload into a numeric tensor.

    The function accepts either comma-separated strings, JSON lists, plain
    iterables of numbers, or raw bytes. Invalid inputs raise ``BadRequest`` so
    the API returns helpful feedback to clients.

    Parameters
    ----------
    payload:
        The raw request payload received from Flask.

    Returns
    -------
    numpy.ndarray
        A ``(1, MODEL_FEATURES)`` float array ready to be passed to the model.

    Raises
    ------
    BadRequest
        If the payload is missing, has the wrong number of features, or contains
        values that cannot be converted to floats.
    """

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

    if len(values) != MODEL_FEATURES:
        raise BadRequest(
            f"Exactly {MODEL_FEATURES} features are required, received {len(values)}."
        )

    return np.asarray(values, dtype=np.float32).reshape(1, MODEL_FEATURES)


@app.errorhandler(BadRequest)
def handle_bad_request(exc: BadRequest):  # type: ignore[override]
    """Return JSON responses for validation errors.

    Flask would otherwise return HTML responses for 400 errors. Serialising the
    error message keeps the API consistent for clients.
    """

    return jsonify({"error": exc.description}), exc.code


@app.get("/")
def healthcheck() -> dict[str, str]:
    """Return a simple success payload to confirm the service is running."""

    return {"status": "ok"}


@app.post("/api")
def api() -> dict[str, float]:
    """Return the model prediction for the provided feature vector.

    The route accepts either JSON ``{"input": [...]}`` or form data
    ``input=...``. Payload validation is delegated to :func:`_coerce_payload`.
    """

    if request.is_json:
        req_data = request.get_json(silent=True) or {}
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
