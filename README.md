# mlpregression

An end-to-end demo of deploying a small regression model behind a Flask API. The
model predicts the median value of owner-occupied homes (in thousands of
dollars) from the Boston Housing dataset using a Multi-Layer Perceptron (MLP).
The repository is intentionally lightweight so it can be used in workshops or
training sessions that demonstrate how to ship a simple machine learning model
as an HTTP service.

## Repository structure

```text
.
├── demo.ipynb        # Notebook used to explore the dataset and train the model
├── model.h5          # Trained Keras model weights saved from the notebook
├── model.py          # Function that builds the MLP architecture
├── server.py         # Flask app that loads the weights and exposes predictions
├── pyproject.toml    # Python dependencies managed by uv
└── Dockerfile        # Container image definition for production or demos
```

### What the project does

1. **Model definition** – `model.py` contains the `def_model()` helper that
   builds the Keras MLP used everywhere in the project. The architecture has
   two hidden layers (50 and 10 neurons) with ReLU activations and a single
   output neuron for the regression target.
2. **Weight loading** – `model.h5` is created from the notebook and contains the
   trained parameters. The Flask service loads these weights at startup.
3. **Prediction API** – `server.py` bootstraps the model, validates requests,
   and returns JSON predictions. The service accepts the 13 numeric features
   required by the dataset either as JSON or form-encoded payloads.

If you want to retrain the model, open `demo.ipynb`, experiment with the data,
and save the weights again using `model.save_weights("model.h5")`.

## Prerequisites

- Python 3.10 or later.
- [uv](https://docs.astral.sh/uv/latest/) for managing dependencies. uv is a
  drop-in replacement for `pip` and `virtualenv` that provides reproducible
  environments.
  - macOS / Linux: `curl -Ls https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell):
    ```powershell
    iwr https://astral.sh/uv/install.ps1 -useb | iex
    ```
- (Optional) Docker if you plan to containerise the service.

## Installation and local development

Clone the repository and install the dependencies inside a project-specific
virtual environment managed by uv:

```bash
uv sync
```

The command reads `pyproject.toml`, creates the virtual environment, and
installs TensorFlow, Flask, and the other dependencies required by the service.
If you would like to run the exploratory notebook, include its optional
dependencies by syncing with the `notebook` extra:

```bash
uv sync --extra notebook
```

Start the development server with:

```bash
uv run python server.py
```

The API listens on <http://127.0.0.1:5002>. A minimal smoke test can be
performed with `curl` by providing the 13 feature values separated by commas or
as a JSON list:

```bash
curl -X POST "http://127.0.0.1:5002/api" \
  -H "Content-Type: application/json" \
  -d '{"input": [0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.09, 1.0, 296.0, 15.3, 396.9, 4.98]}'
```

Example response:

```json
{"prediction": 27.16509437561035}
```

If you send a plain string instead of JSON, the values must still be separated
by commas: `curl -X POST -d "input=0.00632,18.0,..." http://127.0.0.1:5002/api`.

The root endpoint acts as a health check and should return `{"status": "ok"}`.

## Docker usage

The repository includes a Dockerfile for running the service without installing
Python locally. Build and run the container with:

```bash
docker build -t mlpregression:latest .
docker run --rm -p 5002:5002 mlpregression:latest
```

The container executes `python server.py` and exposes port `5002`. The same API
contract shown above is available inside the container.

## Understanding the code

### `model.py`

- `def_model()` constructs a `keras.Sequential` network with the layers described
  earlier. The function also compiles the model with mean squared error (MSE) as
  both the loss and metric, and the Adam optimiser. The helper is imported by
  both the notebook and the Flask service to guarantee the architecture matches
  the saved weights.

### `server.py`

- Loads the model architecture from `def_model()` and immediately reads the
  stored weights from `model.h5`.
- Exposes two routes:
  - `GET /` – returns `{"status": "ok"}` for liveness checks.
  - `POST /api` – expects an `input` field containing the 13 numeric features.
- Uses `_coerce_payload()` to parse payloads supplied as JSON, form data, byte
  strings, or CSV-like text. The helper ensures the request contains the correct
  number of features and that each value can be interpreted as a float. Errors
  raise `BadRequest` which Flask translates into a JSON error message.
- `_load_model_weights()` ensures the weight file exists and surfaces a helpful
  error if it does not, which is useful when onboarding new developers.

## Extending the demo

- **Retraining** – modify `demo.ipynb`, retrain the model, and save new weights
  to `model.h5`. The Flask service will pick them up the next time it starts.
- **Custom features** – if you change the dataset or feature engineering,
  remember to adjust `MODEL_FEATURES` and the validation logic in
  `_coerce_payload()` accordingly.
- **Testing** – consider adding unit tests around `_coerce_payload()` and the
  Flask routes to demonstrate automated validation of the service contract.

## Troubleshooting

- `FileNotFoundError: Model weights not found` – ensure `model.h5` exists in the
  repository root. Re-run the notebook or copy the file into place if needed.
- `BadRequest: Exactly 13 features are required` – check that the payload
  contains each feature expected by the model. The README examples show the
  correct order.

## License

This repository is intended for instructional purposes. Adapt and reuse the
code in your own demos or workshops.
