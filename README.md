# mlpregression

This project implements an MLP regression for a regression prediction of mean home value on the Boston Home training dataset. The model is implemented and trained in Jupyter notebooks then saved to `model.py` (the architecture) and `model.h5` (trained weights).

The prediction service in `server.py` exposes the model behind a small Flask API. The runtime environment is now managed by [uv](https://docs.astral.sh/uv/latest/), which keeps dependencies reproducible and dramatically speeds up installation during local development and Docker builds.

## Local development

```bash
# Create a virtual environment managed by uv and install dependencies
uv sync

# Run the Flask development server
uv run python server.py
```

## Docker image

```bash
docker build -t mlpregression:latest .
docker run --rm -p 5002:5002 mlpregression:latest
```

The container exposes the API on port `5002`. A simple health-check is available at `GET /` and predictions can be requested with `POST /api` by providing an `input` list (JSON) or comma-separated string.
