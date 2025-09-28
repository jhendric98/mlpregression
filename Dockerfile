FROM python:3.11-slim

ENV UV_INSTALL_DIR=/usr/local
ENV PATH="${UV_INSTALL_DIR}/bin:/app/.venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --bindir ${UV_INSTALL_DIR}/bin

WORKDIR /app

COPY pyproject.toml ./

RUN uv sync --no-dev

COPY . .

EXPOSE 5002

CMD ["python", "server.py"]
