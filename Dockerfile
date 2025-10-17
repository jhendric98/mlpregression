# Multi-stage build for mlpregression
# Stage 1: Build stage
FROM python:3.12-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=2.0.0

# Add metadata
LABEL org.opencontainers.image.title="mlpregression" \
      org.opencontainers.image.description="MLP regression model for Boston housing price prediction" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Jim Hendricks" \
      org.opencontainers.image.authors="jhendric98@gmail.com" \
      org.opencontainers.image.url="https://github.com/jimhendricks/mlpregression" \
      org.opencontainers.image.source="https://github.com/jimhendricks/mlpregression" \
      org.opencontainers.image.licenses="MIT"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /build

# Copy pyproject.toml and uv.lock first for better caching
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN uv sync --frozen

# Copy source code
COPY . .

# Install the package
RUN uv pip install -e .

# Stage 2: Runtime stage
FROM python:3.12-slim as runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r mlpuser && useradd -r -g mlpuser -u 1000 mlpuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application files
COPY --from=builder /build/mlpregression /app/mlpregression
COPY --from=builder /build/models /app/models

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5002 \
    FLASK_DEBUG=false \
    MODEL_PATH=/app/models/model.h5

# Set working directory
WORKDIR /app

# Change ownership to non-root user
RUN chown -R mlpuser:mlpuser /app

# Switch to non-root user
USER mlpuser

# Expose port
EXPOSE 5002

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

# Set entrypoint and default command
ENTRYPOINT ["python", "-m", "mlpregression.server"]

# Build-time metadata
ARG BUILD_DATE
ARG VCS_REF
LABEL build_date="${BUILD_DATE}" \
      vcs_ref="${VCS_REF}"