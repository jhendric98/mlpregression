# Installation Guide

This guide provides detailed installation instructions for mlpregression across different environments and use cases.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Docker Installation](#docker-installation)
- [Development Installation](#development-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Python**: 3.10, 3.11, or 3.12 (3.13 not yet supported by TensorFlow)
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Memory**: 4GB RAM (8GB recommended)
- **Disk Space**: 2GB free space
- **Internet**: Required for initial installation

### Recommended Requirements

- **Python**: 3.12 (latest supported)
- **Package Manager**: UV (faster than pip)
- **Memory**: 8GB RAM or more
- **CPU**: Multi-core processor for better performance
- **GPU**: CUDA-compatible GPU for faster training (optional)

## Installation Methods

### Method 1: Using UV (Recommended)

UV is a fast Python package manager. Install mlpregression using UV:

```bash
# Install UV first (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add mlpregression to your project
uv add mlpregression

# Or install globally
uv tool install mlpregression
```

### Method 2: PyPI Installation with pip

Traditional installation via pip from PyPI:

```bash
pip install mlpregression
```

For the latest version with all optional dependencies:

```bash
pip install mlpregression[plotting,docs]
```

### Method 3: From Source with UV

Install the latest development version from GitHub:

```bash
# Clone the repository
git clone https://github.com/jimhendricks/mlpregression.git
cd mlpregression

# Sync dependencies with UV
uv sync
```

### Method 4: From Source with pip

```bash
# Clone the repository
git clone https://github.com/jimhendricks/mlpregression.git
cd mlpregression

# Install in development mode
pip install -e .
```

### Method 5: Specific Version

Install a specific version:

```bash
# With UV
uv add mlpregression==2.0.0

# With pip
pip install mlpregression==2.0.0
```

## Virtual Environment Setup

We strongly recommend using a virtual environment to avoid dependency conflicts.

### Using UV (Recommended)

UV automatically manages virtual environments:

```bash
# UV creates and manages virtual environments automatically
# Just sync your project dependencies
uv sync

# Run commands in the virtual environment
uv run python -c "import mlpregression; print(mlpregression.__version__)"

# Activate the virtual environment manually if needed
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### Using venv (Python 3.3+)

```bash
# Create virtual environment
python -m venv mlp_env

# Activate virtual environment
# On Windows:
mlp_env\Scripts\activate
# On macOS/Linux:
source mlp_env/bin/activate

# Install mlpregression
pip install mlpregression

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create conda environment
conda create -n mlp_env python=3.11
conda activate mlp_env

# Install mlpregression
pip install mlpregression

# Deactivate when done
conda deactivate
```

### Using pipenv

```bash
# Create Pipfile and install
pipenv install mlpregression

# Activate shell
pipenv shell
```

## Docker Installation

### Pre-built Image

Pull and run the pre-built Docker image:

```bash
# Pull the image
docker pull jimhendricks/mlpregression:latest

# Run the container
docker run -p 5002:5002 jimhendricks/mlpregression:latest
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/jimhendricks/mlpregression.git
cd mlpregression

# Build Docker image
docker build -t mlpregression:local .

# Run container
docker run -p 5002:5002 mlpregression:local
```

### Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  mlpregression:
    image: jimhendricks/mlpregression:latest
    ports:
      - "5002:5002"
    environment:
      - FLASK_HOST=0.0.0.0
      - FLASK_PORT=5002
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with:

```bash
docker-compose up -d
```

## Development Installation

For contributors and developers:

### Full Development Setup

```bash
# Clone repository
git clone https://github.com/jimhendricks/mlpregression.git
cd mlpregression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,plotting,docs]"

# Install pre-commit hooks (optional)
pre-commit install
```

### Development Dependencies

The development installation includes:

- **Testing**: pytest, pytest-cov, pytest-flask
- **Code Quality**: black, isort, flake8, mypy
- **Build Tools**: build, twine, setuptools, wheel
- **Documentation**: sphinx, sphinx-rtd-theme
- **Jupyter**: jupyter, ipykernel for examples

### Editable Installation

For active development, install in editable mode:

```bash
pip install -e .
```

This allows you to modify the source code and see changes immediately without reinstalling.

## Verification

### Basic Verification

Test that the installation works:

```python
import mlpregression
print(mlpregression.__version__)
```

### Model Loading Test

```python
from mlpregression import create_model

# Create model
model = create_model()
print("Model created successfully!")
print(f"Model has {model.count_params()} parameters")
```

### API Server Test

Start the server and test:

```bash
# Start server
python -m mlpregression.server

# In another terminal, test the API
curl http://localhost:5002/health
```

Expected response:
```json
{"status": "healthy", "model_loaded": true}
```

### Full Integration Test

```python
import numpy as np
from mlpregression import create_model, validate_input, format_prediction

# Create model
model = create_model()

# Test prediction pipeline
test_input = "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"
processed = validate_input(test_input)
prediction = model.predict(processed, verbose=0)
result = format_prediction(prediction)

print(f"Test prediction: ${result:.2f}k")
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'tensorflow'

**Solution**: Install TensorFlow explicitly:
```bash
pip install tensorflow>=2.15.0
```

#### ModuleNotFoundError: No module named 'mlpregression'

**Solutions**:
1. Ensure you're in the correct virtual environment
2. Reinstall the package: `pip install --force-reinstall mlpregression`
3. Check Python path: `python -c "import sys; print(sys.path)"`

#### Permission Denied (Windows)

**Solution**: Run command prompt as administrator or use:
```bash
pip install --user mlpregression
```

#### SSL Certificate Error

**Solution**: Upgrade pip and certificates:
```bash
pip install --upgrade pip
pip install --upgrade certifi
```

#### Memory Error During Installation

**Solutions**:
1. Close other applications to free memory
2. Use pip's no-cache option: `pip install --no-cache-dir mlpregression`
3. Install dependencies separately

#### Docker Issues

**Port Already in Use**:
```bash
# Find process using port 5002
lsof -i :5002
# Kill process or use different port
docker run -p 5003:5002 mlpregression:latest
```

**Permission Denied (Linux)**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

### Platform-Specific Issues

#### macOS

**Apple Silicon (M1/M2)**:
```bash
# Use conda for better compatibility
conda install tensorflow
pip install mlpregression
```

#### Windows

**Long Path Issues**:
Enable long paths in Windows or use shorter directory names.

**Visual C++ Build Tools**:
Install Microsoft Visual C++ Build Tools if compilation fails.

#### Linux

**Missing System Dependencies**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc gcc-c++
```

### Getting Help

If you encounter issues not covered here:

1. **Check GitHub Issues**: [mlpregression/issues](https://github.com/jimhendricks/mlpregression/issues)
2. **Create New Issue**: Include your OS, Python version, and error message
3. **Email Support**: jhendric98@gmail.com
4. **Stack Overflow**: Tag questions with `mlpregression` and `python`

### Environment Information

To help with troubleshooting, gather environment information:

```python
import sys
import platform
import tensorflow as tf
import mlpregression

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"TensorFlow: {tf.__version__}")
print(f"mlpregression: {mlpregression.__version__}")
```

## Next Steps

After successful installation:

1. **Read the [Usage Guide](usage.md)** for examples and tutorials
2. **Check the [API Reference](api.md)** for detailed documentation
3. **Explore [Examples](../examples/)** for practical use cases
4. **Join the Community** by starring the repository and following updates
