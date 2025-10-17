# mlpregression

[![Python Version](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/badge/pypi-2.1.0-brightgreen.svg)](https://pypi.org/project/mlpregression/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Package Manager: UV](https://img.shields.io/badge/package%20manager-uv-6B73FF.svg)](https://github.com/astral-sh/uv)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.1-FF6F00.svg?logo=tensorflow)](https://tensorflow.org)

**MLP Regression Model for Boston Housing Price Prediction**

A professional-grade neural network implementation using TensorFlow/Keras for predicting median home values in the Boston area. This package includes a trained Multi-Layer Perceptron (MLP) model, a REST API server, and comprehensive utilities for housing price predictions.

---

## 🌟 Features

- **Modern Neural Network**: MLP architecture with customizable layers and hyperparameters
- **Pre-trained Model**: Ready-to-use model trained on Boston Housing dataset
- **REST API**: Production-ready Flask server with comprehensive endpoints
- **Type Hints**: Fully typed codebase for better IDE support and fewer bugs
- **Docker Support**: Containerized deployment for scalability
- **Comprehensive Testing**: Unit tests with pytest
- **Well Documented**: Extensive documentation and examples

---

## 📊 Dataset Information

The model predicts median home values based on 13 features:

| Feature | Description |
|---------|-------------|
| **CRIM** | Per capita crime rate by town |
| **ZN** | Proportion of residential land zoned for lots over 25,000 sq.ft. |
| **INDUS** | Proportion of non-retail business acres per town |
| **CHAS** | Charles River dummy variable (1 if bounds river; 0 otherwise) |
| **NOX** | Nitric oxides concentration (parts per 10 million) |
| **RM** | Average number of rooms per dwelling |
| **AGE** | Proportion of owner-occupied units built prior to 1940 |
| **DIS** | Weighted distances to five Boston employment centres |
| **RAD** | Index of accessibility to radial highways |
| **TAX** | Full-value property-tax rate per $10,000 |
| **PTRATIO** | Pupil-teacher ratio by town |
| **B** | 1000(Bk - 0.63)^2 where Bk is the proportion of African Americans |
| **LSTAT** | Percent lower status of the population |

**Target**: Median value of owner-occupied homes in $1000's

---

## 🚀 Installation

### Using UV (recommended)

```bash
# Install UV first
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install mlpregression
uv add mlpregression
```

### From PyPI with pip

```bash
pip install mlpregression
```

### From Source with UV

```bash
git clone https://github.com/jimhendricks/mlpregression.git
cd mlpregression
uv sync
```

### With Development Dependencies

```bash
# Using UV
uv sync --dev

# Using pip
pip install -e ".[dev]"
```

### Docker

```bash
docker build -t mlpregression .
docker run -p 5002:5002 mlpregression
```

---

## 📖 Quick Start

### Python API

```python
import numpy as np
from mlpregression import create_model, validate_input, format_prediction

# Create and load pre-trained model
model = create_model()
model.load_weights("models/model.h5")

# Example: Predict home value
features = "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"
processed_input = validate_input(features)
prediction = model.predict(processed_input)
price = format_prediction(prediction)

print(f"Predicted home value: ${price:.2f}k")
# Output: Predicted home value: $18.40k
```

### REST API Server

Start the server:

```bash
# Using Python
python -m mlpregression.server

# Or using the console script
mlpregression-server

# With custom configuration
FLASK_HOST=0.0.0.0 FLASK_PORT=8080 python -m mlpregression.server
```

Make predictions via HTTP:

```bash
# Using curl
curl -X POST http://localhost:5002/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"}'

# Response
{
  "prediction": 18.4,
  "unit": "thousands of dollars",
  "success": true
}
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/health` | GET | Health check for container orchestration |
| `/api/features` | GET | Feature names and descriptions |
| `/api/predict` | POST | Make housing price predictions |

---

## 🏗️ Project Structure

```
mlpregression/
├── mlpregression/           # Main package
│   ├── __init__.py         # Package initialization
│   ├── __version__.py      # Version information
│   ├── model.py            # Neural network model definition
│   ├── server.py           # Flask API server
│   └── utils.py            # Utility functions
├── tests/                   # Unit tests
│   ├── __init__.py
│   └── test_model.py
├── examples/                # Usage examples
│   └── demo.ipynb          # Jupyter notebook demo
├── docs/                    # Documentation
│   ├── installation.md
│   ├── usage.md
│   └── api.md
├── models/                  # Pre-trained models
│   └── model.h5
├── .github/                 # GitHub templates
│   └── ISSUE_TEMPLATE/
├── Dockerfile              # Docker configuration
├── pyproject.toml          # Package configuration
├── setup.py                # Setup script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

---

## 🔧 Advanced Usage

### Custom Model Configuration

```python
from mlpregression import create_model

# Create custom model
model = create_model(
    input_dim=13,
    hidden_units_1=100,
    hidden_units_2=20,
    activation="tanh",
    optimizer="sgd",
    learning_rate=0.01
)

# Train on your data
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)
```

### Environment Variables

Configure the server using environment variables:

```bash
export MODEL_PATH=/path/to/model.h5
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5002
export FLASK_DEBUG=false
```

### Docker Deployment

```bash
# Build image
docker build -t mlpregression:latest .

# Run container
docker run -d \
  -p 5002:5002 \
  -e FLASK_HOST=0.0.0.0 \
  -e FLASK_PORT=5002 \
  --name mlp-server \
  mlpregression:latest

# Health check
curl http://localhost:5002/health
```

### Kubernetes Deployment

Deploy multiple instances behind a load balancer:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlpregression
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlpregression
  template:
    metadata:
      labels:
        app: mlpregression
    spec:
      containers:
      - name: mlpregression
        image: mlpregression:latest
        ports:
        - containerPort: 5002
        livenessProbe:
          httpGet:
            path: /health
            port: 5002
        readinessProbe:
          httpGet:
            path: /health
            port: 5002
```

---

## 🧪 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/jimhendricks/mlpregression.git
cd mlpregression

# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates virtual environment automatically)
uv sync --dev
```

### Running Tests

```bash
# Run all tests with UV
uv run pytest

# With coverage report
uv run pytest --cov=mlpregression --cov-report=html

# Run specific test file
uv run pytest tests/test_model.py -v
```

### Code Quality

```bash
# Format and lint code with ruff
uv run ruff format .
uv run ruff check --fix .

# Type checking
uv run mypy mlpregression
```

### Building Distribution

```bash
# Build package with UV
uv build

# Upload to PyPI (test)
uv run twine upload --repository testpypi dist/*

# Upload to PyPI
uv run twine upload dist/*
```

---

## 📚 Documentation

- **[Installation Guide](docs/installation.md)** - Detailed installation instructions
- **[Usage Guide](docs/usage.md)** - Comprehensive usage examples
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines
- **[Changelog](CHANGELOG.md)** - Version history

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code:
- Passes all tests
- Follows code style guidelines (black, isort, flake8)
- Includes appropriate documentation
- Adds tests for new functionality

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Boston Housing dataset from the UCI Machine Learning Repository
- **Framework**: TensorFlow/Keras team for the excellent deep learning framework
- **Community**: All contributors and users of this package

---

## 📮 Contact

**Jim Hendricks** - jhendric98@gmail.com

Project Link: [https://github.com/jimhendricks/mlpregression](https://github.com/jimhendricks/mlpregression)

---

## 📊 Citation

If you use this package in your research, please cite:

```bibtex
@software{mlpregression2024,
  author = {Hendricks, Jim},
  title = {mlpregression: MLP Regression for Boston Housing Prediction},
  year = {2024},
  url = {https://github.com/jimhendricks/mlpregression},
  version = {2.0.0}
}
```

---

## ⚠️ Note on Dataset

The Boston Housing dataset contains features that may reflect historical biases and socioeconomic patterns from the 1970s. This implementation is provided for educational and research purposes. When deploying models in production, consider:

- Potential biases in historical training data
- Ethical implications of automated valuation
- Regular model updates with current data
- Compliance with fair housing regulations

---

**Made with ❤️ by Jim Hendricks**
