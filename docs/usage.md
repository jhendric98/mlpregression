# Usage Guide

This comprehensive guide covers all aspects of using mlpregression, from basic predictions to advanced customization and deployment.

## Table of Contents

- [Quick Start](#quick-start)
- [Python API](#python-api)
- [REST API Server](#rest-api-server)
- [Model Customization](#model-customization)
- [Data Preprocessing](#data-preprocessing)
- [Deployment](#deployment)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Quick Start

### Basic Prediction

```python
import numpy as np
from mlpregression import create_model, validate_input, format_prediction

# Create and load pre-trained model
model = create_model()
model.load_weights("models/model.h5")

# Make a prediction
features = "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"
processed_input = validate_input(features)
prediction = model.predict(processed_input, verbose=0)
price = format_prediction(prediction)

print(f"Predicted home value: ${price:.2f}k")
# Output: Predicted home value: $18.40k
```

### Using the REST API

```bash
# Start the server
python -m mlpregression.server

# Make a prediction
curl -X POST http://localhost:5002/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"}'
```

## Python API

### Model Creation

```python
from mlpregression import create_model

# Default model
model = create_model()

# Custom model
model = create_model(
    input_dim=13,
    hidden_units_1=100,
    hidden_units_2=20,
    activation="tanh",
    optimizer="sgd",
    learning_rate=0.01
)
```

### Input Validation

```python
from mlpregression import validate_input

# From comma-separated string
input_str = "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"
processed = validate_input(input_str)

# From list
input_list = [1.23, 0.0, 8.14, 0.0, 0.538, 6.142, 91.7, 3.98, 4.0, 307.0, 21.0, 396.9, 18.72]
processed = validate_input(input_list)

# From numpy array
input_array = np.array([1.23, 0.0, 8.14, 0.0, 0.538, 6.142, 91.7, 3.98, 4.0, 307.0, 21.0, 396.9, 18.72])
processed = validate_input(input_array)
```

### Feature Information

```python
from mlpregression import get_feature_names, get_feature_descriptions

# Get feature names
features = get_feature_names()
print(f"Features: {', '.join(features)}")

# Get detailed descriptions
descriptions = get_feature_descriptions()
for feature, desc in descriptions.items():
    print(f"{feature}: {desc}")
```

### Batch Predictions

```python
import numpy as np
from mlpregression import create_model, validate_input

model = create_model()
model.load_weights("models/model.h5")

# Multiple predictions
inputs = [
    "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72",
    "0.02,95.0,2.68,0.0,0.416,6.552,100.0,2.36,3.0,157.0,20.2,396.9,7.60"
]

predictions = []
for input_data in inputs:
    processed = validate_input(input_data)
    pred = model.predict(processed, verbose=0)
    predictions.append(format_prediction(pred))

print(f"Predictions: {predictions}")
```

## REST API Server

### Starting the Server

```bash
# Default configuration
python -m mlpregression.server

# Custom configuration
FLASK_HOST=0.0.0.0 FLASK_PORT=8080 python -m mlpregression.server

# With debug mode
FLASK_DEBUG=true python -m mlpregression.server
```

### Available Endpoints

#### Root Endpoint (`/`)

```bash
curl http://localhost:5002/
```

Returns API information and available endpoints.

#### Health Check (`/health`)

```bash
curl http://localhost:5002/health
```

Returns server health status for monitoring.

#### Feature Information (`/api/features`)

```bash
curl http://localhost:5002/api/features
```

Returns feature names and descriptions.

#### Prediction (`/api/predict`)

```bash
# JSON input
curl -X POST http://localhost:5002/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"}'

# Array input
curl -X POST http://localhost:5002/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [1.23, 0.0, 8.14, 0.0, 0.538, 6.142, 91.7, 3.98, 4.0, 307.0, 21.0, 396.9, 18.72]}'

# Form data (legacy support)
curl -X POST http://localhost:5002/api/predict \
  -d "input=1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"
```

### Error Handling

The API returns structured error responses:

```json
{
  "error": "Invalid input",
  "message": "Expected 13 features, got 12",
  "expected_format": "13 comma-separated numeric values or array"
}
```

## Model Customization

### Training a Custom Model

```python
from mlpregression import create_model
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Create custom model
model = create_model(
    hidden_units_1=100,
    hidden_units_2=50,
    activation="relu",
    optimizer="adam",
    learning_rate=0.001
)

# Set up callbacks
callbacks = [
    EarlyStopping(patience=50, restore_best_weights=True),
    ModelCheckpoint("my_model.h5", save_best_only=True)
]

# Train model
history = model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.3,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss = model.evaluate(x_test, y_test, verbose=0)
print(f"Test MSE: {test_loss}")
```

### Model Architecture Variations

```python
# Deeper network
deep_model = create_model(
    hidden_units_1=128,
    hidden_units_2=64,
    activation="relu"
)

# Different activation
tanh_model = create_model(
    activation="tanh",
    optimizer="sgd",
    learning_rate=0.01
)

# Different optimizer
nadam_model = create_model(
    optimizer="nadam",
    learning_rate=0.002
)
```

## Data Preprocessing

### Understanding the Features

The Boston Housing dataset includes 13 features:

```python
from mlpregression import get_feature_descriptions

descriptions = get_feature_descriptions()
for feature, desc in descriptions.items():
    print(f"{feature:8}: {desc}")
```

### Feature Scaling

For custom training, consider feature scaling:

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example with custom data
X_raw = np.array([
    [1.23, 0.0, 8.14, 0.0, 0.538, 6.142, 91.7, 3.98, 4.0, 307.0, 21.0, 396.9, 18.72],
    [0.02, 95.0, 2.68, 0.0, 0.416, 6.552, 100.0, 2.36, 3.0, 157.0, 20.2, 396.9, 7.60]
])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Use scaled data for training
model = create_model()
# ... training code ...
```

### Data Validation

```python
from mlpregression import validate_input

def validate_and_predict(model, input_data):
    """Safely validate input and make prediction."""
    try:
        processed = validate_input(input_data)
        prediction = model.predict(processed, verbose=0)
        return format_prediction(prediction)
    except ValueError as e:
        print(f"Validation error: {e}")
        return None
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t mlpregression:prod .

# Run with production settings
docker run -d \
  --name mlp-prod \
  -p 80:5002 \
  -e FLASK_HOST=0.0.0.0 \
  -e FLASK_PORT=5002 \
  -e FLASK_DEBUG=false \
  --restart unless-stopped \
  mlpregression:prod
```

### Kubernetes Deployment

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
        image: mlpregression:prod
        ports:
        - containerPort: 5002
        env:
        - name: FLASK_HOST
          value: "0.0.0.0"
        - name: FLASK_PORT
          value: "5002"
        livenessProbe:
          httpGet:
            path: /health
            port: 5002
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5002
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: mlpregression-service
spec:
  selector:
    app: mlpregression
  ports:
  - port: 80
    targetPort: 5002
  type: LoadBalancer
```

### Environment Variables

Configure the server using environment variables:

```bash
export MODEL_PATH=/path/to/custom/model.h5
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5002
export FLASK_DEBUG=false
```

## Examples

### Web Application Integration

```python
import requests
import json

class HousePricePredictor:
    def __init__(self, api_url="http://localhost:5002"):
        self.api_url = api_url

    def predict(self, features):
        """Make prediction via API."""
        response = requests.post(
            f"{self.api_url}/api/predict",
            json={"input": features},
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            return response.json()["prediction"]
        else:
            raise Exception(f"API Error: {response.json()}")

# Usage
predictor = HousePricePredictor()
price = predictor.predict("1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72")
print(f"Predicted price: ${price:.2f}k")
```

### Batch Processing

```python
import pandas as pd
from mlpregression import create_model, validate_input, format_prediction

# Load model
model = create_model()
model.load_weights("models/model.h5")

# Process CSV file
df = pd.read_csv("housing_data.csv")

predictions = []
for _, row in df.iterrows():
    # Convert row to comma-separated string
    features = ",".join(map(str, row.values))

    try:
        processed = validate_input(features)
        pred = model.predict(processed, verbose=0)
        price = format_prediction(pred)
        predictions.append(price)
    except Exception as e:
        print(f"Error processing row: {e}")
        predictions.append(None)

# Add predictions to dataframe
df["predicted_price"] = predictions
df.to_csv("housing_predictions.csv", index=False)
```

### Model Comparison

```python
from mlpregression import create_model
from tensorflow.keras.datasets import boston_housing
import numpy as np

# Load test data
(_, _), (x_test, y_test) = boston_housing.load_data()

# Compare different models
models = {
    "default": create_model(),
    "deep": create_model(hidden_units_1=100, hidden_units_2=50),
    "tanh": create_model(activation="tanh"),
}

results = {}
for name, model in models.items():
    model.load_weights("models/model.h5")  # Load same weights for comparison
    predictions = model.predict(x_test, verbose=0)
    mse = np.mean((predictions.flatten() - y_test) ** 2)
    results[name] = mse
    print(f"{name:10}: MSE = {mse:.2f}")
```

## Best Practices

### Performance Optimization

1. **Batch Predictions**: Process multiple inputs together
2. **Model Caching**: Load model once and reuse
3. **Input Validation**: Validate inputs before processing
4. **Error Handling**: Implement robust error handling

### Production Deployment

1. **Health Checks**: Always implement health check endpoints
2. **Logging**: Use structured logging for monitoring
3. **Resource Limits**: Set appropriate memory and CPU limits
4. **Scaling**: Use horizontal scaling for high traffic

### Security

1. **Input Validation**: Always validate and sanitize inputs
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **HTTPS**: Use HTTPS in production
4. **Monitoring**: Monitor for unusual usage patterns

### Monitoring

```python
import logging
import time
from functools import wraps

def monitor_predictions(func):
    """Decorator to monitor prediction performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logging.info(f"Prediction successful in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"Prediction failed after {duration:.3f}s: {e}")
            raise
    return wrapper

@monitor_predictions
def make_prediction(model, input_data):
    processed = validate_input(input_data)
    return model.predict(processed, verbose=0)
```

### Testing

```python
import unittest
from mlpregression import create_model, validate_input

class TestPredictions(unittest.TestCase):
    def setUp(self):
        self.model = create_model()
        self.test_input = "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"

    def test_prediction_range(self):
        """Test that predictions are in reasonable range."""
        processed = validate_input(self.test_input)
        prediction = self.model.predict(processed, verbose=0)
        price = format_prediction(prediction)

        # Boston housing prices should be positive and reasonable
        self.assertGreater(price, 0)
        self.assertLess(price, 100)  # Less than $100k (1970s data)

if __name__ == "__main__":
    unittest.main()
```

## Next Steps

- **Explore the [API Reference](api.md)** for detailed function documentation
- **Check out [Examples](../examples/)** for more practical use cases
- **Read the [Contributing Guide](../CONTRIBUTING.md)** to contribute to the project
- **Join the community** by starring the repository and following updates
