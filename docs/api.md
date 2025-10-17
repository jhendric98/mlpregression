# API Reference

Complete API documentation for the mlpregression package.

## Table of Contents

- [Model Functions](#model-functions)
- [Utility Functions](#utility-functions)
- [REST API Endpoints](#rest-api-endpoints)
- [Type Definitions](#type-definitions)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Model Functions

### `create_model()`

Create an MLP regression model for Boston housing price prediction.

```python
def create_model(
    input_dim: int = 13,
    hidden_units_1: int = 50,
    hidden_units_2: int = 10,
    activation: str = "relu",
    optimizer: str = "adam",
    learning_rate: Optional[float] = None,
) -> keras.Model
```

**Parameters:**

- `input_dim` (int, optional): Number of input features. Default: 13
- `hidden_units_1` (int, optional): Units in first hidden layer. Default: 50
- `hidden_units_2` (int, optional): Units in second hidden layer. Default: 10
- `activation` (str, optional): Activation function for hidden layers. Default: "relu"
- `optimizer` (str, optional): Optimizer for training. Default: "adam"
- `learning_rate` (float, optional): Learning rate for optimizer. Default: None (uses optimizer default)

**Returns:**

- `keras.Model`: Compiled Keras model ready for training or inference

**Supported Optimizers:**

- `"adam"`: Adam optimizer (default)
- `"sgd"`: Stochastic Gradient Descent
- `"rmsprop"`: RMSprop optimizer
- `"nadam"`: Nadam optimizer

**Example:**

```python
from mlpregression import create_model

# Default model
model = create_model()

# Custom model
model = create_model(
    hidden_units_1=100,
    hidden_units_2=20,
    activation="tanh",
    optimizer="sgd",
    learning_rate=0.01
)
```

### `def_model()` (Deprecated)

Legacy function for backward compatibility.

```python
def def_model() -> keras.Model
```

**Returns:**

- `keras.Model`: Compiled Keras model with default parameters

**Note:** This function is deprecated. Use `create_model()` instead.

## Utility Functions

### `validate_input()`

Validate and preprocess input data for model prediction.

```python
def validate_input(
    data: Union[List[float], npt.NDArray[np.float64], str],
    expected_features: int = 13,
) -> npt.NDArray[np.float64]
```

**Parameters:**

- `data`: Input data as list, numpy array, or comma-separated string
- `expected_features` (int, optional): Expected number of features. Default: 13

**Returns:**

- `numpy.ndarray`: Validated array reshaped for model input (1, expected_features)

**Raises:**

- `ValueError`: If input format is invalid or has wrong number of features

**Example:**

```python
from mlpregression import validate_input

# From string
x = validate_input("1.2,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72")

# From list
x = validate_input([1.2, 0.0, 8.14, 0.0, 0.538, 6.142, 91.7, 3.98, 4.0, 307.0, 21.0, 396.9, 18.72])

# From numpy array
import numpy as np
x = validate_input(np.array([1.2, 0.0, 8.14, 0.0, 0.538, 6.142, 91.7, 3.98, 4.0, 307.0, 21.0, 396.9, 18.72]))
```

### `format_prediction()`

Format model prediction output.

```python
def format_prediction(prediction: Union[float, npt.NDArray]) -> float
```

**Parameters:**

- `prediction`: Raw model output (numpy array or float)

**Returns:**

- `float`: Formatted prediction as float (in thousands of dollars)

**Example:**

```python
from mlpregression import format_prediction

result = model.predict(x)
formatted = format_prediction(result)
print(f"Predicted home value: ${formatted:.2f}k")
```

### `get_feature_names()`

Get the list of feature names for Boston housing dataset.

```python
def get_feature_names() -> List[str]
```

**Returns:**

- `List[str]`: List of 13 feature names in order

**Example:**

```python
from mlpregression import get_feature_names

features = get_feature_names()
print(f"Model expects {len(features)} features: {', '.join(features)}")
```

**Feature Names:**

1. `CRIM` - Per capita crime rate by town
2. `ZN` - Proportion of residential land zoned for lots over 25,000 sq.ft.
3. `INDUS` - Proportion of non-retail business acres per town
4. `CHAS` - Charles River dummy variable
5. `NOX` - Nitric oxides concentration
6. `RM` - Average number of rooms per dwelling
7. `AGE` - Proportion of owner-occupied units built prior to 1940
8. `DIS` - Weighted distances to five Boston employment centres
9. `RAD` - Index of accessibility to radial highways
10. `TAX` - Full-value property-tax rate per $10,000
11. `PTRATIO` - Pupil-teacher ratio by town
12. `B` - Proportion of African Americans by town (adjusted)
13. `LSTAT` - Percent lower status of the population

### `get_feature_descriptions()`

Get detailed descriptions of all features.

```python
def get_feature_descriptions() -> Dict[str, str]
```

**Returns:**

- `Dict[str, str]`: Dictionary mapping feature names to their descriptions

**Example:**

```python
from mlpregression import get_feature_descriptions

descriptions = get_feature_descriptions()
for feature, desc in descriptions.items():
    print(f"{feature}: {desc}")
```

## REST API Endpoints

### Root Endpoint

**Endpoint:** `GET /`

**Description:** Returns API information and available endpoints.

**Response:**

```json
{
  "name": "Boston Housing Price Predictor API",
  "version": "2.0.0",
  "description": "MLP regression model for predicting Boston area home values",
  "endpoints": {
    "/": "This help message",
    "/health": "Health check endpoint",
    "/api/predict": "POST endpoint for predictions",
    "/api/features": "GET endpoint for feature information"
  },
  "status": "running"
}
```

### Health Check

**Endpoint:** `GET /health`

**Description:** Health check endpoint for container orchestration.

**Response (Healthy):**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Response (Unhealthy):**

```json
{
  "status": "unhealthy",
  "reason": "Model not loaded"
}
```

**Status Codes:**

- `200`: Service is healthy
- `503`: Service is unhealthy

### Feature Information

**Endpoint:** `GET /api/features`

**Description:** Get information about expected input features.

**Response:**

```json
{
  "features": ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"],
  "descriptions": {
    "CRIM": "Per capita crime rate by town",
    "ZN": "Proportion of residential land zoned for lots over 25,000 sq.ft.",
    ...
  },
  "count": 13,
  "format": "Comma-separated values or JSON array"
}
```

### Prediction

**Endpoint:** `POST /api/predict`

**Description:** Make housing price predictions.

**Request Body (JSON):**

```json
{
  "input": "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"
}
```

**Alternative Request Body (Array):**

```json
{
  "input": [1.23, 0.0, 8.14, 0.0, 0.538, 6.142, 91.7, 3.98, 4.0, 307.0, 21.0, 396.9, 18.72]
}
```

**Response (Success):**

```json
{
  "prediction": 18.4,
  "unit": "thousands of dollars",
  "success": true
}
```

**Response (Error):**

```json
{
  "error": "Invalid input",
  "message": "Expected 13 features, got 12",
  "expected_format": "13 comma-separated numeric values or array"
}
```

**Status Codes:**

- `200`: Prediction successful
- `400`: Invalid input
- `500`: Server error
- `503`: Model not loaded

**cURL Examples:**

```bash
# JSON input
curl -X POST http://localhost:5002/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"}'

# Array input
curl -X POST http://localhost:5002/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [1.23, 0.0, 8.14, 0.0, 0.538, 6.142, 91.7, 3.98, 4.0, 307.0, 21.0, 396.9, 18.72]}'

# Form data (legacy)
curl -X POST http://localhost:5002/api/predict \
  -d "input=1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"
```

## Type Definitions

### Input Types

```python
from typing import Union, List
import numpy.typing as npt
import numpy as np

# Valid input types for validate_input()
InputType = Union[List[float], npt.NDArray[np.float64], str]

# Model prediction output
PredictionType = Union[float, npt.NDArray[np.float64]]
```

### Feature Data

```python
# Feature names type
FeatureNames = List[str]

# Feature descriptions type
FeatureDescriptions = Dict[str, str]
```

## Error Handling

### Common Exceptions

#### `ValueError`

Raised by `validate_input()` when:

- Input has wrong number of features
- String input cannot be parsed as numbers
- Input contains non-numeric values

```python
try:
    processed = validate_input("invalid,input")
except ValueError as e:
    print(f"Input validation failed: {e}")
```

#### `ImportError`

Raised when required dependencies are missing:

```python
try:
    from mlpregression import create_model
except ImportError as e:
    print(f"Missing dependency: {e}")
```

### API Error Responses

#### 400 Bad Request

```json
{
  "error": "Invalid input",
  "message": "Expected 13 features, got 12",
  "expected_format": "13 comma-separated numeric values or array"
}
```

#### 500 Internal Server Error

```json
{
  "error": "Prediction failed",
  "message": "Model prediction error: ..."
}
```

#### 503 Service Unavailable

```json
{
  "error": "Model not loaded",
  "message": "Service is starting up"
}
```

## Examples

### Complete Prediction Pipeline

```python
import numpy as np
from mlpregression import create_model, validate_input, format_prediction

def predict_house_price(features_str: str) -> float:
    """Complete prediction pipeline with error handling."""
    try:
        # Create and load model
        model = create_model()
        model.load_weights("models/model.h5")

        # Validate input
        processed_input = validate_input(features_str)

        # Make prediction
        raw_prediction = model.predict(processed_input, verbose=0)

        # Format output
        price = format_prediction(raw_prediction)

        return price

    except ValueError as e:
        print(f"Input validation error: {e}")
        return None
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# Usage
features = "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"
price = predict_house_price(features)
if price is not None:
    print(f"Predicted price: ${price:.2f}k")
```

### Batch Processing

```python
from mlpregression import create_model, validate_input, format_prediction
import numpy as np

def batch_predict(model, input_list):
    """Process multiple predictions efficiently."""
    predictions = []

    for input_data in input_list:
        try:
            processed = validate_input(input_data)
            pred = model.predict(processed, verbose=0)
            price = format_prediction(pred)
            predictions.append(price)
        except Exception as e:
            print(f"Error processing input {input_data}: {e}")
            predictions.append(None)

    return predictions

# Usage
model = create_model()
model.load_weights("models/model.h5")

inputs = [
    "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72",
    "0.02,95.0,2.68,0.0,0.416,6.552,100.0,2.36,3.0,157.0,20.2,396.9,7.60"
]

results = batch_predict(model, inputs)
for i, result in enumerate(results):
    if result is not None:
        print(f"Input {i+1}: ${result:.2f}k")
```

### Custom Model Training

```python
from mlpregression import create_model
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.callbacks import EarlyStopping

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

# Train with early stopping
early_stop = EarlyStopping(patience=50, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.3,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
test_loss = model.evaluate(x_test, y_test, verbose=0)
print(f"Test MSE: {test_loss}")

# Save model
model.save_weights("custom_model.h5")
```

### API Client

```python
import requests
import json

class MLPRegressionClient:
    """Client for mlpregression REST API."""

    def __init__(self, base_url: str = "http://localhost:5002"):
        self.base_url = base_url

    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_features(self) -> dict:
        """Get feature information."""
        response = requests.get(f"{self.base_url}/api/features")
        response.raise_for_status()
        return response.json()

    def predict(self, features) -> float:
        """Make prediction."""
        payload = {"input": features}
        response = requests.post(
            f"{self.base_url}/api/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()["prediction"]

# Usage
client = MLPRegressionClient()

if client.health_check():
    features = "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"
    price = client.predict(features)
    print(f"Predicted price: ${price:.2f}k")
else:
    print("API is not available")
```

## Version Information

Access version information programmatically:

```python
import mlpregression

print(f"Version: {mlpregression.__version__}")
print(f"Author: {mlpregression.__author__}")
print(f"Email: {mlpregression.__email__}")
print(f"Description: {mlpregression.__description__}")
```

For more examples and tutorials, see the [Usage Guide](usage.md) and [Examples](../examples/) directory.
