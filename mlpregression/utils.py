"""Utility functions for data preprocessing and validation."""


import numpy as np
import numpy.typing as npt


def validate_input(
    data: list[float] | npt.NDArray[np.float64] | str,
    expected_features: int = 13,
) -> npt.NDArray[np.float64]:
    """
    Validate and preprocess input data for model prediction.

    Args:
        data: Input data as list, numpy array, or comma-separated string
        expected_features: Expected number of features (default: 13)

    Returns:
        Validated numpy array reshaped for model input (1, expected_features)

    Raises:
        ValueError: If input format is invalid or has wrong number of features

    Example:
        >>> # From string
        >>> x = validate_input("1.2,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72")
        >>> # From list
        >>> x = validate_input([1.2, 0.0, 8.14, 0.0, 0.538, 6.142, 91.7, 3.98, 4.0, 307.0, 21.0, 396.9, 18.72])
    """
    # Convert string to array if needed
    if isinstance(data, str):
        try:
            data = np.array([float(x.strip()) for x in data.split(",")])
        except ValueError as e:
            raise ValueError(
                f"Invalid input format. Could not parse string to floats: {e}"
            ) from e

    # Convert to numpy array if list
    if isinstance(data, list):
        data = np.array(data, dtype=np.float64)

    # Validate shape
    if data.shape[0] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, got {data.shape[0]}. "
            f"Boston housing dataset requires 13 features: "
            f"CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT"
        )

    # Reshape for model input (batch_size, features)
    return data.reshape(1, expected_features)


def format_prediction(prediction: float | npt.NDArray) -> float:
    """
    Format model prediction output.

    Args:
        prediction: Raw model output (numpy array or float)

    Returns:
        Formatted prediction as float (in thousands of dollars)

    Example:
        >>> result = model.predict(x)
        >>> formatted = format_prediction(result)
        >>> print(f"Predicted home value: ${formatted:.2f}k")
    """
    if isinstance(prediction, np.ndarray):
        return float(prediction.flatten()[0])
    return float(prediction)


def get_feature_names() -> list[str]:
    """
    Get the list of feature names for Boston housing dataset.

    Returns:
        List of 13 feature names in order

    Example:
        >>> features = get_feature_names()
        >>> print(f"Model expects {len(features)} features: {', '.join(features)}")
    """
    return [
        "CRIM",  # per capita crime rate by town
        "ZN",  # proportion of residential land zoned for lots over 25,000 sq.ft.
        "INDUS",  # proportion of non-retail business acres per town
        "CHAS",  # Charles River dummy variable (1 if tract bounds river; 0 otherwise)
        "NOX",  # nitric oxides concentration (parts per 10 million)
        "RM",  # average number of rooms per dwelling
        "AGE",  # proportion of owner-occupied units built prior to 1940
        "DIS",  # weighted distances to five Boston employment centres
        "RAD",  # index of accessibility to radial highways
        "TAX",  # full-value property-tax rate per $10,000
        "PTRATIO",  # pupil-teacher ratio by town
        "B",  # 1000(Bk - 0.63)^2 where Bk is the proportion of African Americans
        "LSTAT",  # percent lower status of the population
    ]


def get_feature_descriptions() -> dict:
    """
    Get detailed descriptions of all features.

    Returns:
        Dictionary mapping feature names to their descriptions
    """
    return {
        "CRIM": "Per capita crime rate by town",
        "ZN": "Proportion of residential land zoned for lots over 25,000 sq.ft.",
        "INDUS": "Proportion of non-retail business acres per town",
        "CHAS": "Charles River dummy variable (1 if tract bounds river; 0 otherwise)",
        "NOX": "Nitric oxides concentration (parts per 10 million)",
        "RM": "Average number of rooms per dwelling",
        "AGE": "Proportion of owner-occupied units built prior to 1940",
        "DIS": "Weighted distances to five Boston employment centres",
        "RAD": "Index of accessibility to radial highways",
        "TAX": "Full-value property-tax rate per $10,000",
        "PTRATIO": "Pupil-teacher ratio by town",
        "B": "1000(Bk - 0.63)^2 where Bk is the proportion of African Americans by town",
        "LSTAT": "Percent lower status of the population",
    }
