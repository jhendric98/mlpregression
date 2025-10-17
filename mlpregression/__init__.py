"""
mlpregression - MLP regression model for Boston housing price prediction.

This package provides a neural network model for predicting median home values
in the Boston area based on 13 input features including crime rate, property tax,
number of rooms, and other socioeconomic factors.
"""

from .__version__ import (
    __author__,
    __description__,
    __email__,
    __version__,
)
from .model import create_model, def_model
from .utils import (
    format_prediction,
    get_feature_descriptions,
    get_feature_names,
    validate_input,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "create_model",
    "def_model",
    "validate_input",
    "format_prediction",
    "get_feature_names",
    "get_feature_descriptions",
]
