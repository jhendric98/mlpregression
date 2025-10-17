"""Unit tests for mlpregression model and utilities."""

import numpy as np
import pytest
from tensorflow import keras

from mlpregression.model import create_model, def_model
from mlpregression.utils import (
    format_prediction,
    get_feature_descriptions,
    get_feature_names,
    validate_input,
)


class TestModel:
    """Test cases for model creation and functionality."""

    def test_create_model_default(self):
        """Test model creation with default parameters."""
        model = create_model()
        assert isinstance(model, keras.Model)
        assert len(model.layers) == 3

    def test_create_model_custom(self):
        """Test model creation with custom parameters."""
        model = create_model(
            input_dim=13,
            hidden_units_1=100,
            hidden_units_2=20,
            activation="tanh",
            optimizer="sgd",
        )
        assert isinstance(model, keras.Model)
        assert model.layers[0].units == 100
        assert model.layers[1].units == 20

    def test_def_model_legacy(self):
        """Test legacy model creation function."""
        model = def_model()
        assert isinstance(model, keras.Model)

    def test_model_input_shape(self):
        """Test that model accepts correct input shape."""
        model = create_model()
        test_input = np.random.rand(1, 13)
        prediction = model.predict(test_input, verbose=0)
        assert prediction.shape == (1, 1)

    def test_model_compilation(self):
        """Test that model is properly compiled."""
        model = create_model()
        assert model.optimizer is not None
        assert model.loss == "mse"


class TestUtils:
    """Test cases for utility functions."""

    def test_validate_input_from_string(self):
        """Test input validation from comma-separated string."""
        input_str = "1.2,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"
        result = validate_input(input_str)
        assert result.shape == (1, 13)
        assert result[0, 0] == pytest.approx(1.2)

    def test_validate_input_from_list(self):
        """Test input validation from list."""
        input_list = [
            1.2,
            0.0,
            8.14,
            0.0,
            0.538,
            6.142,
            91.7,
            3.98,
            4.0,
            307.0,
            21.0,
            396.9,
            18.72,
        ]
        result = validate_input(input_list)
        assert result.shape == (1, 13)

    def test_validate_input_from_array(self):
        """Test input validation from numpy array."""
        input_array = np.array(
            [
                1.2,
                0.0,
                8.14,
                0.0,
                0.538,
                6.142,
                91.7,
                3.98,
                4.0,
                307.0,
                21.0,
                396.9,
                18.72,
            ]
        )
        result = validate_input(input_array)
        assert result.shape == (1, 13)

    def test_validate_input_wrong_size(self):
        """Test that validation fails with wrong number of features."""
        with pytest.raises(ValueError, match="Expected 13 features"):
            validate_input([1.0, 2.0, 3.0])

    def test_validate_input_invalid_string(self):
        """Test that validation fails with invalid string format."""
        with pytest.raises(ValueError, match="Invalid input format"):
            validate_input("not,valid,numbers,here")

    def test_format_prediction_from_array(self):
        """Test prediction formatting from numpy array."""
        pred_array = np.array([[25.5]])
        result = format_prediction(pred_array)
        assert isinstance(result, float)
        assert result == pytest.approx(25.5)

    def test_format_prediction_from_float(self):
        """Test prediction formatting from float."""
        result = format_prediction(30.7)
        assert isinstance(result, float)
        assert result == pytest.approx(30.7)

    def test_get_feature_names(self):
        """Test feature names retrieval."""
        features = get_feature_names()
        assert len(features) == 13
        assert "CRIM" in features
        assert "LSTAT" in features

    def test_get_feature_descriptions(self):
        """Test feature descriptions retrieval."""
        descriptions = get_feature_descriptions()
        assert len(descriptions) == 13
        assert "CRIM" in descriptions
        assert isinstance(descriptions["CRIM"], str)


class TestIntegration:
    """Integration tests for model prediction pipeline."""

    def test_end_to_end_prediction(self):
        """Test complete prediction pipeline."""
        # Create and prepare model
        model = create_model()

        # Sample input
        input_str = "1.23,0.0,8.14,0.0,0.538,6.142,91.7,3.98,4.0,307.0,21.0,396.9,18.72"

        # Validate and predict
        processed_input = validate_input(input_str)
        raw_prediction = model.predict(processed_input, verbose=0)
        prediction = format_prediction(raw_prediction)

        # Verify output
        assert isinstance(prediction, float)
        assert prediction > 0  # Home prices should be positive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
