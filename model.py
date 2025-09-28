"""Utility helpers for constructing the regression model architecture.

The `server.py` module and accompanying notebook import :func:`def_model` to
instantiate a TensorFlow/Keras ``Sequential`` model. Keeping the architecture in
one place ensures the saved weights in ``model.h5`` always align with the
in-memory model used by the Flask API.
"""

from __future__ import annotations

from tensorflow import keras


def def_model() -> keras.Model:
    """Build and compile the MLP regressor used throughout the project.

    Returns
    -------
    keras.Model
        A ``Sequential`` model with two hidden dense layers (50 and 10 units) and
        a single linear output neuron. The model is compiled with mean squared
        error as both the loss function and metric, and uses the Adam optimiser.
    """

    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            50,
            input_dim=13,
            kernel_initializer="he_normal",
            activation="relu",
        )
    )
    model.add(
        keras.layers.Dense(
            10,
            kernel_initializer="he_normal",
            activation="relu",
        )
    )
    model.add(keras.layers.Dense(1, kernel_initializer="he_normal"))
    model.compile(loss="mse", optimizer="adam", metrics=["mse"])
    return model
