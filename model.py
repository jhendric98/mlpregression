"""Model definition utilities for the regression service."""

from __future__ import annotations

from tensorflow import keras


def def_model() -> keras.Model:
    """Create the regression model architecture used by the API."""

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
