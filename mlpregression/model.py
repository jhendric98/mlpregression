"""Model definition for MLP regression on Boston housing data."""


from tensorflow import keras
from tensorflow.keras import layers, models, optimizers


def create_model(
    input_dim: int = 13,
    hidden_units_1: int = 50,
    hidden_units_2: int = 10,
    activation: str = "relu",
    optimizer: str = "adam",
    learning_rate: float | None = None,
) -> keras.Model:
    """
    Create an MLP regression model for Boston housing price prediction.

    This function creates a sequential neural network with two hidden layers
    for predicting median home values based on 13 input features.

    Args:
        input_dim: Number of input features (default: 13 for Boston housing dataset)
        hidden_units_1: Number of units in first hidden layer (default: 50)
        hidden_units_2: Number of units in second hidden layer (default: 10)
        activation: Activation function for hidden layers (default: "relu")
        optimizer: Optimizer to use for training (default: "adam")
        learning_rate: Learning rate for optimizer (default: None, uses optimizer default)

    Returns:
        Compiled Keras model ready for training or inference

    Example:
        >>> model = create_model()
        >>> model.summary()
        >>> # Load pre-trained weights
        >>> model.load_weights("path/to/model.h5")
        >>> # Make predictions
        >>> predictions = model.predict(X_test)
    """
    # Create sequential model
    model = models.Sequential(
        [
            layers.Dense(
                hidden_units_1,
                input_dim=input_dim,
                kernel_initializer="normal",
                activation=activation,
                name="dense_1",
            ),
            layers.Dense(
                hidden_units_2,
                kernel_initializer="normal",
                activation=activation,
                name="dense_2",
            ),
            layers.Dense(
                1,
                kernel_initializer="normal",
                name="output",
            ),
        ],
        name="boston_housing_mlp",
    )

    # Configure optimizer
    if optimizer.lower() == "adam":
        opt = (
            optimizers.Adam(learning_rate=learning_rate)
            if learning_rate
            else optimizers.Adam()
        )
    elif optimizer.lower() == "sgd":
        opt = (
            optimizers.SGD(learning_rate=learning_rate)
            if learning_rate
            else optimizers.SGD()
        )
    elif optimizer.lower() == "rmsprop":
        opt = (
            optimizers.RMSprop(learning_rate=learning_rate)
            if learning_rate
            else optimizers.RMSprop()
        )
    elif optimizer.lower() == "nadam":
        opt = (
            optimizers.Nadam(learning_rate=learning_rate)
            if learning_rate
            else optimizers.Nadam()
        )
    else:
        opt = optimizer

    # Compile model
    model.compile(
        loss="mse",
        optimizer=opt,
        metrics=["mse", "mae"],
    )

    return model


# Legacy function name for backward compatibility
def def_model() -> keras.Model:
    """
    Legacy function for backward compatibility.

    Returns:
        Compiled Keras model with default parameters

    Deprecated:
        Use create_model() instead for more configuration options.
    """
    return create_model()
