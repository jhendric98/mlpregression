# Model definition function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, Adam, SGD, Nadam


def def_model():
    # Model definition
    model = Sequential()
    model.add(Dense(50, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model