import time

from keras.layers.core import Dropout, Dense, Activation
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam


def get_lstm_model(lr=0.001):
    model = Sequential()
    model.add(LSTM(50, input_shape=(None, 1), return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(LSTM(100, return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(LSTM(200, return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(LSTM(300, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(100))
    model.add(Dense(1))

    model.add(Activation('relu'))
    start = time.time()
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam)
    model.summary()
    return model


def get_gru_model(lr=0.001):
    model = Sequential()
    model.add(GRU(50, input_shape=(None, 1), return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(Bidirectional(GRU(100, return_sequences=True)))
    # model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    # model.add(Dropout(0.2))

    model.add(GRU(300, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(200))
    model.add(Dense(1))

    model.add(Activation('relu'))
    start = time.time()
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam)
    model.summary()
    return model


def get_bilstm_model(lr=0.001):
    model = Sequential()
    model.add(LSTM(50, input_shape=(None, 1), return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    # model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    # model.add(Dropout(0.2))

    model.add(LSTM(300, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(200))
    model.add(Dense(1))

    model.add(Activation('relu'))
    start = time.time()
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam)
    model.summary()
    return model
