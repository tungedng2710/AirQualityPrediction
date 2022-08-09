import tensorflow as tf
import pandas as pd
import numpy as np

tf.random.set_seed(42)

def create_model(WINDOW_SIZE: int=168,
                 HORIZON: int=24,
                 name: str=''):

    distances = pd.read_csv("dataset/exp/in_out_location.csv").to_numpy()[:, 1:].astype(np.float32).T
    distances_normalize = 1 / (distances / np.min(distances))
    distances_tensor = tf.constant(np.expand_dims(np.expand_dims(distances_normalize, axis=0), axis=0))

    # Let's build an LSTM model with the Functional API
    inputs = tf.keras.layers.Input(shape=(WINDOW_SIZE, 3 * 11))
    # x = tf.keras.layers.LSTM(128, activation="tanh", return_sequences=True)(inputs)
    x = tf.keras.layers.GRU(128, activation="tanh", name='GRU_1')(inputs)
    # x = tf.keras.layers.Reshape((-1), name='reshape_1')(x)
    x = tf.keras.layers.Dense(HORIZON * 11, activation="relu", name='Dense_1')(x)
    x = tf.keras.layers.Reshape((HORIZON, 11), name='reshape_2')(x)
    # output = tf.keras.layers.Dot(axes=(2, 1), name='Dot')([x, distances_tensor])
    output = tf.keras.layers.Dense(4, activation="relu", name='Dense_2')(x)
    model = tf.keras.Model(inputs=inputs, outputs=output, name=name)

    # inputs = tf.keras.layers.Input(shape=(WINDOW_SIZE, 11, 3))
    # x = tf.keras.layers.Dot(axes=(2, 2), name='Dot')([inputs, distances_tensor])
    # # x = tf.keras.layers.Permute((4, 2, 3, 1), name='permute_1')(x)
    # x = tf.keras.layers.Reshape((WINDOW_SIZE, -1), name='reshape_1')(x)
    # # x = tf.keras.layers.LSTM(128, activation="tanh", return_sequences=True, name='lstm_1')(x)
    # x = tf.keras.layers.LSTM(256, activation="tanh", name='lstm_2')(x)
    # # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(HORIZON*11, activation="relu", name='Dense_1')(x)
    # # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Reshape((HORIZON, 11), name='reshape_2')(x)
    # output = tf.keras.layers.Dense(4, activation="relu", name='Dense_2')(x)
    # # output = tf.keras.layers.Permute((2, 1), name='permute_2')(x)
    # model = tf.keras.Model(inputs=inputs, outputs=output, name=name)

    # inputs = tf.keras.layers.Input(shape=(WINDOW_SIZE, 11, 3))
    # x = tf.keras.layers.Dot(axes=(2, 2), name='Dot')([inputs, distances_tensor])
    # # x = tf.keras.layers.Permute((4, 2, 3, 1), name='permute_1')(x)
    # x = tf.keras.layers.Reshape((WINDOW_SIZE, -1), name='reshape_1')(x)
    # # x = tf.keras.layers.LayerNormalization(axis=1, name='normalize')(x)
    # x = tf.keras.layers.LSTM(12, activation="tanh", return_sequences=True, name='lstm_1')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.LSTM(12, activation="tanh", return_sequences=True, name='lstm_2')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.LSTM(12, activation="tanh", name='lstm_3')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(HORIZON*11, activation="relu", name='Dense_1')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Reshape((HORIZON, 11), name='reshape_2')(x)
    # output = tf.keras.layers.Dense(4, activation="relu", name='Dense_2')(x)
    # # output = tf.keras.layers.Permute((2, 1), name='permute_2')(x)
    # model = tf.keras.Model(inputs=inputs, outputs=output, name=name)

    return model