'''
Script running is in progress
Please check the Quickstart notebook for instant use
'''

import os
from utils.dataset import AI4VN_dataloader
from utils.evaluation import eval_regression_model
import tensorflow as tf
import wandb
import pandas as pd
import numpy as np
wandb.init(project="visualize-tensorflow")


# Create a function to implement a ModelCheckpoint callback with a specific filename
def create_model_checkpoint(model_name, save_path="trained_models"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                              verbose=0,  # only output a limited amount of text
                                              save_best_only=True)  # save only the best model to file

if __name__ == "__main__":
    WINDOW_SIZE = 7*24
    HORIZON = 24
    distances = pd.read_csv("dataset/exp/in_out_location.csv").to_numpy()[:, 1:].astype(np.float32).T
    distances_normalize = 1/(distances/np.min(distances))
    distances_tensor = tf.expand_dims(tf.convert_to_tensor(distances_normalize), axis=0)
    X_train, X_test, y_train, y_test = AI4VN_dataloader(test_split=0.2)

    # 1. Turn train and test arrays into tensor Datasets
    train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train.tolist())
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train.tolist())

    test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test.tolist())
    test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test.tolist())

    # 2. Combine features & labels
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
    test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

    # 3. Batch and prefetch for optimal performance
    BATCH_SIZE = 128  # taken from Appendix D in N-BEATS paper
    N_EPOCHS = 1000
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    tf.random.set_seed(42)

    # Let's build an LSTM model with the Functional API
    inputs = tf.keras.layers.Input(shape=(WINDOW_SIZE, 3*11))
    # x = tf.keras.layers.LSTM(128, activation="relu", return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(256, activation="tanh", )(inputs)
    x = tf.keras.layers.Dense(HORIZON*11, activation="relu")(x)
    x = tf.keras.layers.Reshape((HORIZON, 11))(x)
    output = tf.keras.layers.Dot(axes=(2, 1), name='Dot')([x, distances_tensor])
    # output = tf.keras.layers.Dense(4, activation="relu")(x)
    model = tf.keras.Model(inputs=inputs, outputs=output, name="model_lstm_2")

    # Compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["mae"])
    model.summary()

    # fit model
    model.fit(train_dataset,
              epochs=N_EPOCHS,
              validation_data=test_dataset,
              verbose=1,
              callbacks=[create_model_checkpoint(model_name=model.name),
                         # tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])


    # eval_regression_model(y_test, y_test_pred)
    y_preds = model.predict(test_dataset)
    y_test = np.reshape(y_test, (y_test.shape[0], -1))
    y_preds = np.reshape(y_preds, (y_preds.shape[0], -1))
    eval_regression_model(y_test, y_preds)

