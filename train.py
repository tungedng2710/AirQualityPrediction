'''
Script running is in progress
Please check the Quickstart notebook for instant use
'''

from utils.dataset import AI4VN_dataloader
import tensorflow as tf
import wandb
import pandas as pd
import numpy as np
wandb.init(project="visualize-tensorflow")


if __name__ == "__main__":
    WINDOW_SIZE = 7*24
    HORIZON = 24
    distances = pd.read_csv("exp/in_out_location.csv").to_numpy()[:, 1:].astype(np.float32).T
    distances_normalize = distances/np.max(distances)
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
    N_EPOCHS = 10
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    tf.random.set_seed(42)

    # Let's build an LSTM model with the Functional API
    inputs = tf.keras.layers.Input(shape=(WINDOW_SIZE, 3*11))
    # x = layers.LSTM(128, activation="relu", return_sequences=True)(inputs) # this layer will error if the inputs are not the right shape
    x = tf.keras.layers.LSTM(256, activation="relu")(inputs)  # using the tanh loss function results in a massive error

    # Add another optional dense layer (you could add more of these to see if they improve model performance)
    # x = layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(HORIZON*11, activation="relu")(x)
    x = tf.keras.layers.Reshape((HORIZON, 11))(x)
    output = tf.keras.layers.Dot(axes=(2, 1))([x, distances_tensor])
    model = tf.keras.Model(inputs=inputs, outputs=output, name="model_lstm")

    # Compile model
    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())
    model.summary()

    # fit model
    model.fit(train_dataset,
              epochs=N_EPOCHS,
              validation_data=test_dataset,
              verbose=1,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])


    # eval_regression_model(y_test, y_test_pred)

