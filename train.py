'''
Script running is in progress
Please check the Quickstart notebook for instant use
'''

import os
from datetime import datetime
from utils.dataset import AI4VN_dataloader
from utils.evaluation import eval_regression_model
import tensorflow as tf
import wandb
from utils.tf_model import create_model
import numpy as np
wandb.init(project="visualize-tensorflow")


# Create a function to implement a ModelCheckpoint callback with a specific filename
def create_model_checkpoint(model_name, save_path="trained_models"):
    outpath = os.path.join(save_path, model_name)
    if os.path.exists(outpath):
        os.mkdir(outpath)
    return tf.keras.callbacks.ModelCheckpoint(filepath=outpath,
                                              verbose=0,  # only output a limited amount of text
                                              monitor='val_loss',
                                              save_weights_only=True,
                                              save_best_only=True)  # save only the best model to file

if __name__ == "__main__":

    BATCH_SIZE = 64  # taken from Appendix D in N-BEATS paper
    N_EPOCHS = 1000
    LR = 0.0005
    name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S') + '_' + str(BATCH_SIZE) + '_' + str(LR)

    WINDOW_SIZE = 7*24
    HORIZON = 24
    X_train, X_test, y_train, y_test = AI4VN_dataloader(root_dir='dataset/exp', test_split=0.2)

    # 1. Turn train and test arrays into tensor Datasets
    train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train.tolist())
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train.tolist())

    test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test.tolist())
    test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test.tolist())

    # 2. Combine features & labels
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
    test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

    # 3. Batch and prefetch for optimal performance
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    tf.random.set_seed(42)

    model = create_model(WINDOW_SIZE=WINDOW_SIZE, HORIZON=HORIZON, name=name)

    # Compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=LR),
                  metrics=["mae"])
    model.summary()

    # fit model
    model.fit(train_dataset,
              epochs=N_EPOCHS,
              validation_data=test_dataset,
              verbose=1,
              callbacks=[create_model_checkpoint(model_name=model.name),
                         # tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=150, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])


    # eval_regression_model(y_test, y_test_pred)
    y_preds = model.predict(test_dataset)
    y_test = np.reshape(y_test, (y_test.shape[0], -1))
    y_preds = np.reshape(y_preds, (y_preds.shape[0], -1))
    eval_regression_model(y_test, y_preds)


