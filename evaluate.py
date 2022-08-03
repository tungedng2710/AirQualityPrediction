
import os
import numpy as np
import pandas as pd
from utils.tf_model import create_model


def read_dataset(input_path):
    df_list = []
    for csv_file in sorted(os.listdir(input_path)):
        df = pd.read_csv(os.path.join(input_path, csv_file)).to_numpy()[:, -3:]
        df_list.append(df)
    merged_input = np.transpose(np.array(df_list).astype(np.float32), (1, 0, 2))  # transpose from 11x168x3 to 168x11x3
    merged_input = np.reshape(merged_input, (merged_input.shape[0], -1))  # reshape from 168x11x3 to 168x33

    return np.expand_dims(merged_input, axis=0)

def write2csv(predicts):
    location_train =


def main(path_input: str = "dataset/exp_test/input/",
              path_output: str = "dataset/exp_test/output/",
              path_model: str = 'trained_models/model_lstm_2'):

    model = create_model(name=path_model.split('/')[-1])
    model.load_weights(path_model)

    folders = sorted(os.listdir(path_input))
    for folder in folders:
        raw_files_path = os.path.join(path_input, folder)
        new_file_path = os.path.join(path_output, folder)
        if not os.path.exists(new_file_path):
            os.mkdir(new_file_path)
        data = read_dataset(raw_files_path)
        predicts = np.squeeze(model.predict(data), axis=0)
        print(raw_files_path)
        print('\n', predicts)



write2csv()