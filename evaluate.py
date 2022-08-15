
import os
import numpy as np
import pandas as pd
from utils.tf_model import create_model


def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def read_dataset(input_path):
    df_list = []
    for csv_file in sorted(os.listdir(input_path)):
        df = pd.read_csv(os.path.join(input_path, csv_file)).to_numpy()[-24*2:, -3:]
        df_list.append(df)
        # df_norm = []
        # for i in range(df.shape[1]):
        #     if i == 0:
        #         norm_column = df[:, i]
        #     else:
        #         norm_column = df[:, i] / 2
        #     df_norm.append(norm_column)
        # df_list.append(np.array(df_norm).T)
    merged_input = np.transpose(np.array(df_list).astype(np.float32), (1, 0, 2))  # transpose from 11x168x3 to 168x11x3
    merged_input = np.reshape(merged_input, (merged_input.shape[0], -1))  # reshape from 168x11x3 to 168x33

    return np.expand_dims(merged_input, axis=0)

def write2csv(predicts,
              path_write):
    location_train = pd.read_csv('dataset/data-train/location_output.csv')["station"].values
    location_test = pd.read_csv('dataset/public-test/location.csv')["station"].values

    for i in range(predicts.shape[0]):
        for j in range(location_test.shape[0]):
            if location_train[i] == location_test[j]:
                data = {'PM2.5': predicts[i].T}
                df_new = pd.DataFrame(data=data)
                df_new.to_csv(path_write + '/' + 'res_' + path_write.split('/')[-1] + '_' + str(j+1) + '.csv', index=False)


def main(path_input: str = "dataset/exp_test/input/",
              path_output: str = "dataset/exp_test/output/",
              path_model: str = 'trained_models/model'):

    WINDOW_SIZE = 2*24
    HORIZON = 24
    model = create_model(WINDOW_SIZE=WINDOW_SIZE, HORIZON=HORIZON, name=path_model.split('/')[-1])
    model.load_weights(path_model)

    folders = sorted(os.listdir(path_input))
    for folder in folders:
        raw_files_path = os.path.join(path_input, folder)
        new_file_path = os.path.join(path_output, folder)
        if not os.path.exists(new_file_path):
            os.mkdir(new_file_path)
        data = read_dataset(raw_files_path)
        predicts = np.squeeze(model.predict(data), axis=0).T
        write2csv(predicts, new_file_path)
        print(raw_files_path)
        print('\n', predicts)


if __name__ == "__main__":
    main()
