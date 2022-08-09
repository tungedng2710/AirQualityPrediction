import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

class AI4VN_AirDataset():
    def __init__(self,
                 root_dir: str = "exp/"):
        """
        root_dir: path to data-train directory
        drop_null: drop row missing data
        mode: "train" or "test"
        """
        assert root_dir is not None
        self.root_dir = root_dir
        if self.root_dir[-1] != '/':
            self.root_dir = self.root_dir + '/'

        input_path = self.root_dir + "input/"
        output_path = self.root_dir + "output/"

        df_list_input = []
        df_list_output = []
        for csv_file in sorted(os.listdir(input_path)):
            df = pd.read_csv(input_path + csv_file).to_numpy()[:, -3:]
            df_list_input.append(df)
            # df_norm = []
            # for i in range(df.shape[1]):
            #     if i == 0:
            #         norm_column = df[:, i]
            #     else:
            #         norm_column = df[:, i]/2
            #     df_norm.append(norm_column)
            # df_list_input.append(np.array(df_norm).T)

        self.merged_input = np.transpose(np.array(df_list_input), (1, 0, 2))   # transpose from 11x9000x3 to 9000x11x3
        self.merged_input = np.reshape(self.merged_input,
                                       (self.merged_input.shape[0], -1))  # reshape from 9000x11x3 to 9000x33

        for csv_file in sorted(os.listdir(output_path)):
            df = pd.read_csv(output_path + csv_file).to_numpy()[:, -3]
            df_list_output.append(df)
        self.merged_output = np.transpose(np.array(df_list_output), (1, 0))  # reshape from 4x9000 to 9000x4

    # Create function to label windowed data
    def get_labelled_windows(self, x, horizon):
        """
        Creates labels for windowed dataset.

        E.g. if horizon=1 (default)
        Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
        """
        return x[:, :-horizon], x[:, -horizon:]

    # Create function to view NumPy arrays as windows
    def make_windows(self, window_size=7*24, horizon=24):
        """
        Turns a 1D array into a 2D array of sequential windows of window_size.
        """
        # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
        window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)

        # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
        window_indexes = window_step + np.expand_dims(np.arange(self.merged_input.shape[0] - (window_size+horizon - 1)),
                                                      axis=0).T  # create 2D array of windows of size window_size

        # 3. Index on the target array (time series) with 2D array of multiple window steps
        windowed_array = self.merged_input[window_indexes]
        horizon_array = self.merged_output[window_indexes]

        # 4. Get the labelled windows
        windows, _ = self.get_labelled_windows(windowed_array, horizon=horizon)
        _, labels = self.get_labelled_windows(horizon_array, horizon=horizon)

        return windows, labels


def AI4VN_dataloader(root_dir, test_split, window_size, horizon):
    dataset = AI4VN_AirDataset(root_dir=root_dir)
    X, y = dataset.make_windows(window_size=window_size, horizon=horizon)
    split_size = int(X.shape[0] * (1-test_split))
    X_train = X[:split_size].astype(np.float32)
    X_test = X[split_size:].astype(np.float32)
    y_train = y[:split_size].astype(np.float32)
    y_test = y[split_size:].astype(np.float32)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test
