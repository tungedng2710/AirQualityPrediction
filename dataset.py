import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import sklearn
from sklearn.model_selection import train_test_split


class AI4VN_AirDataset:
    def __init__(self, 
                 root_dir: str = "data-train/",
                 drop_null: bool = False,
                 get_input_data: bool = True,
                 get_output_data: bool = False):
        """
        root_dir: path to data-train directory
        drop_null: drop row missing data
        get_input_data: get csv files in the "input" folder
        get_output_data: get csv files in the "output" folder
        """
        assert root_dir is not None
        self.root_dir = root_dir
        self.get_input_data = get_input_data
        self.get_output_data = get_output_data
        if self.root_dir[-1] != '/':
            self.root_dir = self.root_dir+'/'

        input_df = []
        output_df = []
        print("Loading raw csv files...")
        if self.get_input_data:
            for csv_file in tqdm(os.listdir(self.root_dir+"input/")):
                input_df.append(pd.read_csv(self.root_dir+"input/"+csv_file))
            self.merged_input_df = pd.concat(input_df, ignore_index=True, sort=False).iloc[:, 1:]
            if drop_null:
                self.merged_input_df = self.merged_input_df.dropna()
            self.columns = list(self.merged_input_df.columns)

        if self.get_output_data:
            for csv_file in tqdm(os.listdir(self.root_dir+"output/")):
                output_df.append(pd.read_csv(self.root_dir+"output/"+csv_file))
            self.merged_output_df = pd.concat(input_df, ignore_index=True, sort=False).iloc[:, 1:]
            if drop_null:
                self.merged_output_df = self.merged_output_df.dropna()
            self.columns = list(self.merged_output_df.columns)
    
    def add_location_info(self, 
                          dataframe: pd.DataFrame,
                          location_info_df: pd.DataFrame):
        if dataframe is None:
            dataframe = self.merged_input_df
            location_info_df = pd.read_csv(self.root_dir+"location_input.csv")
        else:
            assert len(dataframe) != len(location_info_df)

    def __getitem__(self, index):
        return self.merged_input_df.iloc[index, :]

    def __len__(self):
        return len(self.merged_input_df)

    def get_data_loader(self, 
                        test_size: float = 0.2, 
                        random_state: int = 42):
        feat_cols = self.columns
        feat_cols.remove("PM2.5")
        feat_cols.remove("timestamp")
        if self.get_input_data:
            y = self.merged_input_df["PM2.5"].values
            X = self.merged_input_df[feat_cols].values
            X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, test_size=test_size, random_state=random_state
                                                )
            return X_train, X_test, y_train, y_test
        if self.get_output_data:
            y = self.merged_output_df["PM2.5"].values
            X = self.merged_output_df[feat_cols].values
            X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, test_size=test_size, random_state=random_state
                                                )
            return X_train, X_test, y_train, y_test
    