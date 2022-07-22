import pandas as pd
import numpy as np
import os

from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split


class AI4VN_AirDataset(Dataset):
    def __init__(self, 
                 root_dir: str = "data-train/",
                 mode: str = "input",
                 drop_null: bool = False):
        """
        root_dir: path to data-train directory
        drop_null: drop row missing data
        mode: "input" or "output"
        """
        assert root_dir is not None
        self.root_dir = root_dir
        if self.root_dir[-1] != '/':
            self.root_dir = self.root_dir+'/'
        self.mode = mode

        input_df = []
        output_df = []
        print("Loading raw csv files...")
        for csv_file in os.listdir(self.root_dir+"input/"):
            input_df.append(pd.read_csv(self.root_dir+"input/"+csv_file))
        self.merged_input_df = pd.concat(input_df, ignore_index=True, sort=False).iloc[:, 1:]
        if drop_null:
            self.merged_input_df = self.merged_input_df.dropna()
        self.columns = list(self.merged_input_df.columns)

        for csv_file in os.listdir(self.root_dir+"output/"):
            output_df.append(pd.read_csv(self.root_dir+"output/"+csv_file))
        self.merged_output_df = pd.concat(input_df, ignore_index=True, sort=False).iloc[:, 1:]
        if drop_null:
            self.merged_output_df = self.merged_output_df.dropna()
        self.columns = list(self.merged_output_df.columns)

        self.X_in, self.y_in, self.X_out, self.y_out = self.preload()
        
    def preload(self):
        feat_cols = self.columns
        feat_cols.remove("PM2.5")
        feat_cols.remove("timestamp")
        y_out = self.merged_output_df["PM2.5"].values
        X_out = self.merged_output_df[feat_cols].values
        y_in = self.merged_input_df["PM2.5"].values
        X_in = self.merged_input_df[feat_cols].values
        return X_in, y_in, X_out, y_out

    def __getitem__(self, index):
        if self.mode == "input":
            return self.X_in[index], self.y_in[index]
        else:
            return self.X_out[index], self.y_out[index]


    def __len__(self):
        if self.mode == "input":
            return len(self.merged_input_df)
        else: 
            return len(self.merged_output_df)

class AI4VN_AirDataLoader:
    def __init__(self,
                 dataset: AI4VN_AirDataset = None,
                 random_seed: int = 42,
                 dataset_type: str = "input",
                 test_size: float = 0.2):
        """
        dataset: AI4VN_AirDataset \
        random_seed (int, default: 64): Controls the shuffling applied to the data before applying the split \
        dataset_type (str, default: input): use data from "input" folder under the data-train \
        test_size (float, default: 0.2): Should be between 0.0 and 1.0 and represent the proportion \
                                         of the dataset to include in the test split.
        """
        if dataset is None:
            self.dataset = AI4VN_AirDataset(drop_null=True, mode=dataset_type)
        else:
            self.dataset = dataset
        self.random_seed = random_seed
        self.dataset_type = dataset_type
        self.test_size = test_size
    
    def get_data_loader_sklearn(self, 
                                test_size: float = 0.2):
        if self.dataset_type == "output":
            X = self.dataset.X_out
            y = self.dataset.y_out
            X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, test_size=test_size, random_state=self.random_seed
                                                )
            return X_train, X_test, y_train, y_test
        else:
            X = self.dataset.X_in
            y = self.dataset.y_in
            X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, test_size=test_size, random_state=self.random_seed
                                                )
            return X_train, X_test, y_train, y_test

    def get_data_loader_pytorch(self, 
                                batch_size: int = 128,
                                num_workers: int = 8):
        dataset_size = len(self.dataset)
        test_size = int(dataset_size*self.test_size)
        train_size = dataset_size - test_size
        train_set, val_set = random_split(self.dataset, [train_size, test_size])
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers)
        val_loader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
        return train_loader, val_loader
        