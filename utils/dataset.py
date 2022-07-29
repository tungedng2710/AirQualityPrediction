import os
from utils.preprocessing import add_location_info
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class AI4VN_AirDataset(Dataset):
    def __init__(self,
                 root_dir: str = "data-train/",
                 mode: str = "train",
                 drop_null: bool = True,
                 use_location_info: bool = True):
        """
        root_dir: path to data-train directory
        drop_null: drop row missing data
        mode: "train" or "test"
        """
        assert root_dir is not None
        self.root_dir = root_dir
        if self.root_dir[-1] != '/':
            self.root_dir = self.root_dir + '/'
        self.mode = mode

        self.df_list = []
        if mode == "train":
            raw_files_path = self.root_dir + "input/"
            location_df = pd.read_csv("./data-train/location_input.csv")
        elif mode == "test":
            raw_files_path = self.root_dir + "output/"
            location_df = pd.read_csv("./data-train/location_output.csv")

        for csv_file in os.listdir(raw_files_path):
            df = pd.read_csv(raw_files_path + csv_file)
            if use_location_info:
                station_name = csv_file.split(".csv")[0]
                df = add_location_info(df, station_name, location_df)
            self.df_list.append(df)

        self.merged_df = pd.concat(self.df_list, ignore_index=True, sort=False).iloc[:, 1:]
        if drop_null:
            self.merged_df = self.merged_df.dropna()
        self.columns = list(self.merged_df.columns)

        self.X, self.y = self.preload()

    def preload(self):
        feat_cols = self.columns
        feat_cols.remove("PM2.5")
        feat_cols.remove("timestamp")
        X = self.merged_df[feat_cols].values
        y = self.merged_df["PM2.5"].values
        return X, y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.merged_df)


class AI4VN_AirDataLoader:
    def __init__(self):
        self.train_set = AI4VN_AirDataset(mode="train")
        self.test_set = AI4VN_AirDataset(mode="test")

    def get_data_loader_sklearn(self):
        X_train = self.train_set.X
        X_test = self.test_set.X
        y_train = self.train_set.y
        y_test = self.test_set.y

        return X_train, X_test, y_train, y_test

    def get_data_loader_pytorch(self,
                                batch_size_train: int = 128,
                                batch_size_test: int = 128,
                                num_workers: int = 8):
        train_loader = DataLoader(dataset=self.train_set,
                                  batch_size=batch_size_train,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set,
                                 batch_size=batch_size_test,
                                 shuffle=False,
                                 num_workers=num_workers)
        return train_loader, test_loader