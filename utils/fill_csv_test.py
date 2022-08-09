import os
import pandas as pd
import numpy as np


def fill(raw_files_path: str,
         new_file_path: str,
         location: np.array):
    """
        raw_files_path: path to data-train csv file directory
        location: array contains location
        new_file_path: path to save new csv file
    """
    list_csv = sorted(os.listdir(raw_files_path))
    print("-------------------------------")
    print("\n", raw_files_path)
    for csv_file in list_csv:
        df = pd.read_csv(os.path.join(raw_files_path, csv_file))
        temperature = df['temperature'].values
        humidity = df['humidity'].values
        PM2_5 = df['PM2.5'].values

        temperature_new = temperature
        humidity_new = humidity
        PM2_5_new = PM2_5

        station_name = csv_file.split(".csv")[0]
        for j in range(location.shape[0]):
            if str(location[j][0]) == station_name:
                for k in range(location.shape[1]):
                    if np.isnan(temperature_new).any() or np.isnan(humidity_new).any() or np.isnan(PM2_5_new).any():
                        csv_file_fill = str(location[j][k]) + ".csv"
                        df_fill = pd.read_csv(os.path.join(raw_files_path, csv_file_fill))
                        temperature_fill = df_fill['temperature'].values
                        humidity_fill = df_fill['humidity'].values
                        PM2_5_fill = df_fill['PM2.5'].values

                        for m in range(len(temperature_new)):
                            if np.isnan(temperature_new[m]):
                                temperature_new[m] = temperature_fill[m]

                            if np.isnan(humidity_new[m]):
                                humidity_new[m] = humidity_fill[m]

                            if np.isnan(PM2_5_new[m]):
                                PM2_5_new[m] = PM2_5_fill[m]

                    else:
                        break

        df.replace(temperature, temperature_new)
        df.replace(humidity, humidity_new)
        df.replace(PM2_5, PM2_5_new)

        if df.isnull().any().any():
            df = df.interpolate(method='linear', axis=0)

        print("{name} is ----------- {state}".format(name=csv_file, state=df.isnull().any().any()))

        df.to_csv(os.path.join(new_file_path, csv_file))


def full_fill(root_dir: str = "../dataset/public-test/input/",
              new_dir: str = "../dataset/exp_test/input/"):
    """
    root_dir: path to data directory
    """
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    assert root_dir, new_dir is not None

    location = pd.read_csv("../dataset/public-test/nearest_location.csv").values
    folders = sorted(os.listdir(root_dir))
    for folder in folders:
        raw_files_path = os.path.join(root_dir, folder)
        new_file_path = os.path.join(new_dir, folder)
        if not os.path.exists(new_file_path):
            os.mkdir(new_file_path)
        fill(raw_files_path, new_file_path, location)


full_fill()