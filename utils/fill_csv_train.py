import os
import pandas as pd
import numpy as np
import shutil


def fill(raw_files_path: str,
         location: np.array,
         mode: str):
    """
        raw_files_path: path to data-train csv file directory
        location: array contains location
        mode: "train" or "test"
    """
    list_csv = sorted(os.listdir(raw_files_path))
    for csv_file in list_csv:
        df = pd.read_csv(raw_files_path + csv_file)
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
                        df_fill = pd.read_csv(raw_files_path + csv_file_fill)
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

        if mode == "train":
            if not os.path.exists("../dataset/exp"):
                os.mkdir("../dataset/exp/")
            if not os.path.exists("../dataset/exp/input"):
                os.mkdir("../dataset/exp/input/")
            df.to_csv("../exp/input/" + csv_file)
        elif mode == "test":
            if not os.path.exists("../dataset/exp"):
                os.mkdir("../dataset/exp/")
            if not os.path.exists("../dataset/exp/output"):
                os.mkdir("../dataset/exp/output/")
            df.to_csv("../exp/output/" + csv_file)



def full_fill(root_dir: str = "../data-train/",
              mode: str = "test"):
    """
    root_dir: path to data-train directory
    drop_null: drop row missing data
    mode: "train" or "test"
    """
    global raw_files_path
    assert root_dir is not None
    if root_dir[-1] != '/':
        root_dir = root_dir + '/'

    if mode == "train":
        raw_files_path = root_dir + "input/"
    elif mode == "test":
        raw_files_path = root_dir + "output/"

    location = pd.read_csv(root_dir + "nearest_location_all.csv").values
    fill(raw_files_path, location, mode)
    # shutil.copyfile(root_dir + "nearest_location.csv", "../exp1/nearest_location.csv")


full_fill()
path = "../dataset/exp/output/"
list_csv = sorted(os.listdir(path))
for csv_file in list_csv:
    df = pd.read_csv(path + csv_file)
    df = df.interpolate(method='linear', axis=0)
    a = df.isnull().any().any()
    print("{name} is ----------- {state}".format(name=csv_file, state=a))

    df.to_csv("../exp/output/" + csv_file)

