import os
import pandas as pd
import numpy as np




def fill(root_dir: str = "data-train/",
         mode: str = "train",
         drop_null: bool = False,
         use_location_info: bool = True):
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

    location = pd.read_csv(root_dir + "nearest_location.csv").values
    for csv_file in os.listdir(raw_files_path):
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

        if not os.path.exists("./exp"):
            os.mkdir("./exp")
        df.to_csv("./exp/" + station_name + ".csv")


