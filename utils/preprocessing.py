import pandas as pd
import os


def add_location_info(df: pd.DataFrame = None,
                      station_name: str = None,
                      location_df: pd.DataFrame = None,
                      save_to_csv: bool = False):
    dataframe_length = len(df)
    new_df = df
    row = location_df.loc[location_df["station"] == station_name]
    longitude = [row["longitude"]] * dataframe_length
    latitude = [row["latitude"]] * dataframe_length
    new_df["longitude"] = longitude
    new_df["latitude"] = latitude

    if save_to_csv:
        if not os.path.exists("./exp"):
            os.mkdir("./exp")
        new_df.to_csv("./exp/" + station_name + ".csv")

    return new_df


def padding(df: pd.DataFrame = None,
            station_name: str = None,
            location_df: pd.DataFrame = None,
            save_to_csv: bool = False):

    dataframe_length = len(df)
    new_df = df
    row = location_df.loc[location_df["station"] == station_name]
    longitude = [row["longitude"]] * dataframe_length
    latitude = [row["latitude"]] * dataframe_length
    new_df["longitude"] = longitude
    new_df["latitude"] = latitude


