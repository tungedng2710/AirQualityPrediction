import numpy as np
import pandas as pd


def use_location(root_dir: str = "./data-train/",
                 save_to_csv: bool = True,
                 mode: str = "training_data"):
    global df_new, df_distances
    if mode == "training_data":
        location_df = pd.read_csv(root_dir + "location_input.csv")

        station = location_df["station"].values
        longitude = location_df["longitude"].values
        latitude = location_df["latitude"].values

        for i in range(len(station)):
            distances = []
            for j in range(len(station)):
                distance = np.linalg.norm(np.array([longitude[i] - longitude[j], latitude[i] - latitude[j]]))
                distances.append(distance)
            order = np.argsort(distances)
            data = {'1st': [station[order[1]]], '2nd': [station[order[2]]], '3rd': [station[order[3]]],
                    '4th': [station[order[4]]], '5th': [station[order[5]]], '6th': [station[order[6]]],
                    '7th': [station[order[7]]], '8th': [station[order[8]]], '9th': [station[order[9]]],
                    '10th': [station[order[10]]]
                    }
            if i == 0:
                df_new = pd.DataFrame(data=data)
            else:
                df = pd.DataFrame(data=data)
                df_new = pd.concat([df_new, df], ignore_index=True)

        df_new.insert(0, "station", station, True)

        if save_to_csv:
            df_new.to_csv(root_dir + 'nearest_location' + ".csv", index=False)

    elif mode == "label_data":
        location_input = pd.read_csv(root_dir + "location_input.csv")
        location_output = pd.read_csv(root_dir + "location_output.csv")

        input_station = location_input["station"].values
        input_longitude = location_input["longitude"].values
        input_latitude = location_input["latitude"].values

        output_station = location_output["station"].values
        output_longitude = location_output["longitude"].values
        output_latitude = location_output["latitude"].values

        for i in range(len(output_station)):
            dict_distance = {}
            for j in range(len(input_station)):
                distance = np.linalg.norm(np.array([input_longitude[j] - output_longitude[i], input_latitude[j] - output_latitude[i]]))
                if j == 0:
                    dict_distance = {input_station[j]: [distance]}
                else:
                    dict_distance.update({input_station[j]: [distance]})

            if i == 0:
                df_distances = pd.DataFrame(data=dict_distance)
            else:
                df = pd.DataFrame(data=dict_distance)
                df_distances = pd.concat([df_distances, df], ignore_index=True)

        df_distances.insert(0, "station", output_station, True)
        if save_to_csv:
            df_distances.to_csv(root_dir + 'in_out_location' + ".csv", index=False)


use_location(mode='label_data')
