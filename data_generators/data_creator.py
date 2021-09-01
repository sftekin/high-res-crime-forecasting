import os

import pandas as pd
import numpy as np


class DataCreator:

    dataset_dir = "dataset"

    def __init__(self, data_raw_path, coord_range, spatial_res, temporal_res, time_range):
        self.__data_path = os.path.join(self.dataset_dir, data_raw_path)

        # spatial settings
        self.coord_range = coord_range  # [[41.60, 42.05], [-87.9, -87.5]]  # Lat-Lon
        self.m, self.n = spatial_res

        # temporal settings
        self.temp_res = temporal_res
        self.start_date, self.end_date = time_range
        self.date_r = pd.date_range(start=self.start_date, end=self.end_date, freq=f'{self.temp_res}H')

        self.data_columns = None

    def create(self):
        crime_df = pd.read_csv(self.__data_path)
        crime_df = self._preprocess(crime_df)
        crime_df = self._crop_spatial(crime_df)

        crime_types = list(crime_df["Primary Type"].unique())
        self.data_columns = crime_types

        grid_arr = []
        for i in range(len(crime_types)):
            in_df = crime_df[crime_df["Primary Type"] == crime_types[i]]
            print(crime_types[i])
            grid = self._convert_grid(in_df=in_df)
            grid_arr.append(grid)
        grid_arr = np.stack(grid_arr, axis=-1)

        print()

        # todo: think about side information, block, description, location desc., arrest, domestic, district
        # todo: think about feature extraction, spatial and temporal distance to previous crime
        # todo: think about time and spatial days
        # todo: after forming one frame save it under temp directory
        # todo: dont forget area merging

    @staticmethod
    def _preprocess(crime_df):
        crime_df.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)

        crime_df.drop(['Case Number', 'IUCR', 'Updated On',
                       'Year', 'FBI Code', 'Beat', 'Ward', 'Community Area', 'Location'], inplace=True, axis=1)

        crime_df.Date = pd.to_datetime(crime_df.Date, format='%m/%d/%Y %I:%M:%S %p')
        crime_df.index = pd.DatetimeIndex(crime_df.Date)

        # take top 20 categories and put the left out categories to OTHER
        loc_to_change = list(crime_df["Location Description"].value_counts()[20:].index)
        desc_to_change = list(crime_df["Description"].value_counts()[20:].index)

        crime_df.loc[crime_df["Location Description"].isin(loc_to_change), "Location Description"] = "OTHER"
        crime_df.loc[crime_df["Description"].isin(desc_to_change), "Description"] = "OTHER"

        crime_df['Primary Type'] = pd.Categorical(crime_df['Primary Type'])
        crime_df['Location Description'] = pd.Categorical(crime_df['Location Description'])
        crime_df['Description'] = pd.Categorical(crime_df['Description'])

        # Take top 10 crimes
        top10crime_types = list(crime_df["Primary Type"].value_counts()[:10].index)
        crime_df = crime_df[crime_df["Primary Type"].isin(top10crime_types)]

        return crime_df

    def _crop_spatial(self, crime_df):
        lat_idx = (self.coord_range[0][0] <= crime_df["Latitude"]) &\
                  (crime_df["Latitude"] <= self.coord_range[0][1])
        lon_idx = (self.coord_range[1][0] <= crime_df["Longitude"]) & \
                  (crime_df["Longitude"] <= self.coord_range[1][1])
        crime_df = crime_df[lat_idx & lon_idx]

        return crime_df

    def _convert_grid(self, in_df):
        x_ticks = np.linspace(self.coord_range[1][0], self.coord_range[1][1], self.n + 1)
        y_ticks = np.linspace(self.coord_range[0][0], self.coord_range[0][1], self.m + 1)

        time_len = len(self.date_r)
        grid = np.zeros((time_len, self.m, self.n))
        for j in range(self.m):
            for i in range(self.n):
                lat_idx = (y_ticks[j] < in_df["Latitude"]) & (in_df["Latitude"] <= y_ticks[j + 1])
                lon_idx = (x_ticks[i] < in_df["Longitude"]) & (in_df["Longitude"] <= x_ticks[i + 1])
                cell_arr = in_df[lat_idx & lon_idx].resample('H').size().reindex(self.date_r, fill_value=0).values
                grid[:, self.n - j - 1, i] = cell_arr

        return grid


