import os

import pandas as pd


class DataCreator:
    dataset_dir = "dataset"
    figures_dir = "figures"
    temp_dir = "temp"

    def __init__(self, data_params):
        self.__data_path = os.path.join(self.dataset_dir, data_params["data_raw_path"])
        self.plot = data_params["plot"]

        # spatial settings
        self.coord_range = data_params["coord_range"]  # [[41.60, 42.05], [-87.9, -87.5]]  # Lat-Lon
        self.m, self.n = data_params["spatial_res"]

        # temporal settings
        self.temp_res = data_params["temporal_res"]
        self.start_date, self.end_date = data_params["time_range"]
        self.date_r = pd.date_range(start=self.start_date, end=self.end_date, freq=f'{self.temp_res}H')

        self.save_dir = os.path.join(self.temp_dir, f"data_dump_{self.temp_res}_{self.m}_{self.n}")

        # create the data_dump directory
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def create(self):
        crime_df = pd.read_csv(self.__data_path)
        crime_df = self._preprocess(crime_df)
        crime_df = self._crop_spatial(crime_df)
        crime_df = self._perform_dummies(crime_df)

        return crime_df

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
        lat_idx = (self.coord_range[0][0] <= crime_df["Latitude"]) & \
                  (crime_df["Latitude"] <= self.coord_range[0][1])
        lon_idx = (self.coord_range[1][0] <= crime_df["Longitude"]) & \
                  (crime_df["Longitude"] <= self.coord_range[1][1])
        crime_df = crime_df[lat_idx & lon_idx]

        return crime_df

    @staticmethod
    def _perform_dummies(crime_df):
        crime_df = crime_df[["Primary Type", "Description", "Location Description",
                             "Arrest", "Domestic", "District", "Latitude", "Longitude"]]
        crime_df = crime_df.dropna()

        df_list = []
        categorical = ["Primary Type", "Description", "Location Description"]
        for category in categorical:
            c_df = pd.get_dummies(crime_df[category], dtype=float)
            c_df.columns = [f"{category}_{i}" for i in range(len(c_df.columns))]
            df_list.append(c_df)
        cat_df = pd.concat(df_list, axis=1)
        crime_df = crime_df.drop(columns=categorical)
        crime_df = pd.concat([crime_df, cat_df], axis=1)

        crime_df["Arrest"] = crime_df["Arrest"].astype(float)
        crime_df["Domestic"] = crime_df["Domestic"].astype(float)

        return crime_df
