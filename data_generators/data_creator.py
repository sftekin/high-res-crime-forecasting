import os

import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap


class DataCreator:
    dataset_dir = "dataset"
    figures_path = "figures"

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
            self.plot_2d(grid, crime_types[i])
            self.plot_3d(in_grid=grid, title=crime_types[i])
            self.plot_3d_bar(grid, title=crime_types[i])
            # grid_arr.append(grid)
        # grid_arr = np.stack(grid_arr, axis=-1)

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
        lat_idx = (self.coord_range[0][0] <= crime_df["Latitude"]) & \
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
                grid[:, self.m - j - 1, i] = cell_arr

        return grid

    def plot_2d(self, in_grid, title):
        sum_arr = np.sum(in_grid, axis=0)

        fig, ax = plt.subplots(figsize=(10, 15))
        self.plot_background(ax=ax)

        # create your own custom color
        color_array = plt.cm.get_cmap('Reds')(range(1000))
        color_array[:, -1] = np.linspace(0.6, 1, 1000)
        map_object = LinearSegmentedColormap.from_list(name='fading_red', colors=color_array)
        plt.register_cmap(cmap=map_object)

        ax.imshow(X=sum_arr,
                  cmap="fading_red",
                  interpolation='nearest',
                  extent=[*self.coord_range[1], *self.coord_range[0]])
        ax.set_title(title, fontsize=22)

        dir_path = os.path.join(self.figures_path, "2d")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_path = os.path.join(dir_path, f"{title}.png")
        plt.savefig(save_path, dpi=250, bbox_inches='tight')

    def plot_3d(self, in_grid, title):
        sum_arr = np.sum(in_grid, axis=0)
        x = np.linspace(self.coord_range[1][0], self.coord_range[1][1], self.n)
        y = np.linspace(self.coord_range[0][0], self.coord_range[0][1], self.m)
        xx, yy = np.meshgrid(x, y)

        ax = Axes3D(plt.figure())
        ax.plot_surface(xx, yy, np.flip(sum_arr, axis=1), cmap=plt.cm.viridis, cstride=1, rstride=1)
        ax.set_xticks(x[::10])
        ax.set_yticks(y[::10])
        ax.set_xticklabels(np.round(x[::10], decimals=2))
        ax.set_yticklabels(np.round(y[::10], decimals=2))
        ax.view_init(30, 90)

        dir_path = os.path.join(self.figures_path, "3d", "surface")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_path = os.path.join(dir_path, f"{title}.png")
        plt.savefig(save_path, dpi=250, bbox_inches='tight')

    def plot_3d_bar(self, in_grid, title):
        import matplotlib.colors as colors
        import matplotlib.cm as cm

        sum_arr = np.flip(np.sum(in_grid, axis=0), axis=1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_data, y_data = np.meshgrid(np.arange(sum_arr.shape[1]),
                                     np.arange(sum_arr.shape[0]))
        x_data = x_data.flatten()
        y_data = y_data.flatten()
        z_data = sum_arr.flatten()

        dz = z_data
        offset = dz + np.abs(dz.min())
        fracs = offset.astype(float) / offset.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        color_values = cm.viridis(norm(fracs.tolist()))

        ax.bar3d(x_data,
                 y_data,
                 np.zeros(len(z_data)),
                 1, 1, z_data, color=color_values)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.view_init(30, 90)
        dir_path = os.path.join(self.figures_path, "3d", "bar")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_path = os.path.join(dir_path, f"{title}.png")
        plt.savefig(save_path, dpi=250, bbox_inches='tight')

    def plot_background(self, ax):
        background_path = "eda/background/chicago.png"
        x_ticks = np.linspace(self.coord_range[1][0], self.coord_range[1][1], self.n + 1)
        y_ticks = np.linspace(self.coord_range[0][0], self.coord_range[0][1], self.m + 1)

        x_tick_labels = ["{:2.3f}".format(long) for long in x_ticks]
        y_tick_labels = ["{:2.3f}".format(lat) for lat in y_ticks]

        ax.set_xticks(ticks=x_ticks)
        # ax.set_xticklabels(labels=x_tick_labels,
        #                    rotation=30,
        #                    size=12)
        ax.set_yticks(ticks=y_ticks)
        # ax.set_yticklabels(labels=y_tick_labels,
        #                    size=12)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        img = mpimg.imread(background_path)
        ax.imshow(img,
                  interpolation='bilinear',
                  extent=[*self.coord_range[1], *self.coord_range[0]])
        ax.grid(True)
        return ax
