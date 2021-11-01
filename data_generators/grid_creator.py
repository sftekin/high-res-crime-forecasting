import os
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import matplotlib.cm as cm

from data_generators.data_creator import DataCreator
from helpers.plot_helper import plot_hist_dist


class GridCreator(DataCreator):
    def __init__(self, data_params, grid_params):
        super(GridCreator, self).__init__(data_params)
        self.m, self.n = grid_params["spatial_res"]
        self.include_side_info = grid_params["include_side_info"]

        # create the data_dump directory
        self.grid_save_dir = os.path.join(self.temp_dir, "grid", f"data_dump_{self.temp_res}_{self.m}_{self.n}")
        if not os.path.exists(self.grid_save_dir):
            os.makedirs(self.grid_save_dir)

    def create_grid(self):
        crime_df = super().create()
        crime_types = self.data_columns

        return_ds = None
        for crime_name in crime_types:
            print(crime_name)
            in_df = crime_df[crime_df[crime_name] == 1]
            grid = self._convert_grid(in_df=in_df, dataset_name=crime_name)
            event_counts = grid[..., 2]
            if self.plot:
                self._plot_surf(event_counts, title=crime_name)

        all_grid = self._convert_grid(crime_df, dataset_name="all")
        event_counts = all_grid[..., 2]
        if self.plot:
            self._plot_surf(event_counts, title="All")
            flatten_grid = np.sum(event_counts, axis=0).flatten()
            zero_ratio = sum(flatten_grid == 0) / len(flatten_grid) * 100
            save_path = os.path.join(self.figures_dir, "all_hist.png")
            plot_hist_dist(flatten_grid,
                           x_label="Total Event per Cell",
                           title=f"Zero Ratio {zero_ratio:.2f}",
                           save_path=save_path)

        print(f"Data Creation finished, data saved under {self.grid_save_dir}")

    def check_is_created(self):
        if not os.path.exists(self.grid_save_dir) or \
                len(os.listdir(self.grid_save_dir)) == 0:
            return False

        crime_names = os.listdir(self.grid_save_dir)
        if len(crime_names) != len(self.crime_types) + 1:
            return False

        for ds in crime_names:
            grid_paths = self.get_paths(dataset_name=ds)
            if len(grid_paths) != len(self.date_r):
                return False
            sample_path = grid_paths[-1]
            with open(sample_path, "rb") as f:
                grid = np.load(f)
            num_features = 3 if not self.include_side_info else 47
            if grid.shape != (self.m, self.n, num_features):
                return False

        return True

    def get_paths(self, dataset_name):
        npy_paths = self.get_npy_paths(self.grid_save_dir, dataset_name)
        return npy_paths

    def _convert_grid(self, in_df, dataset_name="all"):
        x_ticks = np.linspace(self.coord_range[1][0], self.coord_range[1][1], self.n + 1)
        y_ticks = np.linspace(self.coord_range[0][0], self.coord_range[0][1], self.m + 1)

        if self.include_side_info:
            num_feats = in_df.shape[1] + 1  # categorical features + event_count + node_location
        else:
            num_feats = 3  # event count + node_location

        time_len = len(self.date_r)
        grid = np.zeros((time_len, self.m, self.n, num_feats))
        for j in range(self.m):
            for i in range(self.n):
                lat_idx = (y_ticks[j] < in_df["Latitude"]) & (in_df["Latitude"] <= y_ticks[j + 1])
                lon_idx = (x_ticks[i] < in_df["Longitude"]) & (in_df["Longitude"] <= x_ticks[i + 1])
                centroid = np.array([(x_ticks[i + 1] + x_ticks[i]) / 2, (y_ticks[j] + y_ticks[j + 1]) / 2])
                cell_arr = in_df[lat_idx & lon_idx].resample(f'{self.temp_res}H')\
                    .size().reindex(self.date_r, fill_value=0).values
                grid[:, self.m - j - 1, i, :2] = centroid
                grid[:, self.m - j - 1, i, 2] = cell_arr  # start from left-bot
                if self.include_side_info:
                    cat_df = in_df[lat_idx & lon_idx].resample(f"{self.temp_res}H").mean().\
                        reindex(self.date_r, fill_value=0)
                    cat_df = cat_df.fillna(0)
                    cat_df = cat_df.drop(columns=["Latitude", "Longitude"])
                    grid[:, self.m - j - 1, i, 3:] = cat_df.values

        # save each time frame in temp directory
        save_dir = os.path.join(self.grid_save_dir, dataset_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for t in range(time_len):
            grid_t = grid[t]
            save_path = os.path.join(save_dir, f"{t}.npy")
            with open(save_path, "wb") as f:
                np.save(f, grid_t)
        return grid

    def _plot_2d(self, in_grid, title):
        sum_arr = np.sum(in_grid, axis=0)

        fig, ax = plt.subplots(figsize=(10, 15))
        self._plot_background(ax=ax)

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

        dir_path = os.path.join(self.figures_dir, "2d")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_path = os.path.join(dir_path, f"{title}.png")
        plt.savefig(save_path, dpi=250, bbox_inches='tight')

    def _plot_3d(self, in_grid, title):
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

        dir_path = os.path.join(self.figures_dir, "3d", "surface")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_path = os.path.join(dir_path, f"{title}.png")
        plt.savefig(save_path, dpi=250, bbox_inches='tight')

    def _plot_3d_bar(self, in_grid, title):
        sum_arr = np.sum(in_grid, axis=0)
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
        dir_path = os.path.join(self.figures_dir, "3d", "bar")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_path = os.path.join(dir_path, f"{title}.png")
        plt.savefig(save_path, dpi=250, bbox_inches='tight')

    def _plot_background(self, ax, tick_labels=False):
        background_path = "eda/background/chicago.png"
        x_ticks = np.linspace(self.coord_range[1][0], self.coord_range[1][1], self.n + 1)
        y_ticks = np.linspace(self.coord_range[0][0], self.coord_range[0][1], self.m + 1)
        ax.set_xticks(ticks=x_ticks)
        ax.set_yticks(ticks=y_ticks)

        if tick_labels:
            x_tick_labels = ["{:2.3f}".format(long) for long in x_ticks]
            y_tick_labels = ["{:2.3f}".format(lat) for lat in y_ticks]
            ax.set_xticklabels(labels=x_tick_labels, rotation=30, size=12)
            ax.set_yticklabels(labels=y_tick_labels, size=12)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        img = mpimg.imread(background_path)
        ax.imshow(img, interpolation='bilinear', extent=[*self.coord_range[1], *self.coord_range[0]])
        ax.grid(True)

        return ax

    def _plot_surf(self, grid, title):
        grid = np.squeeze(grid)
        self._plot_2d(grid, title=title)
        self._plot_3d(in_grid=grid, title=title)
        self._plot_3d_bar(grid, title=title)

    @staticmethod
    def get_npy_paths(grid_save_dir, dataset_name):
        npy_path = os.path.join(grid_save_dir, dataset_name, "*.npy")
        grid_files = [file for file in glob.glob(npy_path)]
        file_arr = np.array(grid_files)
        sorted_idx = np.argsort(np.array([int(os.path.basename(path).split(".")[0]) for path in grid_files]))
        sorted_file_arr = file_arr[sorted_idx]
        return sorted_file_arr
