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

        # create the data_dump directory
        self.save_dir = os.path.join(self.temp_dir, "grid", f"data_dump_{self.temp_res}_{self.m}_{self.n}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def create_grid(self):
        crime_df = super().create()
        crime_types = self.data_columns

        for i in range(len(crime_types)):
            in_df = crime_df[crime_df[crime_types[i]] == 1]
            print(crime_types[i])
            grid = self._convert_grid(in_df=in_df)
            if self.plot:
                self._plot_surf(grid, title=crime_types[i])

        grid = self._convert_grid(crime_df, mode="all")
        if self.plot:
            self._plot_surf(grid, title="All")
            flatten_grid = np.sum(grid, axis=0).flatten()
            zero_ratio = sum(flatten_grid == 0) / len(flatten_grid) * 100
            save_path = os.path.join(self.figures_dir, "all_hist.png")
            plot_hist_dist(flatten_grid,
                           x_label="Total Event per Cell",
                           title=f"Zero Ratio {zero_ratio:.2f}",
                           save_path=save_path)

        print(f"Data Creation finished, data saved under {self.save_dir}")
        return grid, crime_df

    def check_is_created(self):
        if not os.path.exists(self.save_dir):
            return False

        grid = self.load_grid(mode="seperated")
        if grid.shape != (len(self.date_r), self.m, self.n, self.top_k):
            return False

        grid = self.load_grid(mode="all")
        if grid.shape != (len(self.date_r), self.m, self.n, 1):
            return False
        return True

    def load_grid(self, mode="seperated"):
        npy_paths = self.get_npy_paths(self.save_dir, mode)
        grid = []
        for path in npy_paths:
            with open(path, "rb") as f:
                grid.append(np.load(f))
        grid = np.stack(grid)
        return grid

    def _convert_grid(self, in_df, mode="seperated"):
        x_ticks = np.linspace(self.coord_range[1][0], self.coord_range[1][1], self.n + 1)
        y_ticks = np.linspace(self.coord_range[0][0], self.coord_range[0][1], self.m + 1)

        time_len = len(self.date_r)
        grid = np.zeros((time_len, self.m, self.n))
        for j in range(self.m):
            for i in range(self.n):
                lat_idx = (y_ticks[j] < in_df["Latitude"]) & (in_df["Latitude"] <= y_ticks[j + 1])
                lon_idx = (x_ticks[i] < in_df["Longitude"]) & (in_df["Longitude"] <= x_ticks[i + 1])
                cell_arr = in_df[lat_idx & lon_idx].resample(f'{self.temp_res}H')\
                    .size().reindex(self.date_r, fill_value=0).values
                grid[:, self.m - j - 1, i] = cell_arr  # start from left-bot
        grid = np.expand_dims(grid, -1)

        # save each time frame in temp directory
        for t in range(time_len):
            grid_t = grid[t]
            save_path = os.path.join(self.save_dir, mode, f"{t}.npy")
            if os.path.exists(save_path):
                with open(save_path, "rb") as f:
                    saved_arr = np.load(f)
                grid_t = np.concatenate([saved_arr, grid_t], axis=-1)
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
    def get_npy_paths(save_dir, mode):
        npy_path = os.path.join(save_dir, mode, "*.npy")
        grid_files = [file for file in glob.glob(npy_path)]
        file_arr = np.array(grid_files)
        sorted_idx = np.argsort(np.array([int(os.path.basename(path).split(".")[0]) for path in grid_files]))
        sorted_file_arr = file_arr[sorted_idx]
        return sorted_file_arr
