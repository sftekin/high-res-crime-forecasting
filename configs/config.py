class Config:
    def __init__(self):
        self.data_params = {
            "data_raw_path": "chicago_raw.csv",
            "coord_range": [[41.60, 42.05], [-87.9, -87.5]],
            "temporal_res": 24,
            "time_range": ("2015-01-01", "2017-01-01"),
            "crime_categories": ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE',
                                 'NARCOTICS', 'ROBBERY', 'ASSAULT',
                                 'DECEPTIVE PRACTICE', 'BURGLARY'],
            "plot": False
        }
        self.grid_params = {
            "spatial_res": (50, 33),  # 500mx500m
            "include_side_info": False
        }

        self.graph_params = {
            "event_threshold": 1000,
            "include_side_info": False,
            "grid_name": "all",
            "min_cell_size": (2, 2),
            "normalize_coords": True,
            "k_nearest": 10
        }

        self.batch_gen_params = {}
        self.trainer_params = {}
        self.model_params = {}
        self.experiment_params = {}
