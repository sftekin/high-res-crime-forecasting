class Config:
    def __init__(self):
        self.data_params = {
            "data_raw_path": "chicago_raw.csv",
            "coord_range": [[41.60, 42.05], [-87.9, -87.5]],
            "temporal_res": 3,
            "time_range": ("2015-01-01", "2016-01-01"),
            "crime_categories": ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE',
                                 'NARCOTICS', 'ROBBERY', 'ASSAULT',
                                 'DECEPTIVE PRACTICE', 'BURGLARY'],
            "plot": True
        }
        self.grid_params = {
            "spatial_res": (100, 66),  # 500mx500m
            "include_side_info": False
        }

        self.graph_params = {
            "event_threshold": 5000,
            "include_side_info": False,
            "grid_name": "all",
            "min_cell_size": (2, 2),
            "normalize_coords": True
        }

        self.batch_gen_params = {}
        self.trainer_params = {}
        self.model_params = {}
