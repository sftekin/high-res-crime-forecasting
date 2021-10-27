class Config:
    def __init__(self):
        self.data_params = {
            "data_raw_path": "chicago_raw.csv",
            "coord_range": [[41.60, 42.05], [-87.9, -87.5]],
            "temporal_res": 24,
            "time_range": ("2015-01-01", "2019-11-01"),
            "top_k": 10,
            "plot": False
        }
        self.grid_params = {
            "spatial_res": (50, 33),  # 500mx500m
            "include_side_info": False
        }

        self.graph_params = {
            "event_threshold": 5000,
            "include_side_info": False,
            "grid_name": "all",
            "min_cell_size": (5, 5),
            "normalize_coords": True
        }

        self.batch_gen_params = {}
        self.trainer_params = {}
        self.model_params = {}
