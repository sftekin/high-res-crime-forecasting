

class Config:
    def __init__(self):
        self.data_params = {
            "data_raw_path": "chicago_raw.csv",
            "coord_range": [[41.60, 42.05], [-87.9, -87.5]],
            "temporal_res": 24,
            "time_range": ("2015-01-01", "2019-11-01"),
            "top_k": 10,
            "plot": True
        }

        self.graph_params = {
            "event_threshold": 5000,
            "include_side_info": False,
            "grid_name": "all",
            "min_cell_size": (4, 4)
        }

        self.grid_params = {
            "spatial_res": (100, 66)  # 500mx500m
        }

        self.batch_gen_params = {
            "graph": {
                "test_size": 266,
                "val_ratio": 0.20,
                "window_in_len": 10,
                "window_out_len": 1,
                "batch_size": 5,
                "shuffle": False,
            }
        }
