

class Config:
    def __init__(self):
        self.data_params = {
            "data_raw_path": "chicago_raw.csv",
            "coord_range": [[41.60, 42.05], [-87.9, -87.5]],
            "temporal_res": 24,
            "time_range": ("2015-01-01", "2019-11-01"),
            "plot": False
        }

        self.graph_params = {
            "event_threshold": 5000,
            "include_side_info": False,
        }

        self.grid_params = {
            "spatial_res": (100, 66)
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
