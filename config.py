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
            "min_cell_size": (2, 2)
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

        self.graph_trainer_prams = {
            "device": 'cuda',
            "num_epochs": 100,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": 0.0006,
            "clip": 5,
            "early_stop_tolerance": 4
        }

        self.model_params = {
            "graph_model": {
                "input_dim": 3,
                "hidden_dims": [30, 20, 10],
                "num_layers": 3,
                "filter_sizes": [3, 3, 3],
                "bias": True,
                "normalization": "sym",
            }
        }
