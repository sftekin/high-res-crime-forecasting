from config import Config


class GridConfig(Config):
    def __init__(self):
        super(GridConfig, self).__init__()
        self.batch_gen_params = {
            "test_size": 266,
            "val_ratio": 0.20,
            "window_in_len": 10,
            "window_out_len": 1,
            "batch_size": 16,
            "shuffle": False,
            "normalize_flag": True,
            "normalize_methods": ["min_max"],
            "normalization_dims": "all",
            "dataset_name": "grid"
        }

        self.trainer_params = {
            "device": 'cuda',
            "num_epochs": 100,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": 0.001,
            "clip": 5,
            "early_stop_tolerance": 4,
            "loss_function": "BCE"
        }

        self.model_params = {
            "convlstm": {
                "input_size": (100, 66),
                "window_in": 10,  # should be same with batch_gen["window_in_len"]
                "window_out": 1,  # should be same with batch_gen["window_out_len"]
                "num_layers": 3,
                "encoder_params": {
                    "input_dim": 1,
                    "hidden_dims": [1, 16, 32],
                    "kernel_size": [5, 3, 3],
                    "bias": False,
                    "peephole_con": False
                },
                "decoder_params": {
                    "input_dim": 32,
                    "hidden_dims": [32, 16, 1],
                    "kernel_size": [3, 3, 3],
                    "bias": False,
                    "peephole_con": False
                }
            },
            "convlstm_one_block": {
                "input_dim": 1,
                "input_size": (100, 66),
                "hidden_dims": [32, 16, 1],
                "kernel_sizes": [3, 3, 3],
                "window_in": 10,
                "window_out": 1,
                "num_layers": 3,
                "bias": True
            }
        }
