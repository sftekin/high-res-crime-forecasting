from configs.config import Config


class GridConfig(Config):
    def __init__(self):
        super(GridConfig, self).__init__()
        self.experiment_params = {
            "train_size": 10,  # months
            "val_size": 1,  # months
            "test_size": 1,  # months
        }

        self.batch_gen_params = {
            "window_in_len": 10,
            "window_out_len": 1,
            "batch_size": 16,
            "shuffle": False,
            "normalize_flag": False,
            "normalize_methods": ["min_max"],
            "normalization_dims": "all",
            "dataset_name": "grid"
        }

        self.trainer_params = {
            "device": 'cuda',
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0,
            "learning_rate": 0.001,
            "clip": 5,
            "early_stop_tolerance": 5,
            "loss_function": "MSE"
        }

        self.model_params = {
            "convlstm": {
                "input_size": (50, 33),
                "window_in": 5,  # should be same with batch_gen["window_in_len"]
                "window_out": 1,  # should be same with batch_gen["window_out_len"]
                "num_layers": 4,
                "encoder_params": {
                    "input_dim": 1,
                    "hidden_dims": [1, 16, 32, 64],
                    "kernel_size": [5, 3, 3, 3],
                    "bias": True,
                    "peephole_con": False
                },
                "decoder_params": {
                    "input_dim": 64,
                    "hidden_dims": [64, 32, 16, 1],
                    "kernel_size": [5, 3, 3, 3],
                    "bias": True,
                    "peephole_con": False
                }
            },
            "convlstm_one_block": {
                "input_dim": 1,
                "input_size": (50, 33),
                "hidden_dims": [32, 16, 1],
                "kernel_sizes": [3, 3, 3],
                "window_in": 10,
                "window_out": 1,
                "num_layers": 3,
                "bias": True
            },
            "fc_lstm": {
                "input_dim": 50 * 33,
                "hidden_dim": 50 * 33,  # number of nodes also
                "num_layers": 3,
                "drop_out": 0.1
            }
        }
