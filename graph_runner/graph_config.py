from config import Config


class GraphConfig(Config):
    def __init__(self):
        super(GraphConfig, self).__init__()
        self.batch_gen_params = {
            "test_size": 266,
            "val_ratio": 0.20,
            "window_in_len": 10,
            "window_out_len": 1,
            "batch_size": 5,
            "shuffle": False,
            "normalize_flag": False,
            "normalize_methods": ["min_max"],
            "normalization_dims": "all",
        }

        self.trainer_params = {
            "device": 'cuda',
            "num_epochs": 100,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": 0.0001,
            "clip": 5,
            "early_stop_tolerance": 4,
            "loss_function": "prob_loss"
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
