from configs.config import Config


class GraphConfig(Config):
    def __init__(self):
        super(GraphConfig, self).__init__()
        self.experiment_params = {
            "train_size": 10,  # months
            "val_size": 1,  # months
            "test_size": 1,  # months
        }

        self.batch_gen_params = {
            "window_in_len": 10,
            "window_out_len": 1,
            "batch_size": 5,
            "shuffle": False,
            "normalize_flag": False,
            "normalize_methods": ["min_max"],
            "normalization_dims": "all",
            "dataset_name": "graph"
        }

        self.trainer_params = {
            "device": 'cuda',
            "num_epochs": 100,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0,
            "learning_rate": 0.003,
            "clip": 10,
            "early_stop_tolerance": 5,
            "loss_function": "prob_loss",  # likelihood or prob_loss
            "node_dist_constant": 0  # available only in likelihood
        }

        self.model_params = {
            "graph_conv_gru": {
                "input_dim": 16,  # 52 if side info is included else 16
                "hidden_dims": [50, 20, 10],
                "num_layers": 3,
                "filter_sizes": [3, 3, 3],
                "bias": True,
                "normalization": "sym",
            },
            "lstm": {
                "input_dim": 16,
                "hidden_dim": 100,  # number of nodes also
                "num_layers": 3,
                "drop_out": 0.1
            }
        }
