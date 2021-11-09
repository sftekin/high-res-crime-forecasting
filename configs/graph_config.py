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
            "num_epochs": 80,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0,
            "learning_rate": 0.001,
            "clip": 10,
            "early_stop_tolerance": 10,
            "loss_function": "likelihood",  # or prob_loss
            "node_dist_constant": 0.2
        }

        self.model_params = {
            "graph_model": {
                "input_dim": 16,  # 52 if side info is included else 16
                "hidden_dims": [50, 10, 50],
                "num_layers": 3,
                "filter_sizes": [5, 3, 5],
                "bias": True,
                "normalization": "sym",
            }
        }
