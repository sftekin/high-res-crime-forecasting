from configs.config import Config


class StatsConfig(Config):
    def __init__(self):
        super(StatsConfig, self).__init__()
        self.experiment_params = {
            "train_size": 18,  # months
            "val_size": 3,  # months
            "test_size": 3,  # months
        }

        self.model_params = {
            "arima": {
                "order": (1, 0, 1),
                "seasonal_order": (0, 0, 0, 7),
                "trend": "c"
            },
            "svr": {
                "kernel": "rbf",
                "degree": 3,
                "gamma": "scale",
                "coef0": 0.0,
                "tol": 4e-4,
                "C": 5.0,
                "epsilon": 0.001,
                "shrinking": True,
                "cache_size": 500,
                "verbose": False,
                "max_iter": -1
            },
            "random_forest": {
                'n_estimators': 200,
                'criterion': 'mse',
                'max_depth': 2,
                'min_samples_split': 3,
                'min_samples_leaf': 3,
                'min_weight_fraction_leaf': 0.0,
                'max_features': 'auto',
                'max_leaf_nodes': None,
                'min_impurity_decrease': 0.0,
                'bootstrap': True,
                'oob_score': False,
                'n_jobs': None,
                'random_state': None,
                'warm_start': False,
                'ccp_alpha': 0.0,
                'max_samples': None
            }
        }
