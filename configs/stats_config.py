from configs.config import Config


class StatsConfig(Config):
    def __init__(self):
        super(StatsConfig, self).__init__()
        self.batch_gen_params = {
            "train_size": 6.5,  # months
            "val_size": 0.5,  # months
            "test_size": 1,  # months
        }

        self.model_params = {
            "arima": {
                "order": (1, 0, 1),
                "seasonal_order": (0, 0, 0, 7),
                "trend": "c"
            }
        }
