from config import Config


class StatsConfig(Config):
    def __init__(self):
        super(StatsConfig, self).__init__()
        self.batch_gen_params = {
            "test_size": 266,
            "val_ratio": 0.20,
        }

        self.model_params = {
            "arima": {
                "order": (1, 0, 1),
                "seasonal_order": (0, 0, 0, 7),
                "trend": "c"
            }
        }
