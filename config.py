

class Config:
    def __init__(self):
        self.data_params = {
            "data_raw_path": "chicago_raw.csv",
            "coord_range": [[41.60, 42.05], [-87.9, -87.5]],
            "spatial_res": (50, 33),
            "temporal_res": 1,
            "time_range": ("2015-01-01", "2019-11-01")
        }
