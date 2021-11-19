class Config:
    def __init__(self):
        self.data_params = {
            "simulation_mode": False,
            "sim_data_path": "simulation.csv",
            "data_raw_path": "chicago_raw.csv",
            "coord_range": [[41.60, 42.05], [-87.9, -87.5]],
            "temporal_res": 24,
            "time_range": ("2015-01-01", "2019-01-02"),
            "crime_categories": ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE',
                                 'NARCOTICS', 'ROBBERY', 'ASSAULT',
                                 'DECEPTIVE PRACTICE', 'BURGLARY'],
            "simulation_categories": ["group_0", "group_1", "group_2", "group_3"],
            "plot": False
        }
        self.grid_params = {
            "spatial_res": (50, 33),  # 1kmx1km
            "include_side_info": False
        }

        self.graph_params = {
            "event_threshold": 5000,
            "include_side_info": False,
            "min_cell_size": (2, 2),
            "normalize_coords": True,
            "k_nearest": 5,
            "use_calendar": True
        }

        self.batch_gen_params = {}
        self.trainer_params = {}
        self.model_params = {}
        self.experiment_params = {}
