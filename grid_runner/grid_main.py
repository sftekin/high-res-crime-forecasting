from config import Config
from data_generators.grid_creator import GridCreator


def run():
    config = Config()
    data_creator = GridCreator(data_params=config.data_params,
                               grid_params=config.grid_params)

    if not data_creator.check_is_created():
        data_creator.create_grid()
        print()


if __name__ == '__main__':
    run()
