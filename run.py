
from config import Config
from data_generators.data_creator import DataCreator


def run():
    config = Config()
    data_creator = DataCreator(**config.data_params)
    data_creator.create()
    print()


if __name__ == '__main__':
    run()
