
from graph_runner.config import Config
from data_generators.graph_creator import GraphCreator


def run():
    config = Config()
    data_creator = GraphCreator(data_params=config.data_params,
                                graph_params=config.graph_params)
    data_creator.create()
    print()


if __name__ == '__main__':
    run()
