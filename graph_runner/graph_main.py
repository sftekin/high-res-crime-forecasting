from config import Config
from data_generators.graph_creator import GraphCreator
from batch_generators.graph_generator import GraphGenerator


def run():
    config = Config()
    data_creator = GraphCreator(data_params=config.data_params,
                                graph_params=config.graph_params)
    loaded = data_creator.load()
    if not loaded:
        print(f"Data is not found in {data_creator.save_dir}. Starting data creation...")
        data_creator.create()
    else:
        print(f"Data found. Data is loaded from {data_creator.save_dir}.")

    generator = GraphGenerator(node_features=data_creator.node_features,
                               labels=data_creator.labels,
                               edge_index=data_creator.edge_index,
                               batch_gen_params=config.batch_gen_params["graph"])

    for batch in generator.generate(dataset_name="train"):
        print(batch)

    print()


if __name__ == '__main__':
    run()
