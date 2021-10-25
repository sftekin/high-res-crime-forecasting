from graph_config import GraphConfig
from data_generators.graph_creator import GraphCreator
from data_generators.grid_creator import GridCreator
from batch_generators.batch_generator import BatchGenerator
from trainer.trainer import Trainer
from models.graph_model import GraphModel


def run():
    config = GraphConfig()
    graph_creator = GraphCreator(data_params=config.data_params,
                                 graph_params=config.graph_params)
    loaded = graph_creator.load()
    if not loaded:
        grid_creator = GridCreator(data_params=config.data_params, grid_params=config.grid_params)
        if grid_creator.check_is_created():
            grid = grid_creator.load_grid(mode="all")
        else:
            grid = grid_creator.create_grid()
        print(f"Data is not found in {graph_creator.graph_save_dir}. Starting data creation...")
        graph_creator.create_graph(grid=grid)
    else:
        print(f"Data found. Data is loaded from {graph_creator.graph_save_dir}.")

    config.batch_gen_params["dataset_name"] = "graph"
    generator = BatchGenerator(in_data=graph_creator.node_features,
                               labels=graph_creator.labels,
                               edge_index=graph_creator.edge_index,
                               regions=graph_creator.regions,
                               batch_gen_params=config.batch_gen_params)

    model = GraphModel(device=config.trainer_params["device"],
                       node_count=graph_creator.node_features.shape[1],
                       **config.model_params["graph_model"])

    trainer = Trainer(**config.trainer_params, node2cell=graph_creator.node2cells)
    trainer.fit(model=model, batch_generator=generator)


if __name__ == '__main__':
    run()
