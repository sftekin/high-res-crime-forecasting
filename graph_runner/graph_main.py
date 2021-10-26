import os
import glob
import pickle as pkl

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
            grid = grid_creator.load_grid(dataset_name="all")
        else:
            grid = grid_creator.create_grid(dataset_name="all")
        events = grid[..., [2]]
        print(f"Data is not found in {graph_creator.graph_save_dir}. Starting data creation...")
        graph_creator.create_graph(grid=events)
    else:
        print(f"Data found. Data is loaded from {graph_creator.graph_save_dir}.")

    events = graph_creator.labels > 0
    events = events.astype(int)
    generator = BatchGenerator(in_data=graph_creator.node_features,
                               labels=events,
                               edge_index=graph_creator.edge_index,
                               regions=graph_creator.regions,
                               batch_gen_params=config.batch_gen_params)
    model = GraphModel(device=config.trainer_params["device"],
                       node_count=graph_creator.node_features.shape[1],
                       **config.model_params["graph_model"])
    trainer = Trainer(**config.trainer_params, node2cell=graph_creator.node2cells)

    # train model and get the train, val, train+val losses
    train_losses = trainer.fit(model=model, batch_generator=generator)

    # perform prediction
    test_loss = trainer.transform(model=model, batch_generator=generator)
    train_losses.append(test_loss)

    # saving model and losses
    model = model.to("cpu")
    save_dir = get_save_dir(model_name="graph_model")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    loss_path = os.path.join(save_dir, "loss.pkl")
    model_path = os.path.join(save_dir, "model.pkl")
    trainer_path = os.path.join(save_dir, "trainer.pkl")
    paths = [loss_path, model_path, trainer_path]
    objs = [train_losses, model, trainer]
    for path, obj in zip(paths, objs):
        with open(path, "wb") as f:
            pkl.dump(obj, f)


def get_save_dir(model_name):
    results_dir = "results"
    save_dir = os.path.join(results_dir, model_name)
    num_exp_dir = len(glob.glob(os.path.join(save_dir, 'exp_*')))
    save_dir = os.path.join(save_dir, "exp_" + str(num_exp_dir + 1))
    return save_dir


if __name__ == '__main__':
    run()
