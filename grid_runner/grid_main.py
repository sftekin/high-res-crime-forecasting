import os

from grid_config import GridConfig
from data_generators.grid_creator import GridCreator
from batch_generators.batch_generator import BatchGenerator
from models.convlstm import ConvLSTM
from models.convlstm_one_block import ConvLSTMOneBlock
from trainer.trainer import Trainer
from helpers.static_helper import get_save_dir

model_dispatcher = {
    "convlstm": ConvLSTM,
    "convlstm_one_block": ConvLSTMOneBlock
}


def run():
    config = GridConfig()
    grid_creator = GridCreator(data_params=config.data_params,
                               grid_params=config.grid_params)

    if not grid_creator.check_is_created():
        print(f"Data is not found in {grid_creator.grid_save_dir}. Starting data creation...")
        grid = grid_creator.create_grid()
    else:
        grid = grid_creator.load_grid(dataset_name="all")
        print(f"Data found. Data is loaded from {grid_creator.grid_save_dir}.")

    # create save path
    save_dir = get_save_dir(model_name="graph_model")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    events = grid[..., [2]] > True
    events = events.astype(int)
    generator = BatchGenerator(in_data=grid,
                               labels=events,
                               batch_gen_params=config.batch_gen_params)

    model_name = "convlstm_one_block"
    model = model_dispatcher[model_name](device=config.trainer_params["device"],
                                         **config.model_params["convlstm_one_block"])

    trainer = Trainer(**config.trainer_params, save_dir=save_dir)

    # train model
    trainer.fit(model=model, batch_generator=generator)

    # perform prediction
    trainer.transform(model=model, batch_generator=generator)


if __name__ == '__main__':
    run()
