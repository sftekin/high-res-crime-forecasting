from config import Config
from data_generators.grid_creator import GridCreator
from batch_generators.batch_generator import BatchGenerator
from models.convlstm import ConvLSTM
from trainer.trainer import Trainer


def run():
    config = Config()
    grid_creator = GridCreator(data_params=config.data_params,
                               grid_params=config.grid_params)

    if not grid_creator.check_is_created():
        print(f"Data is not found in {grid_creator.grid_save_dir}. Starting data creation...")
        grid = grid_creator.create_grid()
    else:
        grid = grid_creator.load_grid(mode="all")
        print(f"Data found. Data is loaded from {grid_creator.grid_save_dir}.")

    config.batch_gen_params["dataset_name"] = "grid"
    generator = BatchGenerator(in_data=grid,
                               labels=grid,
                               batch_gen_params=config.batch_gen_params)

    model = ConvLSTM(device=config.graph_trainer_prams["device"],
                     **config.model_params["convlstm"])

    trainer = Trainer(**config.graph_trainer_prams)
    trainer.fit(model=model, batch_generator=generator)


if __name__ == '__main__':
    run()
