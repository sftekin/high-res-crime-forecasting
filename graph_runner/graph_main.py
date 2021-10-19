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
        data_creator.create_graph()
    else:
        print(f"Data found. Data is loaded from {data_creator.save_dir}.")

    generator = GraphGenerator(node_features=data_creator.node_features,
                               labels=data_creator.labels,
                               edge_index=data_creator.edge_index,
                               batch_gen_params=config.batch_gen_params["graph"])

    for batch in generator.generate(dataset_name="train"):
        print(batch)

    print()

    # from scipy.stats import multivariate_normal as mvn
    # import numpy as np
    # mean = np.array([1, 5])
    # covariance = np.array([[1, 0.3], [0.3, 1]])
    # dist = mvn(mean=mean, cov=covariance)
    # print("CDF:", dist.cdf(np.array([2, 4])))


if __name__ == '__main__':
    run()
