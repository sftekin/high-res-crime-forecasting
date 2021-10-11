
from data_generators.data_creator import DataCreator


class GraphCreator(DataCreator):
    def __init__(self, data_params, graph_params):
        super(DataCreator, self).__init__(data_params)

    def create(self):
        crime_df = super().create()
