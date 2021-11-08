from sklearn.ensemble import RandomForestRegressor


class RandomForest:
    def __init__(self, n_estimators=100, criterion='mse', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, n_jobs=None,
                 random_state=None, warm_start=False, ccp_alpha=0.0, max_samples=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

        self.__clf = None

    @property
    def clf(self):
        return self.__clf

    def fit(self, X, y):
        rf_params = {key: val for key, val in self.__dict__.items() if '__' not in key}  # get non private params
        self.__clf = RandomForestRegressor(**rf_params)
        self.__clf.fit(X, y)

    def predict(self, X):
        pred = self.clf.predict(X)
        return pred


