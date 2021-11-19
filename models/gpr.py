from sklearn.gaussian_process import GaussianProcessRegressor as GPR


class GaussianProcessRegression:
    def __init__(self, kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0, normalize_y=False, copy_X_train=True,
                 random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

        self.__clf = None

    @property
    def clf(self):
        return self.__clf

    def fit(self, X, y):
        clf_params = {key: val
                      for key, val in self.__dict__.items()
                      if "__" not in key}  # get non private params
        self.__clf = GPR(**clf_params)
        self.__clf.fit(X, y)

    def predict(self, X):
        pred = self.clf.predict(X)
        return pred
