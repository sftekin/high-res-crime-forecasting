from sklearn.svm import SVR


class SupportVectorRegression:
    def __init__(self, kernel="rbf", degree=3, gamma="scale", coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1, ):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.__clf = None

    @property
    def clf(self):
        return self.__clf

    def fit(self, X, y):
        clf_params = {key: val for key, val in self.__dict__.items() if '__' not in key}  # get non private params
        self.__clf = SVR(**clf_params)
        self.__clf.fit(X, y)

    def predict(self, X):
        pred = self.clf.predict(X)
        return pred
