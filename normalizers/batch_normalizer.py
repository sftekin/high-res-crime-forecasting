import numpy as np
from sklearn.preprocessing import StandardScaler


class BatchNormalizer:
    def __init__(self, normalize_methods, normalization_dims):
        self.norm_methods = []
        for method in normalize_methods:
            self.norm_methods.append(getattr(self, method))
        self.normalization_dims = normalization_dims

    def norm(self, x):
        out = x
        for norm_method in self.norm_methods:
            out = norm_method(out)
        return out

    def min_max(self, in_arr):
        norm_dims = self.normalization_dims if self.normalization_dims != "all" else range(in_arr.shape[-1])
        for i in norm_dims:
            arr = in_arr[..., i]
            min_a = arr.min()
            max_a = arr.max()
            in_arr[..., i] = (arr - min_a) / (max_a - min_a)

        return in_arr

    def standardize(self, in_arr):
        scalers = []
        norm_dims = self.normalization_dims if self.normalization_dims != "all" else range(in_arr.shape[1])
        for i in norm_dims:
            arr = in_arr[:, i]
            scaler = StandardScaler()
            in_arr[:, i] = scaler.fit_transform(arr.reshape(-1, 1))[:, 0]
            scalers.append(scaler)
        self.stats["standardize"] = scalers
        return in_arr

    def log(self, in_arr):
        norm_dims = self.normalization_dims if self.normalization_dims != "all" else range(in_arr.shape[1])
        in_arr[:, norm_dims] = np.log1p(in_arr[:, norm_dims])
        return in_arr
