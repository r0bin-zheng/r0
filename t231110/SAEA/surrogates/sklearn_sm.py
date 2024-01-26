"""sklearn库的代理模型封装"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from SAEA.surrogates.base import SurrogateModel
from SAEA.surrogates.my_sklearn_sm import MyGPR


class SklearnSurrogateModel(SurrogateModel):
    def __init__(self, type=None, setting=None):
        super().__init__(type=type, setting=setting)
        self.from_ = "sklearn"

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X).flatten()
    
    def predict_one(self, X):
        X_ = np.array([X])
        y = self.predict(X_)
        y = y.flatten()[0]
        return y
    
    def predict_variances(self, X):
        _, std = self.model.predict(X, return_std=True)
        return std.flatten()
    
    def predict_variances_one(self, X):
        X_ = np.array([X])
        var = self.predict_variances(X_)
        return var.flatten()[0]
    
    def predict_value_variance(self, X):
        y, var = self.model.predict(X, return_std=True)
        return y.flatten(), var.flatten()
    

class SklearnGaussianProcessRegressor(SklearnSurrogateModel):
    def __init__(self, type=None, setting=None):
        super().__init__("kriging", setting=setting)
        self.predict_variances_flag = True

    def init(self):
        self.name = "sk_gp"
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3))
        self.model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=9)
        
class MySklearnGaussianProcessRegressor(SklearnSurrogateModel):
    def __init__(self, type=None, setting=None):
        super().__init__("kriging", setting=setting)
        self.predict_variances_flag = True

    def init(self):
        self.name = "sk_gp"
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3))
        self.model = MyGPR(kernel=kernel, n_restarts_optimizer=9)
