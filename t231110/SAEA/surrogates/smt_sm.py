"""smt库的代理模型封装类"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from smt.surrogate_models import KRG, RBF, KPLS, KPLSK
from SAEA.surrogates.base import SurrogateModel


class SmtSurrogateModel(SurrogateModel):
    def __init__(self, type=None, setting=None):
        super().__init__(type=type, setting=setting)
        self.from_ = "smt"

    def train(self, X, y):
        self.model.set_training_values(X, y)
        self.model.train()

    def predict(self, X):
        X_ = np.array(X)
        return self.model.predict_values(X_).flatten()
    
    def predict_one(self, X):
        X_ = np.array([X])
        y = self.predict(X_)
        y = y.flatten()[0]
        return y
    
    def predict_variances(self, X):
        return self.model.predict_variances(X).flatten()
    
    def predict_variances_one(self, X):
        X_ = np.array([X])
        var = self.predict_variances(X_)
        var = var.flatten()[0]
        return var
    
    def predict_value_variance(self, X):
        X_ = np.array(X)
        y = self.model.predict_values(X_)
        var = self.model.predict_variances(X_)
        return y.flatten(), var.flatten()
    

class SmtKriging(SmtSurrogateModel):
    def __init__(self, type=None, setting=None):
        super().__init__(type="kriging", setting=setting)
        self.predict_variances_flag = True

    def init(self):
        self.name = "smt_kriging"
        self.model = KRG(theta0=[1e-2], print_global=False)


class SmtKPLSK(SmtSurrogateModel):
    def __init__(self, type=None, setting=None):
        super().__init__(type="kplsk", setting=setting)
        self.predict_variances_flag = True

    def init(self):
        self.name = "smt_kplsk"
        self.model = KPLSK(theta0=[1e-2], print_global=False)


class SmtRBF(SmtSurrogateModel):
    def __init__(self, type=None, setting=None):
        super().__init__("rbf", setting=setting)

    def init(self):
        self.name = "smt_rbf"
        self.model = RBF(print_global=False)
    
    def predict_variances(self, X):
        return None