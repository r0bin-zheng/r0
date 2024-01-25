"""代理模型基类"""

import numpy as np


class SurrogateModel:
    """
    代理模型

    eg.
    >>> sm = SurrogateModel("kriging")
    >>> sm.train(X, y)
    >>> sm.predict(X)
    """
    def __init__(self, type=None, setting=None):
        self.type = type
        self.setting = setting
        self.predict_variances_flag = False
        self.init()

    def init(self):
        pass
    
    def train(self, X, y):
        pass
    
    def predict(self, X):
        pass

    def predict_one(self, X):
        pass
    
    def predict_variances(self, X):
        pass

    def predict_variances_one(self, X):
        pass
