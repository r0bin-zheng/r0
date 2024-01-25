"""多代理类"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from SAEA.surrogates.base import SurrogateModel
from SAEA.surrogates.sm_mapping import SM_DICT


class MultiSurrogate(SurrogateModel):
    def __init__(self, types, setting):
        super().__init__(types, setting)
        self.from_ = "multi"
        self.predict_strategy = "mean"
        """
        代理模型集合的预测策略
        - mean: 平均值
        - min: 最小值
        - max: 最大值
        """
        self.variance_strategy = "mean"
        """
        代理模型集合的方差策略
        - mean: 平均值
        - min: 最小值
        - max: 最大值
        """

    def init(self):
        self.model = []
        for type in self.type:
            self.model.append(SM_DICT[type]())

    def train(self, X, y):
        for m in self.model:
            m.train(X, y)
    
    def predict(self, X):
        if self.predict_strategy == "mean":
            return np.mean([m.predict(X) for m in self.model], axis=0)
        elif self.predict_strategy == "min":
            return np.min([m.predict(X) for m in self.model], axis=0)
        elif self.predict_strategy == "max":
            return np.max([m.predict(X) for m in self.model], axis=0)
    
    def predict_one(self, X):
        if self.predict_strategy == "mean":
            return np.mean([m.predict_one(X) for m in self.model], axis=0)
        elif self.predict_strategy == "min":
            return np.min([m.predict_one(X) for m in self.model], axis=0)
        elif self.predict_strategy == "max":
            return np.max([m.predict_one(X) for m in self.model], axis=0)
    
    def predict_variances(self, X):
        var_for_each_model = []
        for m in self.model:
            if m.predict_variances_flag:
                var_for_each_model.append(m.predict_variances(X))
        if len(var_for_each_model) == 0:
            return None
        else:
            if self.variance_strategy == "mean":
                return np.mean(var_for_each_model, axis=0)
            elif self.variance_strategy == "min":
                return np.min(var_for_each_model, axis=0)
            elif self.variance_strategy == "max":
                return np.max(var_for_each_model, axis=0)
            
    def predict_variances_one(self, X):
        var_for_each_model = []
        for m in self.model:
            if m.predict_variances_flag:
                var_for_each_model.append(m.predict_variances_one(X))
        if len(var_for_each_model) == 0:
            return None
        else:
            if self.variance_strategy == "mean":
                return np.mean(var_for_each_model, axis=0)
            elif self.variance_strategy == "min":
                return np.min(var_for_each_model, axis=0)
            elif self.variance_strategy == "max":
                return np.max(var_for_each_model, axis=0)
            
    def predict_value_variance(self, X):
        value_for_each_model = []
        var_for_each_model = []
        for m in self.model:
            if m.predict_variances_flag:
                value, var = m.predict_value_variance(X)
                value_for_each_model.append(value)
                var_for_each_model.append(var)
        if len(value_for_each_model) == 0:
            return None
        else:
            if self.variance_strategy == "mean":
                return np.mean(value_for_each_model, axis=0), np.mean(var_for_each_model, axis=0)
            elif self.variance_strategy == "min":
                return np.min(value_for_each_model, axis=0), np.min(var_for_each_model, axis=0)
            elif self.variance_strategy == "max":
                return np.max(value_for_each_model, axis=0), np.max(var_for_each_model, axis=0)
