"""代理模型工厂类"""
import numpy as np
from smt.surrogate_models import KRG, RBF, KPLS, KPLSK
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def get_surrogate_model(name):
    if name == None or name == "":
        return None
    return surrogate_list[name]

def predict_value(self, X):
    X_ = np.array([X])
    y = self.predict_values(X_)[0][0]
    return y

surrogate_list = {
    "kriging": KRG,
    "rbf": RBF,
    "kpls": KPLS,
    "kplsk": KPLSK,
    'sk_gp': GaussianProcessRegressor,
}

class SurrogateModel:
    """
    代理模型
    """
    def __init__(self, type, setting=None):
        """来源，可选值是'sklearn'或者'smt'"""
        self.from_ = "sklearn" if type.startswith("sk") else "smt"
        self.type = type
        self.setting = setting
        self.model = None


    
    def init(self):
        pass
    
    def train(self, X, y):
        if self.from_ == "sklearn":
            self.model.fit(X, y)
        elif self.from_ == "smt":
            self.model.set_training_values(X, y)
            self.model.train()
    
    def predict_values(self, X):
        if self.from_ == "sklearn":
            return self.model.predict(X)
        elif self.from_ == "smt":
            return self.model.predict_values(X)
    
    def predict_variances(self, X):
        if self.from_ == "sklearn":
            _, std = self.model.predict(X, return_std=True)
            return std
        elif self.from_ == "smt":
            return self.model.predict_variances(X)

class SurrogateFactory:
    """
    代理模型工厂类

    eg.
    >>> surrogate_model_type = ["kriging"] # 代理模型名称，可选代理模型见surrogate.py
    >>> surr_setting = {
        "print_training": False,
        "print_prediction": False,
        "print_problem": False,
        "print_solver": False,
    }
    >>> sf = SurrogateFactory(surrogate_model_type, surr_setting)
    >>> surrogate_model = sf.get_sm(X, y)
    """
    def __init__(self, type_list, setting=None):
        self.type_list = type_list
        self.is_single_sm = True if len(type_list) == 1 else False
        self.setting = setting
    
    def get_sm(self, X, y):
        if self.is_single_sm:
            return self._get_surr_model(self.type_list[0], X, y)
        else:
            surrogate_models = []
            for type in self.type_list:
                sm = self._get_surr_model(type, X, y)
                surrogate_models.append(sm)
            return surrogate_models
    
    def _get_surr_model(self, type, X, y):
        sm_class = self._get_surrogate_model_class(type)
        if type == "gp":
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            sm = sm_class(kernel=kernel, n_restarts_optimizer=9)
            sm.fit(X, y)
            setattr(sm, 'fit_one', predict_value.__get__(sm))
        else:
            sm = sm_class(print_training=self.setting["print_training"],
                        print_prediction=self.setting["print_prediction"], 
                        print_problem=self.setting["print_problem"], 
                        print_solver=self.setting["print_solver"])
            sm.set_training_values(X, y)
            sm.train()
            # sm.fit_one = predict_value
            setattr(sm, 'fit_one', predict_value.__get__(sm))
        return sm

    def _get_surrogate_model_class(self, name):
        return get_surrogate_model(name)
    