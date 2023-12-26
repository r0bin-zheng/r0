# 基于smt的代理模型
from smt.surrogate_models import KRG, RBF, KPLS

def get_surrogate_model(name):
    if name == None or name == "":
        return None
    return surrogate_list[name]

# Kriging

# RBF

# KPLS

# R1


# 列表
surrogate_list = {
    "kriging": KRG,
    "rbf": RBF,
    "kpls": KPLS,
    "r1": None
}

class SurrogateFactory:
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
        sm = sm_class(print_training=self.setting["print_training"],
                      print_prediction=self.setting["print_prediction"], 
                      print_problem=self.setting["print_problem"], 
                      print_solver=self.setting["print_solver"])
        sm.set_training_values(X, y)
        sm.train()
        return sm

    def _get_surrogate_model_class(self, name):
        return get_surrogate_model(name)