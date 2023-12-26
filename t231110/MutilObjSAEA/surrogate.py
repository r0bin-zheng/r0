# 基于smt的代理模型
from smt.surrogate_models import KRG, RBF, KPLS

def get_surrogate_model(name):
    if name == None or name == "":
        return None
    return surrogate_list[name]

# Kriging

# RBF

# KPLS

# 列表
surrogate_list = {
    "kriging": KRG,
    "rbf": RBF,
    "kpls": KPLS,
}