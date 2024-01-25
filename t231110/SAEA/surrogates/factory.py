"""代理模型工厂类"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.surrogates.sm_mapping import SM_DICT
from SAEA.surrogates.multi_sm import MultiSurrogate


class SurrogateModelFactory:
    """
    代理模型工厂类

    eg.

    单代理
    -------
    >>> types = ["smt_kriging"]
    >>> setting = {}
    >>> factory = SurrogateModelFactory(types, setting)
    >>> sm = factory.create()
    >>> sm.train(X, y)
    >>> sm.predict(X)
    
    多代理
    ------
    >>> types = ["smt_kriging", "sklearn_gpr"]
    >>> setting = {}
    >>> factory = SurrogateModelFactory(types, setting)
    >>> sm = factory.create()
    >>> sm.train(X, y)
    >>> sm.predict(X)
    """
    def __init__(self, types, setting):
        self.types = types
        self.setting = setting

    def create(self, X, y):
        if len(self.types) == 1:
            sm = SM_DICT[self.types[0]](self.types[0], self.setting)
        else:
            sm = MultiSurrogate(self.types, self.setting)
        sm.train(X, y)
        return sm