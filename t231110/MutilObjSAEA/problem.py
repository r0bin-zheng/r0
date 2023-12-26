# 定义单目标问题

import numpy as np
import deap.benchmarks as bm
from pymoo.core.problem import Problem
from pymoo.problems import get_problem
from pymoo.util.misc import stack

# 代理模型辅助的问题
class ProblemWithSurrogate(Problem):
    def __init__(self, problem, n_var, n_obj, n_constr=0, xl=None, xu=None, surrogate=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.real_problem = problem
        self.surrogate = surrogate

    def _evaluate(self, X, out, *args, **kwargs):
        if self.surrogate:
            out["F"] = self.surrogate.predict_values(X)
        else:
            out["F"] = np.array([self.real_problem(ind) for ind in X])

    def _evaluate(self, X, out, *args, **kwargs):
        if self.surrogate is not None:
            # 使用代理模型进行预测
            Y = self.surrogate.predict_values(X)
        else:
            Y = []
            for x in X:
                y = self.real_problem._evaluate(x, return_values_of=["F"])
                Y.append(y)
            Y = stack(Y, axis=0)
        out["F"] = Y

def get_problem(name="", n_var=1, n_obj=2, n_constr=0, xl=None, xu=None, surrogate=None, setting=None):
    if setting == None:
        if name == "":
            return None
        return ProblemWithSurrogate(problem_list[name], n_var, n_obj, n_constr, xl, xu, surrogate)
    else:
        return ProblemWithSurrogate(problem_list[setting["name"]],
                                    n_var=setting["n_var"],
                                    n_obj=setting["n_obj"],
                                    n_constr=setting["n_constr"],
                                    xl=setting["xl"],
                                    xu=setting["xu"],
                                    surrogate=surrogate)

# 连续变量问题


# ztd1函数
# 目标函数2
def zdt1():
    return get_problem("zdt1")

# 字典
problem_list = {
    "ztd1": zdt1,
}
