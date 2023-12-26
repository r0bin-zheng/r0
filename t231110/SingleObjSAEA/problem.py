# 定义单目标问题

import numpy as np
import deap.benchmarks as bm
from pymoo.core.problem import Problem

# 代理模型辅助的问题
# 单代理辅助的问题
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

# 代理集合辅助的问题，使用代理之间的方差作为不确定性，不确定性+预测值作为输出
class ProblemWithSurrogates(Problem):
    def __init__(self, problem, n_var, n_obj, n_constr=0, xl=None, xu=None, surrogates=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.real_problem = problem
        self.surrogates = surrogates

    def _evaluate(self, X, out, *args, **kwargs):
        if self.surrogates is not None and len(self.surrogates) > 0:
            y_pred = []

            # 计算预测值
            for surrogate in self.surrogates:
                y_pred.append(surrogate.predict_values(X))
            
            # 计算不确定性
            y_pred = np.array(y_pred) # 计算预测与平均值的差的平方，作为不确定性
            y_pred = np.squeeze(y_pred) # 去掉最底的一维
            y_pred = np.transpose(y_pred) # 两个维度互换
            y_var = np.var(y_pred, axis=1) # 计算方差
            
            # 合并
            y_pred = np.column_stack([y_pred, y_var]) # 将方差合并到y_pred中
            out["F"] = y_pred
        else:
            out["F"] = np.array([self.real_problem(x) for x in X])

def get_problem(name, setting=None, surrogate=None, surrogates=None):
    if name == None:
        return None
    config = problem_list[name]["default_conf"]
    if setting:
        _update_config(config, setting)

    if surrogates is not None:
        return ProblemWithSurrogates(problem=problem_list[name]["problem_func"],
                                    n_var=config["n_var"],
                                    n_obj=config["n_obj"],
                                    n_constr=config["n_constr"],
                                    xl=config["xl"],
                                    xu=config["xu"],
                                    surrogates=surrogates)
    else:
        return ProblemWithSurrogate(problem=problem_list[name]["problem_func"],
                                    n_var=config["n_var"],
                                    n_obj=config["n_obj"],
                                    n_constr=config["n_constr"],
                                    xl=config["xl"],
                                    xu=config["xu"],
                                    surrogate=surrogate)

def _update_config(config, setting):
    if setting is None:
        return config
    for key in setting:
        config[key] = setting[key]




# 连续变量问题


# ackley函数
# 最小化问题
# 上下界[−15,30]
# 最优解[0,0,...,0]，f(x)=0
def ackley(individual):
    return bm.ackley(individual)

# bohachevsky函数
# 最小化问题
# 上下界[-100,100]
# 最优解[0,0,...,0]，f(x)=0
def bohachevsky(individual):
    return bm.bohachevsky(individual)

# griewank函数
# 最小化问题
# 上下界[-600,600]
# 最优解[0,0,...,0]，f(x)=0
def griewank(individual):
    return bm.griewank(individual)

# h1函数，二维
# 最大化问题
# 上下界[-100,100]
# 最优解8.6998,6.7665, f(x)=2n
def h1(individual):
    return bm.h1(individual)

# himmelblau函数，二维
# 最小化问题
# 上下界[-6,6]
# 最优解(3,2), (-2.805118,3.131312), (-3.779310,-3.283186), (3.584428,-1.848126)
def himmelblau(individual):
    return bm.himmelblau(individual)

# 字典
problem_list = {
    "ackley": {
        "name": "ackley",
        "default_conf": {
            "n_var": 10,
            "n_obj": 1,
            "n_constr": 0,
            "xl": -15,
            "xu": 30,
        },
        "problem_func": ackley,
        "best_sln": 0,
    },
    "bohachevsky": {
        "name": "bohachevsky",
        "default_conf": {
            "n_var": 2,
            "n_obj": 1,
            "n_constr": 0,
            "xl": -100,
            "xu": 100,
        },
        "problem_func": bohachevsky,
        "best_sln": 0,
    },
    "griewank": {
        "name": "griewank",
        "default_conf": {
            "n_var": 10,
            "n_obj": 1,
            "n_constr": 0,
            "xl": -600,
            "xu": 600,
        },
        "problem_func": griewank,
        "best_sln": 0,
    },
    "himmelblau": {
        "name": "himmelblau",
        "default_conf": {
            "n_var": 2,
            "n_obj": 1,
            "n_constr": 0,
            "xl": -6,
            "xu": 6,
        },
        "problem_func": himmelblau,
        "best_sln": {
            "x1": [3, 2],
            "f1": 0,
            "x2": [-2.805118, 3.131312],
            "f2": 0,
            "x3": [-3.779310, -3.283186],
            "f3": 0,
            "x4": [3.584428, -1.848126],
            "f4": 0,
        },
    },
}
