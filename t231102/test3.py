import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.core.problem import Problem
from smt.surrogate_models import RBF, QP, KPLS

# numOfFuncEval = 0

# 定义Ackley函数
def ackley(x, a=20, b=0.2, c=2*np.pi):
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x**2) / d))
    cos_term = -np.exp(sum(np.cos(c * x) for x in x) / d)
    return a + np.exp(1) + sum_sq_term + cos_term

def expensive_evaluation(x):
    return ackley(np.array(x))

# 定义问题
class MyProblem(Problem):
    def __init__(self, n_var=3, n_obj=3, n_constr=0, xl=-10, xu=10, surrogate1=None, surrogate2=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.surrogate1 = surrogate1
        self.surrogate2 = surrogate2

    def _evaluate(self, X, out, *args, **kwargs):
        if self.surrogate1 and self.surrogate2:
            y_pred1 = self.surrogate1.predict_values(X)
            y_pred2 = self.surrogate2.predict_values(X)
            uncertainty = np.abs(y_pred1 - y_pred2)  # 绝对差异作为不确定性
            print(np.column_stack([y_pred1, y_pred2, uncertainty]).shape)
            out["F"] = np.column_stack([y_pred1, y_pred2, uncertainty])
            # out["F"] = np.column_stack([(y_pred1 + y_pred2) / 2, uncertainty])  # 优化平均值和不确定性
        else:
            out["F"] = np.array([expensive_evaluation(x) for x in X])

# 初始化
nFE = 0  # 初始化函数评估次数
max_nFE = 500  # 最大函数评估次数
pop_size = 100  # 种群大小
n_offsprings = 25  # 每一代的后代数

# 初始种群的采样
lhs_sampling = get_sampling("real_lhs")
# X = lhs_sampling.do(MyProblem(), n_samples=pop_size)
# y = np.array([expensive_evaluation(xi) for xi in X])
X_data = lhs_sampling.do(MyProblem(), n_samples=pop_size)
X = np.array([ind.X for ind in X_data])
y = np.array([expensive_evaluation(ind) for ind in X]).reshape(-1, 1)

# 更新函数评估次数
nFE += pop_size

# 循环直到函数评估次数达到上限
while nFE < max_nFE:
    # 创建代理模型并训练
    surrogate1 = RBF()
    surrogate2 = KPLS(theta0=[1e-2])
    surrogate1.set_training_values(X, y)
    surrogate2.set_training_values(X, y)
    surrogate1.train()
    surrogate2.train()

    # 定义问题
    problem = MyProblem(surrogate1=surrogate1, surrogate2=surrogate2)

    # 使用代理模型进行进化算法
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        sampling=lhs_sampling,
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
    )

    # # 计算这一代可以运行多少个后代
    # remaining_nFE = max_nFE - nFE
    # current_n_offsprings = min(n_offsprings, remaining_nFE)

    # 进行优化
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 1),
        verbose=False
    )
    print("res.X: ", res.X)

    # 更新函数评估次数
    nFE += 1
    print("nFE: ", nFE)

    # 保存一个最优个体，如果有多个，随机选择一个
    X_best = res.X[np.random.choice(np.flatnonzero(res.F[:, 0] == res.F[:, 0].min()))]
    y_best = np.array(expensive_evaluation(X_best)).reshape(-1, 1)
    # X_best = res.X[np.argmin(res.F, axis=0)]
    # y_best = np.array([expensive_evaluation(x) for x in X_best]).reshape(-1, 1)

    # 更新数据集
    X = np.vstack((X, X_best))
    y = np.vstack((y, y_best))

# 输出最优解
print("Best Individual: ", X_best)
print("Fitness: ", y_best)