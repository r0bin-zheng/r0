import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.core.problem import Problem
from smt.surrogate_models import RBF, KPLS

# 定义Rosenbrock函数（50维）
def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def expensive_evaluation(x):
    if isinstance(x, (int, float)):
        x = np.array([x])  # 将标量转换为数组
    return rosenbrock(x)

# 定义问题
class MyProblem(Problem):
    def __init__(self, n_var=50, n_obj=1, n_constr=0, xl=-10, xu=10, surrogate1=None, surrogate2=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.surrogate1 = surrogate1
        self.surrogate2 = surrogate2

    def _evaluate(self, X, out, *args, **kwargs):
        if self.surrogate1 and self.surrogate2:
            y_pred1 = self.surrogate1.predict_values(X)
            y_pred2 = self.surrogate2.predict_values(X)
            uncertainty = np.abs(y_pred1 - y_pred2)  # 绝对差异作为不确定性
            out["F"] = np.column_stack([y_pred1, y_pred2, uncertainty])
        else:
            out["F"] = np.array([expensive_evaluation(x) for x in X])

# 初始化
nFE = 0  # 初始化函数评估次数
max_nFE = 500  # 最大函数评估次数
pop_size = 100  # 种群大小
n_offsprings = 25  # 每一代的后代数

# 初始种群的采样
lhs_sampling = get_sampling("real_lhs")
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
    problem = MyProblem(n_var=50, surrogate1=surrogate1, surrogate2=surrogate2)

    # 使用代理模型进行进化算法
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        sampling=lhs_sampling,
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
    )

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
    # min_fitness_index = np.argmin(res.F[:, 0])
    # X_best = res.X[min_fitness_index]
    # 将res.X固定为二维数组
    resx = np.atleast_2d(res.X)
    resf = np.atleast_2d(res.F)
    print("resx: ", resx)
    min_fitness_index = np.argmin(resf[:, 0])
    X_best = resx[min_fitness_index]
    y_best = np.array(expensive_evaluation(X_best)).reshape(-1, 1)

    # 更新数据集
    X = np.vstack((X, X_best))
    y = np.vstack((y, y_best))

# 输出最优解
print("Best Individual: ", X_best)
print("Fitness: ", y_best)
