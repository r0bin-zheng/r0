# 测试基础的SAEA
# 2023-11-10
# pymoo 0.6.0

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from smt.surrogate_models import RBF, KPLS

# 定义问题
# 定义bohachevsky函数（2维）
def bohachevsky(x):
    return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1]) + 0.7

# 定义真实函数问题
# 将bohachevsky函数作为真实函数
class RealProblem(Problem):
    def __init__(self, n_var=2, n_obj=1, n_constr=0, xl=-100, xu=100):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([bohachevsky(x) for x in X])

# 定义代理模型问题
class MyProblem(Problem):
    def __init__(self, n_var=2, n_obj=1, n_constr=0, xl=-100, xu=100, surrogate=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.surrogate = surrogate

    def _evaluate(self, X, out, *args, **kwargs):
        if self.surrogate:
            out["F"] = self.surrogate.predict_values(X)
        else:
            out["F"] = np.array([bohachevsky(x) for x in X])

# 输出1



# 构建贝叶斯优化算法
# 参数设置
# 初始化样本点数量
n_init_sample = 100
# 函数评估次数上限
max_nFE = 200
# 每一代的后代数
n_offsprings = 25
# 保存位置
save_path = "./pic"

# 初始种群的采样
sampling = LHS()
init_data = sampling(MyProblem(), n_init_sample).get("X")
X = np.array(init_data)
y = np.array([bohachevsky(ind) for ind in X]).reshape(-1, 1)

# 更新函数评估次数
nFE = n_init_sample

# 循环直到函数评估次数达到上限
while nFE < max_nFE:
    print("epoch: ", nFE - n_init_sample + 1)

    # 创建代理模型并训练，细节不打印
    surrogate = RBF(print_training=False)
    surrogate.set_training_values(X, y)
    surrogate.train()

    # 定义问题
    problem = MyProblem(n_var=2, surrogate=surrogate)

    # 优化，初始数据使用X y
    initial_population = Population.new("X", X)
    # algorithm = PSO(eliminate_duplicates=True)
    algorithm = GA(
        pop_size=2 * initial_population.size,
        eliminate_duplicates=True)
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 1),
        seed=1,
        verbose=False,
        pop=initial_population
    )
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    # 更新函数评估次数
    x_opt = res.X
    y_opt = bohachevsky(x_opt)
    nFE += 1

    # 更新样本点
    X = np.vstack((X, x_opt))
    y = np.vstack((y, y_opt))

# 从数据X y中输出最优解
best_idx = np.argmin(y)
best_x = X[best_idx]
best_y = y[best_idx]
print("best_x: ", best_x)
print("best_y: ", best_y)

# 保存所有样本点
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.save(os.path.join(save_path, "X.npy"), X)
np.save(os.path.join(save_path, "y.npy"), y)

# 打印函数图像和所有样本点位置在同一张图上
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成网格数据
x1 = np.linspace(-100, 100, 100)
x2 = np.linspace(-100, 100, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = bohachevsky([X1, X2])

# 绘制所有样本点
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()

# 绘制最优解
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')
ax.scatter(best_x[0], best_x[1], best_y, c='b', marker='o')
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()

# 绘制最优解和所有样本点
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')
ax.scatter(best_x[0], best_x[1], best_y, c='b', marker='o')
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
