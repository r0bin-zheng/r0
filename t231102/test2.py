from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from smt.surrogate_models import RBF, QP, KPLS
import numpy as np

nFE = 0

# 定义一个昂贵的评估函数
def expensive_evaluation(x):
    # 输出当前评估次数
    global nFE
    nFE += 1
    print("nFE: ", nFE)
    return [sum(val**2 for val in x)]

# 定义问题
class MyProblem(Problem):
    def __init__(self, surrogate1=None, surrogate2=None):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=-10, xu=10)
        self.surrogate1 = surrogate1
        self.surrogate2 = surrogate2

    def _evaluate(self, X, out, *args, **kwargs):
        if self.surrogate1 and self.surrogate2:
            y_pred1 = self.surrogate1.predict_values(X)
            y_pred2 = self.surrogate2.predict_values(X)
            uncertainty = np.abs(y_pred1 - y_pred2)  # 绝对差异作为不确定性
            out["F"] = np.column_stack([(y_pred1 + y_pred2) / 2, uncertainty])  # 优化平均值和不确定性
        else:
            out["F"] = [expensive_evaluation(x) for x in X]


# 使用LHS生成初始种群
lhs_sampling = get_sampling("real_lhs")
X_data = lhs_sampling.do(MyProblem(), n_samples=100)
X = np.array([ind.X for ind in X_data])
y = np.array([expensive_evaluation(ind) for ind in X]).reshape(-1, 1)

# 创建并训练两个代理模型
surrogate1 = RBF()
surrogate1.set_training_values(X, y)
surrogate1.train()

surrogate2 = QP()
surrogate2.set_training_values(X, y)
surrogate2.train()

# 使用代理模型进行进化算法
problem = MyProblem(surrogate1=surrogate1, surrogate2=surrogate2)
algorithm = NSGA2(
    pop_size=100,
    n_offsprings=25,
    sampling=lhs_sampling,  # 使用LHS进行初始化
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True,
)

# 运行算法
res = minimize(
    problem,
    algorithm,
    ('n_gen', 100),
    verbose=True
)

# 定义真实的Pareto前沿
pareto_front = np.array([[x, 1-x] for x in np.linspace(0, 1, 100)])

# 计算IGD的值
ind = IGD(pareto_front)
print("IGD", ind.do(res.F))

# 计算HV
reference_point = np.array([1.1, 1.1])  # 这应该根据您的问题进行修改
hv = Hypervolume(ref_point=reference_point)
hv_value = hv.do(res.F)
print("HV: ", hv_value)
