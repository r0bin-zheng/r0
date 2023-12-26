from pymoo.operators.sampling.lhs import LHS
from pymoo.core.population import Population
from pymoo.optimize import minimize
import problem as p
import surrogate as s
import numpy as np

# 算法框架
# 使用conda环境moo1
# 单代理

# 定义问题
problem_setting = {
    "name": "bohachevsky", # 问题名称，可选问题见problem.py
    "n_var": 2, # 变量维度
    "n_obj": 1, # 目标维度
    "n_constr": 0, # 约束维度
    "xl": -100, # 下界
    "xu": 100, # 上界
}
real_problem = p.get_problem(setting=problem_setting, surrogate=None)

# 定义代理模型类型
surrogate_model_type = "kpls" # 代理模型名称，可选代理模型见surrogate.py
surrogate = s.get_surrogate_model(surrogate_model_type)

# 定义求解器
ea_type = ""
# opt_alg = 

# 算法执行过程
# 1 初始化
# 1.1 初始化参数
alg_args = {
    "n_init_sample": 100, # 初始样本点数量
    "max_nFE": 200, # 函数评估次数上限
    "n_offsprings": 25, # 每一代的后代数
    "save_path": "./pic", # 保存位置
}

# 1.2 生成初始样本集合
sampling = LHS()
init_data = sampling(real_problem, alg_args["n_init_sample"]).get("X")
X = np.array(init_data)
# print(real_problem.evaluate(X))
y = np.array([real_problem.evaluate(ind) for ind in X]).reshape(-1, 1)

# 2 主循环
nFE = alg_args["n_init_sample"]
from pymoo.algorithms.soo.nonconvex.ga import GA
while nFE < alg_args["max_nFE"]:
    print("epoch: ", nFE-alg_args["n_init_sample"]+1)

    # 2.1 创建代理模型并训练
    surrogate_model = surrogate(print_training=False)
    surrogate_model.set_training_values(X, y)
    surrogate_model.train()
    # print(surrogate_model.predict_values(X))
    problem_predicted_by_surrogate = p.get_problem(setting=problem_setting, surrogate=surrogate_model)
    # print(problem_predicted_by_surrogate.evaluate(X))

    # 2.2 优化
    initial_population = Population.new("X", X)
    algorithm = GA(
        pop_size=2 * initial_population.size,
        eliminate_duplicates=True)
    res = minimize(
        problem_predicted_by_surrogate,
        algorithm,
        ('n_gen', 1),
        seed=1,
        verbose=False,
        pop=initial_population
    )
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    # 2.3 更新样本点集合
    # 更新函数评估次数
    x_opt = res.X
    y_opt = real_problem.evaluate(x_opt)
    nFE += 1
    X = np.vstack((X, x_opt))
    y = np.vstack((y, y_opt))

# 从X y中选出最优解，并打印
best_ind = np.argmin(y)
print("Best solution found: \nX = %s\nF = %s" % (X[best_ind], y[best_ind]))
