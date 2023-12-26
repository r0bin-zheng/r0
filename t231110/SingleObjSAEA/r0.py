from pymoo.operators.sampling.lhs import LHS
from pymoo.core.population import Population
from pymoo.optimize import minimize
import problem as p
import surrogate as s
import numpy as np
import sys
import os
from scipy.stats import norm

sys.path.append("..")
from utils import visualization, common

# 算法框架
# 使用conda环境moo1
# 代理集合
# 交替阶段

# 定义EI
def EI(mu, sigma, y_best):
    z = (y_best - mu) / sigma
    return (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

# 生成实验目录
exp_folder_name = common.create_exp_folder("test_assemble_surr")
exp_path = os.path.dirname(os.path.abspath(__file__)) + "/" + exp_folder_name
exp_data = {}

# 定义问题
problem_name = "ackley" # 问题名称，可选问题见problem.py
problem_setting1 = {
    "n_var": 2, # 变量维度
    "n_obj": 1, # 目标维度
    "n_constr": 0, # 约束维度
    "xl": -15, # 下界
    "xu": 30, # 上界
}
problem_setting2 = {
    "n_var": 2, # 变量维度
    "n_obj": 3, # 目标维度
    "n_constr": 0, # 约束维度
    "xl": -15, # 下界
    "xu": 30, # 上界
}
# problem_setting1 = {
#     "name": "griewank", # 问题名称，可选问题见problem.py
#     "n_var": 2, # 变量维度
#     "n_obj": 1, # 目标维度
#     "n_constr": 0, # 约束维度
#     "xl": -600, # 下界
#     "xu": 600, # 上界
# }
# problem_setting2 = {
#     "name": "griewank", # 问题名称，可选问题见problem.py
#     "n_var": 2, # 变量维度
#     "n_obj": 3, # 目标维度
#     "n_constr": 0, # 约束维度
#     "xl": -600, # 下界
#     "xu": 600, # 上界
# }
real_problem = p.get_problem(name=problem_name, setting=problem_setting1, surrogate=None)

# 定义代理模型类型
surrogate_model_types = ["kpls", "rbf"] # 代理模型名称，可选代理模型见surrogate.py
surrogates = []
for surrogate_model_type in surrogate_model_types:
    surrogates.append(s.get_surrogate_model(surrogate_model_type))
local_surrogate_model_type = "kriging"
local_surrogate_model = s.get_surrogate_model(local_surrogate_model_type)

# 定义求解器
ea_type = ""

# 算法执行过程
# 1 初始化
# 1.1 初始化参数
alg_args = {
    "n_init_sample": 100, # 初始样本点数量
    "max_nFE": 200, # 函数评估次数上限
    "n_offsprings": 25, # 每一代的后代数
    "save_path": "./pic", # 保存位置
    "global_phase_rate": 0.5, # 全局阶段比例
}

# 1.2 生成初始样本集合
sampling = LHS()
init_data = sampling(real_problem, alg_args["n_init_sample"]).get("X")
X = np.array(init_data)
y = np.array([real_problem.evaluate(ind) for ind in X]).reshape(-1, 1)
exp_data["X_0"] = X
exp_data["y_0"] = y

# 2 主循环
# 参数设置
max_epoch = alg_args["max_nFE"] - alg_args["n_init_sample"] # 最大迭代次数
global_phase_epoch = int(max_epoch * alg_args["global_phase_rate"]) # 全局阶段代数
local_phase_epoch = max_epoch - global_phase_epoch # 局部阶段代数
save_flag = 10 # 每隔save_flag代保存一次
nFE = alg_args["n_init_sample"] # 当前函数评估次数
x_of_best_sln_per_epoch = np.array([])
y_of_best_sln_per_epoch = np.array([])

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2

print("max_epoch: ", max_epoch)
print("global_phase_epoch: ", global_phase_epoch)
print("local_phase_epoch: ", local_phase_epoch)
while nFE < alg_args["max_nFE"]:
    epoch = nFE-alg_args["n_init_sample"]+1
    print("epoch: ", epoch)

    if epoch <= global_phase_epoch:
        # 训练多代理模型
        surrogate_models = []
        for surrogate in surrogates:
            surrogate_model = surrogate(print_training=False)
            surrogate_model.set_training_values(X, y)
            surrogate_model.train()
            surrogate_models.append(surrogate_model)
        problem_predicted_by_surrogate = p.get_problem(problem_name, setting=problem_setting2, surrogates=surrogate_models)

        # 2.2 优化
        initial_population = Population.new("X", X)
        algorithm = NSGA2(
            pop_size=2 * initial_population.size,
            eliminate_duplicates=True)
        res = minimize(
            problem_predicted_by_surrogate,
            algorithm,
            ('n_gen', 2),
            seed=1,
            verbose=False,
            pop=initial_population
        )
        pop = res.algorithm.pop
        pop_rank_0 = pop[pop.get("rank") == 0]
        # print("pop_rank_0: ", pop_rank_0)
        # 从pop_rank_0中选择出最小且不重复的个体
        x_opt = pop_rank_0[0].X
        for i in range(1, len(pop_rank_0)):
            if pop_rank_0[i].X not in X:
                x_opt = pop_rank_0[i].X

        # # 根据EI选择出最优解，保存到x_opt中
        # if len(res.X) == 1:
        #     x_opt = res.X[0]
        # else:
        #     x_opt = res.X[np.argmax(EI(res.F[:, 0], res.F[:, 1], np.min(y)))]

    else:
        # 2.1 创建代理模型并训练
        local_surr = local_surrogate_model(print_training=False)
        local_surr.set_training_values(X, y)
        local_surr.train()
        problem_predicted_by_surrogate = p.get_problem(problem_name, setting=problem_setting1, surrogate=local_surr)

        # 2.2 优化
        # 生成子代
        initial_population = Population.new("X", X)
        algorithm = GA(
            pop_size=2 * initial_population.size,
            eliminate_duplicates=True)
        res = minimize(
            problem_predicted_by_surrogate,
            algorithm,
            ('n_gen', 2), # 一次迭代的代数
            seed=1,
            verbose=False,
            pop=initial_population
        ) 
        off = res.algorithm.off

        # 从子代中选择出最小且不重复的个体
        x_opt = off[0].X
        min = off[0].F
        for i in range(1, len(off)):
            if off[i].F < min and off[i].X not in X:
                min = off[i].F
                x_opt = off[i].X

    # 2.3 更新样本点集合
    # 更新函数评估次数
    y_opt = real_problem.evaluate(x_opt)
    print("New solution to fill: \nX = %s\nF = %s" % (x_opt, y_opt))
    nFE += 1
    X = np.vstack((X, x_opt))
    y = np.vstack((y, y_opt))

    # 检查X中是否存在复数
    # n4 = 0
    # for i in range(0, len(X)):
    #     if np.isnan(X[i][0]) or np.isnan(X[i][1]):
    #         n4 += 1
    # print("n4: ", n4)
    
    # 从X y中选出最优解，并保存到x_of_best_sln_per_epoch y_of_best_sln_per_epoch
    best_ind = np.argmin(y)
    x_of_best_sln_per_epoch = np.append(x_of_best_sln_per_epoch, np.array(X[best_ind]))
    y_of_best_sln_per_epoch = np.append(y_of_best_sln_per_epoch, y[best_ind])

    if epoch % save_flag == 0:
        n = epoch // save_flag
        exp_data["X_"+str(n)] = X
        exp_data["y_"+str(n)] = y

# 从X y中选出最优解，并打印
best_ind = np.argmin(y)
print("Best solution found: \nX = %s\nF = %s" % (X[best_ind], y[best_ind]))

# 保存数据
n += 1
exp_data["problem_name"] = problem_name
exp_data["problem_setting"] = problem_setting1
exp_data["surrogate_model_type"] = surrogate_model_type
exp_data["alg_args"] = alg_args
exp_data["x_of_best_sln_per_epoch"] = x_of_best_sln_per_epoch
exp_data["y_of_best_sln_per_epoch"] = y_of_best_sln_per_epoch
exp_data["X"] = X
exp_data["y"] = y
exp_data["X_"+str(n)] = X
exp_data["y_"+str(n)] = y
np.savez(exp_folder_name+"/exp_data", exp_data)

# 画图
visualization.plot_best_sln_curve(y_of_best_sln_per_epoch, nFE-alg_args["n_init_sample"], exp_path)
visualization.plot_data_2d(X, problem_setting1["xl"], problem_setting1["xu"], exp_path)
visualization.plot_data_2d_1f(X, y, problem_setting1["xl"], problem_setting1["xu"], exp_path)
visualization.plot_gif(exp_data, n, problem_setting1["xl"], problem_setting1["xu"], exp_path)
