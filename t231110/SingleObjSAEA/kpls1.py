from pymoo.operators.sampling.lhs import LHS
from pymoo.core.population import Population
from pymoo.optimize import minimize
import problem as p
import surrogate as s
import numpy as np
import sys
import os
import copy
import time
from scipy.stats import norm

sys.path.append("..")
from utils import visualization, common
from surrogate import SurrogateFactory

# 算法框架
# 使用conda环境moo1
# 代理集合，使用灵活的阶段切换策略

# 定义EI
def EI(mu, sigma, y_best):
    z = (y_best - mu) / sigma
    return (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

# 生成实验目录
cur_dir = os.path.abspath(os.path.dirname(__file__))
exp_folder_name, exp_path = common.create_exp_folder("kpls1_ackley_var2_epoch200_ei", cur_dir+"/exps")
print("exp_path: ", exp_path)
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

# problem_name = "bohachevsky" # 问题名称，可选问题见problem.py
# problem_setting1 = {
#     "n_var": 30, # 变量维度
#     "n_obj": 1, # 目标维度
#     "n_constr": 0, # 约束维度
#     "xl": -100, # 下界
#     "xu": 100, # 上界
# }

# problem_name = "griewank" # 问题名称，可选问题见problem.py
# problem_setting1 = {
#     "n_var": 10, # 变量维度
#     "n_obj": 1, # 目标维度
#     "n_constr": 0, # 约束维度
#     "xl": -600, # 下界
#     "xu": 600, # 上界
# }

# problem_name = "himmelblau" # 问题名称，可选问题见problem.py
# problem_setting1 = {
#     "n_var": 2, # 变量维度
#     "n_obj": 1, # 目标维度
#     "n_constr": 0, # 约束维度
#     "xl": -6, # 下界
#     "xu": 6, # 上界
# }

problem_setting2 = copy.deepcopy(problem_setting1)
problem_setting2["n_obj"] = 3
real_problem = p.get_problem(name=problem_name, setting=problem_setting1, surrogate=None)

# 定义代理模型类型
surrogate_model_type = ["kpls"] # 代理模型名称，可选代理模型见surrogate.py
surr_setting = {
    "print_training": False,
    "print_prediction": False,
    "print_problem": False,
    "print_solver": False,
}
sf = SurrogateFactory(surrogate_model_type, surr_setting)

# 定义求解器
selection_type = "ei"

# 算法执行过程

start_time = time.time()

# ----------------------- 1 初始化 -----------------------

# 1.1 初始化参数
alg_args = {
    "n_init_sample": 100, # 初始样本点数量
    "max_nFE": 200, # 函数评估次数上限
    "n_offsprings": 25, # 每一代的后代数
    "save_path": "./pic", # 保存位置
    "top_k": 0.05, # 局部搜索时，选择前百分之多少个个体
}

# 1.2 生成初始样本集合
sampling = LHS()
init_data = sampling(real_problem, alg_args["n_init_sample"]).get("X")
X = np.array(init_data)
y = np.array([real_problem.evaluate(ind) for ind in X]).reshape(-1, 1)
exp_data["X_0"] = X
exp_data["y_0"] = y

# ----------------------- 2 主循环 -----------------------

# 循环参数设置
max_epoch = alg_args["max_nFE"] - alg_args["n_init_sample"] # 最大迭代次数
# global_flag = True # 是否进入全局搜索阶段
y_best = np.min(y) # 当前最优解
save_flag = 10 # 每隔save_flag代保存一次
nFE = alg_args["n_init_sample"] # 当前函数评估次数
x_of_best_sln_per_epoch = np.array([])
y_of_best_sln_per_epoch = np.array([])

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2

print("max_epoch: ", max_epoch)
while nFE < alg_args["max_nFE"]:
    epoch = nFE-alg_args["n_init_sample"]+1
    print("\n################## Epoch " + str(epoch) + " ##################\n")
    print("y len: ", len(y))

    # 2.1 创建代理模型并训练
    surr = sf.get_sm(X, y)
    problem_predicted_by_surrogate = p.get_problem(problem_name, setting=problem_setting1, surrogate=surr)

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

    if selection_type == "":
        off = res.algorithm.off

        # 从子代中选择出最小且不重复的个体
        x_opt = None
        min = np.inf
        for i in range(0, len(off)):
            if off[i].F < min and off[i].X not in X:
                min = off[i].F
                x_opt = off[i].X
    elif selection_type == "ei":
        pop = res.algorithm.pop
        pop_x = pop.get("X")
        pop_F = pop.get("F")
        pop_var = surr.predict_variances(pop_x)

        # print("var: ", pop_var)
        # 计算每个个体的EI
        pop_F = pop.get("F")
        # 计算改进量，即与y_best的差距
        pop_improvement = y_best - pop_F
        # 避免除以零的情况，获取标准差
        std_dev = np.sqrt(np.maximum(pop_var, 1e-9))
        # 计算EI
        pop_EI = pop_improvement * norm.cdf(pop_improvement / std_dev) + std_dev * norm.pdf(pop_improvement / std_dev)
        # print("pop_EI: ", pop_EI)

        # 选出EI值最大且在X中不重复的个体
        x_new = None
        # 默认为负无穷
        ei_max = -np.inf
        for i in range(0, len(pop)):
            if pop_EI[i] > ei_max and pop_x[i] not in X:
                x_new = pop_x[i]
                ei_max = pop_EI[i]
        x_opt = x_new

    # 2.3 更新样本点集合
    # 更新函数评估次数
    if x_opt is not None:
        y_opt = real_problem.evaluate(x_opt)
        print("New solution to fill: \nX = %s\nF = %s" % (x_opt, y_opt))
        nFE += 1
        X = np.vstack((X, x_opt))
        y = np.vstack((y, y_opt))

        # 2.4 切换阶段
        print("y_opt: ", y_opt)
        print("y_best: ", y_best)
        if y_opt < y_best:
            y_best = y_opt
    
        # # 检查X是否重复
        # n1 = 0
        # for i in range(0, len(X)):
        #     for j in range(i+1, len(X)):
        #         if np.array_equal(X[i], X[j]):
        #             n1 += 1
        # print("duplicated X: ", n1)

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
    
    else:
        print("No new solution to fill.")

# ----------------------- 3 结束 -----------------------

n += 1
exp_data["X_"+str(n)] = X
exp_data["y_"+str(n)] = y

# 从X y中选出最优解，并打印
best_ind = np.argmin(y)
print("Best solution found: \nX = %s\nF = %s" % (X[best_ind], y[best_ind]))

end_time = time.time()
print("Time cost: ", end_time-start_time)

# ------------------ 数据保存和可视化 -------------------

exp_data_path = exp_path + "/data"
exp_pic_path = exp_path + "/pic"
if not os.path.exists(exp_data_path):
    os.makedirs(exp_data_path)
if not os.path.exists(exp_pic_path):
    os.makedirs(exp_pic_path)

# 保存数据
exp_data["x_of_best_sln_per_epoch"] = x_of_best_sln_per_epoch
exp_data["y_of_best_sln_per_epoch"] = y_of_best_sln_per_epoch
exp_data["X"] = X
exp_data["y"] = y


exp_info = {
    "problem_name": problem_name,
    "problem_setting": problem_setting1,
    "problem_surr_setting": problem_setting2,
    "surrogate_model_type": surrogate_model_type,
    "surr_setting": surr_setting,
    "alg_args": alg_args
}
common.save_dict(exp_info, "exp_info", exp_data_path)

for k, v in exp_data.items():
    common.save_numpy_array(v, k, exp_data_path)

# 数据可视化
visualization.plot_best_sln_curve(y_of_best_sln_per_epoch, nFE-alg_args["n_init_sample"], exp_pic_path)
visualization.plot_data_2d(X, problem_setting1["xl"], problem_setting1["xu"], exp_pic_path)
visualization.plot_data_2d_1f(X, y, problem_setting1["xl"], problem_setting1["xu"], exp_pic_path)
visualization.plot_gif(exp_data, n, problem_setting1["xl"], problem_setting1["xu"], exp_pic_path)
visualization.plot_surrogate_1d_with_x2(real_problem, surr, problem_setting1["xl"], problem_setting1["xu"], save_path=exp_pic_path)
