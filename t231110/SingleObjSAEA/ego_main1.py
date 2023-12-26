from pymoo.operators.sampling.lhs import LHS
from pymoo.core.population import Population
from pymoo.optimize import minimize as pymoo_minimize
from scipy.optimize import minimize as scipy_minimize
import problem as p
import surrogate as s
import numpy as np
import sys
import os
import copy
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


# def expected_improvement(X, kriging_model, y_best):
#     mu, sigma = kriging_model.predict_values(X.reshape(1, -1)), kriging_model.predict_variances(X.reshape(1, -1))
#     sigma = np.sqrt(np.abs(sigma))
#     with np.errstate(divide='warn'):
#         Z = (y_best - mu) / sigma
#         ei = (y_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
#         return -ei

def expected_improvement(X1, kriging_model, y_best):
    mu, sigma = kriging_model.predict_values(X1.reshape(1, -1)), kriging_model.predict_variances(X1.reshape(1, -1))
    sigma = np.sqrt(np.abs(sigma))
    with np.errstate(divide='warn'):
        Z = (y_best - mu) / sigma
        ei = (y_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -ei

# 生成实验目录
cur_dir = os.path.abspath(os.path.dirname(__file__))
exp_folder_name, exp_path = common.create_exp_folder("ego_ackley_var2_epoch200", cur_dir+"/exps")
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
xlimits = np.array([[problem_setting1["xl"], problem_setting1["xu"]]] * problem_setting1["n_var"])
print("xlimits: ", xlimits)

real_problem = p.get_problem(name=problem_name, setting=problem_setting1, surrogate=None)

# 定义代理模型类型
surrogate_model_type = ["kriging"] # 代理模型名称，可选代理模型见surrogate.py
surr_setting = {
    "print_training": False,
    "print_prediction": False,
    "print_problem": False,
    "print_solver": False,

}
sf = SurrogateFactory(surrogate_model_type, surr_setting)

# 定义求解器
ea_type = ""

# 算法执行过程

# ----------------------- 1 初始化 -----------------------

# 1.1 初始化参数
alg_args = {
    "n_init_sample": 100, # 初始样本点数量
    "max_nFE": 200, # 函数评估次数上限
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
y_best = np.min(y) # 当前最优解
save_flag = 10 # 每隔save_flag代保存一次
nFE = alg_args["n_init_sample"] # 当前函数评估次数
x_of_best_sln_per_epoch = np.array([])
y_of_best_sln_per_epoch = np.array([])

print("max_epoch: ", max_epoch)
while nFE < alg_args["max_nFE"]:
    epoch = nFE-alg_args["n_init_sample"]+1
    print("\n################## Epoch " + str(epoch) + " ##################\n")
    print("y len: ", len(y))

    # 2.1 训练代理模型
    surr = sf.get_sm(X, y)
    y_best = np.min(y)
    
    # 2.2 优化
    # res = minimize(lambda X1: -expected_improvement(X, surr, y_best),
    #                x0=X[-1], bounds=xlimits, method='L-BFGS-B')
    res = minimize(lambda X1: -expected_improvement(X1.reshape(1, -1), surr, y_best),
               x0=X[-1], bounds=xlimits, method='L-BFGS-B')
    
    # 2.3 更新样本点集合
    # 更新函数评估次数
    new_x = res.x
    new_y = real_problem.evaluate(new_x)
    print("New solution to fill: \nX = %s\nF = %s" % (new_x, new_y))
    nFE += 1
    X = np.vstack((X, new_x))
    y = np.append(y, new_y)
        
    # 从X y中选出最优解，并保存到x_of_best_sln_per_epoch y_of_best_sln_per_epoch
    best_ind = np.argmin(y)
    x_of_best_sln_per_epoch = np.append(x_of_best_sln_per_epoch, np.array(X[best_ind]))
    y_of_best_sln_per_epoch = np.append(y_of_best_sln_per_epoch, y[best_ind])

    if epoch % save_flag == 0:
        n = epoch // save_flag
        exp_data["X_"+str(n)] = X
        exp_data["y_"+str(n)] = y

# ----------------------- 3 结束 -----------------------

n += 1
exp_data["X_"+str(n)] = X
exp_data["y_"+str(n)] = y

# 从X y中选出最优解，并打印
best_ind = np.argmin(y)
print("Best solution found: \nX = %s\nF = %s" % (X[best_ind], y[best_ind]))

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
