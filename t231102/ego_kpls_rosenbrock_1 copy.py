import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from smt.surrogate_models import KPLS
from scipy.stats import norm

# Rosenbrock函数定义
def rosenbrock_function(X):
    d = X.shape[1]
    sum_term = np.sum(100.0 * (X[:, 1:] - X[:, :-1]**2)**2 + (1 - X[:, :-1])**2, axis=1)
    return sum_term

# 期望提升函数
def expected_improvement(X, X_sample, Y_sample, kpls, xi=0.01):
    mu, sigma = kpls.predict_values(X), np.sqrt(kpls.predict_variances(X))
    mu_sample_opt = np.min(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

# 提出新位置的函数
def propose_location(acquisition, X_sample, Y_sample, kpls, bounds, n_restarts=25):
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, kpls).reshape(-1)

    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           

    return min_x.reshape(-1, 1)

# KPLS模型
kpls = KPLS()

# 随机选择初始采样点
dim = 30  # 30维
X_sample = np.random.uniform(-5, 5, size=(5, dim))
Y_sample = rosenbrock_function(X_sample)

# 更新KPLS模型
kpls.set_training_values(X_sample, Y_sample)
kpls.train()

bounds = np.tile([[-5, 5]], (dim, 1))  # 30维的边界

# 使用EI进行迭代优化
n_iter = 50
for i in range(n_iter):
    X_next = propose_location(expected_improvement, X_sample, Y_sample, kpls, bounds)
    Y_next = rosenbrock_function(X_next.reshape(1, -1))

    X_next = X_next.reshape(1, -1)  # Reshape X_next to have the correct shape
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.append(Y_sample, Y_next)

    kpls.set_training_values(X_sample, Y_sample)
    kpls.train()

# 最优的输出
optimal_index = np.argmin(Y_sample)
optimal_x = X_sample[optimal_index]
optimal_y = Y_sample[optimal_index]

print(f"Optimal X: {optimal_x}")
print(f"Optimal Y: {optimal_y}")
