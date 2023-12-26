import numpy as np
from scipy.optimize import minimize
from smt.surrogate_models import KRG
from scipy.stats import norm

# Ackley函数定义
def ackley_function(x, a=20, b=0.2, c=2*np.pi):
    d = x.shape[1]
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x ** 2, axis=1) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x), axis=1) / d)
    return a + np.exp(1) + sum_sq_term + cos_term

# 期望提升函数
def expected_improvement(X, X_sample, Y_sample, kriging, xi=0.01):
    mu, sigma = kriging.predict_values(X), np.sqrt(kriging.predict_variances(X))
    mu_sample = kriging.predict_values(X_sample)

    sigma = sigma.reshape(-1, 1)
    
    # 计算提升
    with np.errstate(divide='warn'):
        imp = mu - np.min(mu_sample) - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei.reshape(-1)

# 提议新位置的函数
def propose_location(acquisition, X_sample, Y_sample, kriging, bounds, n_restarts=25):
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, kriging)

    # Find the best optimum by starting from n_restart different random points.
    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=starting_point, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x.reshape(1, -1)

    return min_x.reshape(-1, 1)

# Kriging模型
kriging = KRG(print_global=False)

# 随机选择初始采样点
bounds = np.array([[-5, 5], [-5, 5]])
X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(5,2))
Y_sample = ackley_function(X_sample)

# 更新Kriging模型
kriging.set_training_values(X_sample, Y_sample)
kriging.train()

# 使用EI进行迭代优化
for i in range(100):
    # 提议新位置
    X_next = propose_location(expected_improvement, X_sample, Y_sample, kriging, bounds)
    
    # 得到新点的目标函数值
    Y_next = ackley_function(X_next.reshape(1, -1))

    # 添加新点到样本中
    X_sample = np.vstack((X_sample, X_next.reshape(1, -2)))
    Y_sample = np.vstack((Y_sample, Y_next))

    # 更新Kriging模型
    kriging.set_training_values(X_sample, Y_sample)
    kriging.train()

# 最优的输出
optimal_index = np.argmin(Y_sample)
optimal_x = X_sample[optimal_index]
optimal_y = Y_sample[optimal_index]

print(f"Optimal X: {optimal_x}")
print(f"Optimal Y: {optimal_y}")
