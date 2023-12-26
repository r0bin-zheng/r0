import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# 目标函数（待优化）
def objective_function(x):
    return np.sin(x) * np.cos(x/2)

# 期望提升函数
def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    
    # 计算提升
    with np.errstate(divide='warn'):
        imp = mu - np.max(mu_sample) - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

# 高斯过程回归模型
kernel = Matern(nu=2.5)
gpr = GaussianProcessRegressor(kernel=kernel)

# 随机选择初始采样点
X_sample = np.random.choice(np.arange(-10, 10, 0.1), size=(5,1))
Y_sample = objective_function(X_sample)

# 更新高斯过程模型
gpr.fit(X_sample, Y_sample)

# 使用EI进行迭代优化
for i in range(20):
    # 在域上定义了一组随机点，用于计算EI并选择下一个点
    X = np.random.choice(np.arange(-10, 10, 0.1), size=(100,1))
    ei = expected_improvement(X, X_sample, Y_sample, gpr)
    X_next = X[np.argmax(ei)]  # 选择具有最大EI的点作为下一个点

    # 得到新点的目标函数值
    Y_next = objective_function(X_next)

    # 添加新点到样本中
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))

    # 更新高斯过程模型
    gpr.fit(X_sample, Y_sample)

# 最优的输出
optimal_index = np.argmin(Y_sample)
optimal_x = X_sample[optimal_index]
optimal_y = Y_sample[optimal_index]

print(f"Optimal X: {optimal_x}")
print(f"Optimal Y: {optimal_y}")
