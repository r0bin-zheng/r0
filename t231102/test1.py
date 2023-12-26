import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

# 定义目标函数（昂贵的评估函数）
def expensive_evaluation(individual):
    return sum(ind**2 for ind in individual),  # 注意返回值是元组

# 定义适应度函数
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化问题
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册交叉、变异和选择操作
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def expected_improvement(X, X_sample, Y_sample, model, xi=0.01):
    mu, sigma = model.predict(X, return_std=True)
    mu_sample = model.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    sigma[sigma < 1e-10] = 1e-10  # 防止除以零

    # 计算改进量
    imp = mu - np.min(mu_sample) - xi
    Z = imp / sigma
    # ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    # return ei
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei.ravel()  # 将 ei 转换为一维数组

def main():
    # 初始种群
    pop = toolbox.population(n=50)
    # 初始代理模型训练数据
    X = np.array([toolbox.individual() for _ in range(10)])
    y = np.array([expensive_evaluation(ind)[0] for ind in X]).reshape(-1, 1)

    # 代理模型
    kernel = Matern(nu=2.5)
    # surrogate = GaussianProcessRegressor(kernel=kernel)
    surrogate = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)  # 增加一个小的噪声水平

    # 进化
    for gen in range(100):
        # 使用代理模型评估种群
        surrogate.fit(X, y)
        for ind in pop:
            ind.fitness.values = surrogate.predict([ind])[0],

        # 选择、交叉和变异
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)

        # 根据预期改进选择个体进行真实评估
        X_offspring = np.array([ind for ind in offspring])
        # ei = expected_improvement(X_offspring, X, y, surrogate)
        # chosen_indices = np.argsort(-ei)[:5]
        # chosen_ones = [offspring[i] for i in chosen_indices]
        ei = expected_improvement(X_offspring, X, y, surrogate)
        chosen_indices = np.argsort(-ei)[:5].tolist()  # 确保是一维数组
        chosen_ones = [offspring[i] for i in chosen_indices]

        for ind in chosen_ones:
            ind.fitness.values = expensive_evaluation(ind)

            # 更新训练数据
            X = np.vstack((X, np.array(ind)))
            y = np.vstack((y, np.array(ind.fitness.values).reshape(1, -1)))

        # 环境选择
        pop = toolbox.select(offspring, k=len(pop))

    # 输出最佳解
    best_ind = tools.selBest(pop, 1)[0]
    print("Best Individual: ", best_ind)
    print("Fitness: ", best_ind.fitness.values[0])

if __name__ == "__main__":
    main()
