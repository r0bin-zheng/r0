"""遗传算法基类"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

import numpy as np
from algs.base.alg import Algorithm_Impl
from algs.algorithm_genetic.GA_Unit import GA_Unit


class GA_Base(Algorithm_Impl):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list)
        self.name = 'GA'
        self.cross_rate = 0.8
        """交叉率"""
        self.alter_rate = 0.05
        """变异率"""
        self.temp_unit_list = []
        """临时种群列表"""
        self.cross_strategy = 2
        """
        交叉策略
        - 1: 原始交叉
        - 2: 算术交叉
        """

    def init(self):
        super().init()
        self.unit_list = []
        self.temp_unit_list = []
        for _ in range(self.size):
            unit = GA_Unit()
            unit.position = np.random.uniform(self.range_min_list, self.range_max_list)
            unit.fitness = self.cal_fitfunction(unit.position)
            self.unit_list.append(unit)
            self.temp_unit_list.append(unit)

    def update(self, iter):
        super().update(iter)
        self.roulette()
        self.cross()
        self.altered()
        self.cal_value()

    def altered(self):
        """变异操作"""
        for i in range(self.size):
            if np.random.rand() < self.alter_rate:
                cur_dim = np.random.randint(0, self.dim)
                rand_pos = np.random.uniform(self.range_min_list[cur_dim], self.range_max_list[cur_dim])
                self.unit_list[i].position[cur_dim] = rand_pos

    def cross1(self):
        """交叉操作"""
        for i in range(0, self.size, 2):
            if i + 1 < self.size and np.random.rand() < self.cross_rate:
                cur_dim = np.random.randint(0, self.dim)
                temp = self.unit_list[i].position[cur_dim]
                self.unit_list[i].position[cur_dim] = self.unit_list[i + 1].position[cur_dim]
                self.unit_list[i + 1].position[cur_dim] = temp

    def cross(self):
        """交叉操作，包括原始交叉和算术交叉"""
        for i in range(0, self.size, 2):
            if i + 1 < self.size:
                if np.random.rand() < self.cross_rate:
                    if self.cross_strategy == 1:
                        """原始交叉"""
                        cur_dim = np.random.randint(0, self.dim)
                        temp = self.unit_list[i].position[cur_dim]
                        self.unit_list[i].position[cur_dim] = self.unit_list[i + 1].position[cur_dim]
                        self.unit_list[i + 1].position[cur_dim] = temp
                    elif self.cross_strategy == 2:
                        """算术交叉"""
                        alpha = np.random.rand()  # 加权系数
                        for cur_dim in range(self.dim):
                            value1 = self.unit_list[i].position[cur_dim]
                            value2 = self.unit_list[i + 1].position[cur_dim]
                            # 计算加权平均值
                            self.unit_list[i].position[cur_dim] = alpha * value1 + (1 - alpha) * value2
                            self.unit_list[i + 1].position[cur_dim] = alpha * value2 + (1 - alpha) * value1


    def roulette(self):
        """轮盘赌选择操作，同时保留最优解，并在选择过程中排除最优个体"""
        # 找出最优个体（假设适应度越大越好）
        best_unit_index = np.argmax([unit.fitness for unit in self.unit_list])
        # best_pos = self.unit_list[best_unit_index].position

        # 创建一个不包括最优个体的临时种群列表进行轮盘赌选择
        temp_units_for_selection = [unit for i, unit in enumerate(self.unit_list) if i != best_unit_index]

        # 获取轮盘赌选择概率列表，不包括最优个体
        roulette_rate = self.get_roulette_rate(temp_units_for_selection)
        roulette_sum = sum(roulette_rate)

        # 先将最优个体复制到新种群的第一个位置
        self.temp_unit_list[0].position = self.unit_list[best_unit_index].position.copy()
        self.temp_unit_list[0].fitness = self.unit_list[best_unit_index].fitness

        # 对剩余的位置使用轮盘赌选择填充
        for i in range(1, self.size):
            rand = np.random.uniform(0, roulette_sum)
            roulette_temp = 0
            for j, unit in enumerate(temp_units_for_selection):
                roulette_temp += roulette_rate[j]
                if rand < roulette_temp:
                    self.temp_unit_list[i] = unit
                    break

        # 更新种群
        for i in range(self.size):
            self.unit_list[i].position = self.temp_unit_list[i].position.copy()
            self.unit_list[i].fitness = self.temp_unit_list[i].fitness        

    def get_roulette_rate(self, units):
        """根据给定的个体列表计算轮盘赌选择概率"""
        values = np.array([unit.fitness for unit in units])
        max_value = np.max(values)
        min_value = np.min(values)
        mean_value = np.mean(values)

        if max_value == min_value:
            return np.ones(len(units)) / len(units)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        adjusted_probabilities = sigmoid((values - mean_value) / (max_value - min_value))
        normalized_probabilities = adjusted_probabilities / np.sum(adjusted_probabilities)
        return normalized_probabilities

    def cal_value(self):
        """计算适配度"""
        for i in range(self.size):
            self.unit_list[i].fitness = self.cal_fitfunction(self.unit_list[i].position)
