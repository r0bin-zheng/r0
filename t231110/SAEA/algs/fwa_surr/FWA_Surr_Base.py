"""烟花算法，结合代理模型改进，基类"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from EA.algs.algorithm_fireworks.FWA_Base import FWA_Base

class FWA_Surr(FWA_Base):
    """
    烟花算法，结合代理模型改进，基类
    eg.
    alg = FWA_Surr(dim=2, size=50, iter_max=50, range_min_list=[-100, -100], range_max_list=[100, 100], is_cal_max=False, iter_max_main=50)
    alg.surr = surr
    alg.w_strategy = 2
    """
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list, alpha=0.6, beta=0.5):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list)
        self.name = 'FWA_Surr'
        self.surr = None
        # self.iter_max_main = iter_max_main # 主算法迭代次数

        self.iter_num = 0
        self.iter_num_main = None
        self.iter_max_main = None
        self.w_strategy = 2
        self.alpha = alpha
        self.beta = beta

    def update(self, iter):
        self.iter_num = iter

        for i in range(self.size):
            if self.unit_list[i].fitness > self.value_best:
                self.value_best = self.unit_list[i].fitness
                self.position_best = self.unit_list[i].position
            self.unit_list[i].save()

        if not self.silence:
            print(f"第 {iter} 代")
        if self.is_cal_max:
            self.value_best_history.append(self.value_best)
            if not self.silence:
                print(f"最优值= {self.value_best}")
        else:
            self.value_best_history.append(-self.value_best)
            if not self.silence:
                print(f"最优值= {-self.value_best}")

        self.position_best_history.append(self.position_best)
        if not self.silence:
            print(f"最优解= {self.position_best}")
        
        # 生成火星
        # print("生成火星")
        for i in range(self.size):
            self.obtain_spark(i)
        
        # print("生成特殊火星")
        # 生成特殊火星
        for num in range(self.spec_num):
            rand_id = np.random.randint(0, self.size)
            self.obtain_spec_spark(rand_id)

        # print("选择火星")
        # 选择火星
        self.select_sparks()
    
    def get_spark_num(self, id):
        """计算火星数量，考虑不确定性"""
        T = self.iter_num_main
        f_max = max(unit.fitness for unit in self.unit_list)
        f_i = self.unit_list[id].fitness
        U_i = self.get_uncertainty(self.unit_list[id].position)

        numerator = (f_max - f_i) + self.W(T) * U_i
        denominator = sum((f_max - unit.fitness) + self.W(T) * self.get_uncertainty(unit.position) for unit in self.unit_list)

        num = round(self.m * (numerator / (denominator + np.finfo(float).eps)))
        num = max(round(self.a * self.m), min(num, round(self.b * self.m)))
        return num

    def select_sparks(self):
        """选择火星，考虑不确定性"""
        T = self.iter_num_main
        self.all_list.extend(self.unit_list)
        f_max = max(unit.fitness for unit in self.all_list)

        R = np.array([unit.fitness for unit in self.all_list])
        U = np.array([self.get_uncertainty(unit.position) for unit in self.all_list])

        # rates = (R + self.W(T) * U) / np.sum(R + self.W(T) * U)

        # for i in range(self.size):
        #     chosen_id = np.random.choice(range(len(self.all_list)), p=rates)
        #     self.unit_list[i].position = self.all_list[chosen_id].position
        #     self.unit_list[i].fitness = self.all_list[chosen_id].fitness

        # self.all_list = []

        # 对R中的值进行调整，以保证都为正数
        min_R = np.min(R)
        adjusted_R = R - min_R

        rates = (adjusted_R + self.W(T) * U) / np.sum(adjusted_R + self.W(T) * U)

        for i in range(self.size):
            chosen_id = np.random.choice(range(len(self.all_list)), p=rates)
            self.unit_list[i].position = self.all_list[chosen_id].position
            self.unit_list[i].fitness = self.all_list[chosen_id].fitness

        self.all_list = []

    # def update_surrogate_model(self):
    #     """更新代理模型"""
    #     X = np.array([unit.position for unit in self.unit_list])
    #     Y = np.array([unit.fitness for unit in self.unit_list]).reshape(-1, 1)
    #     self.surr.set_training_values(X, Y)
    #     self.surr.train()

    def get_uncertainty(self, X):
        """获取给定解的不确定性"""
        _, std = self.surr.predict_values(X.reshape(1, -1)), self.surr.predict_variances(X.reshape(1, -1))
        return std[0, 0]

    def W(self, T):
        """计算权重因子"""
        if self.w_strategy == 1:
            """计算权重因子，随迭代次数指数衰减"""
            return 1 / (1 + self.alpha * T)
        elif self.w_strategy == 2:
            """计算权重因子，随迭代次数线性衰减"""
            return max(0, self.beta * (1 - T / self.iter_max_main))
