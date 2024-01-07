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
        self.WT = None

        self.pop_fitness_max = None
        self.pop_fitness_min = None
        self.pop_uncertainty_max = None
        self.pop_uncertainty_min = None

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
        self.statistics_unit_list()
        self.denominator = None
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
    
    def get_spark_num2(self, id):
        """
        计算火星数量，考虑不确定性。
        <公式>
        这是烟花算法中，求正常火星数量的公式，适应度越高，数量越多
        在这个公式中，a和b是预先设定的参数，用于控制火星的数量。这两个参数的设定通常基于启发式或问题的特定需求。a和b的值都小于 1（a<b<1），这确保了火星数量的调整保持在合理的范围内。
        a 通常是一个较小的数，用于确保至少有一定数量的火星。
        b 则是一个较大的数，用于限制火星数量的上限，以避免生成过多的火星
        """
        T = self.iter_num_main
        f_max = max(unit.fitness for unit in self.unit_list)
        f_i = self.unit_list[id].fitness
        U_i = self.get_unit_uncertainty(self.unit_list[id].position)

        numerator = (f_max - f_i) * self.W(T) * U_i
        denominator = sum((f_max - unit.fitness) * self.W(T) * self.get_unit_uncertainty(unit.position) for unit in self.unit_list)

        num = round(self.m * (numerator / (denominator + np.finfo(float).eps)))
        num = max(round(self.a * self.m), min(num, round(self.b * self.m)))
        return num
    
    def get_spark_num(self, id):
        WT = self.W(self.iter_num_main)
        f_i = self.unit_list[id].fitness
        U_i = self.get_unit_uncertainty(self.unit_list[id])
        
        # 最小-最大标准化适应度和不确定性
        f_range = self.pop_fitness_max - self.pop_fitness_min
        U_range = self.pop_uncertainty_max - self.pop_uncertainty_min

        normalized_f_i = (f_i - self.pop_fitness_min) / f_range if f_range > 0 else 0
        normalized_U_i = (U_i - self.pop_uncertainty_min) / U_range if U_range > 0 else 0
        
        # 计算分子和分母
        numerator = normalized_f_i + WT * normalized_U_i
        if self.denominator == None:
            self.denominator = sum((unit.fitness - self.pop_fitness_min) / f_range if f_range > 0 else 0 + WT * (self.get_unit_uncertainty(unit) - self.pop_uncertainty_min) / U_range if U_range > 0 else 0 for unit in self.unit_list)

        # 计算并返回火星数量
        num = round(self.m * (numerator / (self.denominator + np.finfo(float).eps)))
        num = max(round(self.a * self.m), min(num, round(self.b * self.m)))
        return num

    def select_sparks2(self):
        """选择火星，考虑不确定性"""
        T = self.iter_num_main
        self.all_list.extend(self.unit_list)
        f_max = max(unit.fitness for unit in self.all_list)

        R = np.array([unit.fitness for unit in self.all_list])
        U = np.array([self.get_unit_uncertainty(unit) for unit in self.all_list])

        # rates = (R + self.W(T) * U) / np.sum(R + self.W(T) * U)

        # for i in range(self.size):
        #     chosen_id = np.random.choice(range(len(self.all_list)), p=rates)
        #     self.unit_list[i].position = self.all_list[chosen_id].position
        #     self.unit_list[i].fitness = self.all_list[chosen_id].fitness

        # self.all_list = []

        # 对R中的值进行调整，以保证都为正数
        min_R = np.min(R)
        adjusted_R = R - min_R

        rates = (adjusted_R * self.W(T) * U) / np.sum(adjusted_R * self.W(T) * U)

        for i in range(self.size):
            chosen_id = np.random.choice(range(len(self.all_list)), p=rates)
            self.unit_list[i].position = self.all_list[chosen_id].position
            self.unit_list[i].fitness = self.all_list[chosen_id].fitness

        self.all_list = []

    def select_sparks(self):
        WT = self.W(self.iter_num_main)
        R_max = max(unit.fitness for unit in self.all_list)
        R_min = min(unit.fitness for unit in self.all_list)
        R_range = R_max - R_min
        U_max = max(self.get_unit_uncertainty(unit) for unit in self.all_list)
        U_min = min(self.get_unit_uncertainty(unit) for unit in self.all_list)
        U_range = U_max - U_min

        # 计算每个个体的选择概率
        probabilities = []
        for unit in self.all_list:
            R_norm = ((unit.fitness - R_min) / R_range) if R_range > 0 else 0
            U_norm = ((self.get_unit_uncertainty(unit) - U_min) / U_range) if U_range > 0 else 0
            numerator = R_norm + WT * U_norm
            probabilities.append(numerator)

        denominator = sum(probabilities) + np.finfo(float).eps  # 为了防止除以零
        probabilities = [p / denominator for p in probabilities]

        # 根据计算出的概率选择火星
        for i in range(self.size):
            chosen_id = np.random.choice(range(len(self.all_list)), p=probabilities)
            self.unit_list[i].position = self.all_list[chosen_id].position
            self.unit_list[i].fitness = self.all_list[chosen_id].fitness

        self.all_list = []

    def get_uncertainty(self, X):
        """获取给定解的不确定性"""
        _, std = self.surr.predict_values(X.reshape(1, -1)), self.surr.predict_variances(X.reshape(1, -1))
        return std[0, 0]
    
    def get_unit_uncertainty(self, unit):
        """获取给定单元的不确定性"""
        if unit.uncertainty is not None:
            return unit.uncertainty
        unit.uncertainty = self.get_uncertainty(unit.position)
        return unit.uncertainty

    def W(self, T):
        """计算权重因子"""
        if self.WT is None:
            if self.w_strategy == 1:
                """计算权重因子，随迭代次数指数衰减"""
                self.WT = 1 / (1 + self.alpha * T)
            elif self.w_strategy == 2:
                """计算权重因子，随迭代次数线性衰减"""
                self.WT = max(0, self.beta * (1 - T / self.iter_max_main))
            print("WT: ", self.WT)
        return self.WT
        
    def statistics_unit_list(self):
        """统计unit_list的信息"""
        # 取最小值
        fitness_max = np.finfo(float).min
        fitness_min = np.finfo(float).max
        uncertainty_max = np.finfo(float).min
        uncertainty_min = np.finfo(float).max
        for unit in self.unit_list:
            if unit.fitness > fitness_max:
                fitness_max = unit.fitness
            if unit.fitness < fitness_min:
                fitness_min = unit.fitness
            if self.get_unit_uncertainty(unit) > uncertainty_max:
                uncertainty_max = self.get_unit_uncertainty(unit)
            if self.get_unit_uncertainty(unit) < uncertainty_min:
                uncertainty_min = self.get_unit_uncertainty(unit)
        self.pop_fitness_max = fitness_max
        self.pop_fitness_min = fitness_min
        self.pop_uncertainty_max = uncertainty_max
        self.pop_uncertainty_min = uncertainty_min
