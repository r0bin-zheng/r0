"""
FWA_Surr_Impl1的性能优化版本
"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
import math
from EA.algs.algorithm_fireworks.FWA_Unit import FWA_Unit
from SAEA.algs.fwa_surr.FWA_Surr_Impl1 import FWA_Surr_Impl1


class FWA_Surr_Impl2(FWA_Surr_Impl1):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list, alpha=0.6, beta=0.5):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list, alpha=alpha, beta=beta)
        self.name = 'FWA_Surr_Impl2'

    def update(self, iter):
        """新增计算适配度和不确定性的步骤"""
        self.iter_num = iter

        # 更新烟花信息
        self.update_firework(iter)
        self.statistics_unit_list()
        
        # 生成火星
        self.denominator = None
        for i in range(self.size):
            self.obtain_spark(i)
        
        # 生成特殊火星
        for num in range(self.spec_num):
            rand_id = np.random.randint(0, self.size)
            self.obtain_spec_spark(rand_id)

        # 选择火星
        self.get_fitness_uncertainty()
        self.select_sparks()
    
    def obtain_spark(self, id):
        """生成正常火星，不再计算适配度和不确定性"""
        num_sparks = self.get_spark_num(id)
        if self.unit_list[id].position.all() == self.CF.position.all():
            # 核心烟花振幅
            amplitude = self.CF_amplitude
        else:
            amplitude = self.get_spark_amplitude(id)
        for n in range(num_sparks):
            pos = self.unit_list[id].position.copy()
            for d in range(self.dim):
                if np.random.rand() < 0.5:
                    pos[d] += amplitude * np.random.uniform(-1, 1)
                    # pos[d] = self.get_out_bound_value_rand(pos[d], self.range_min_list[d], self.range_max_list[d])
            pos = self.get_out_bound_value_rand(pos, self.range_min_list, self.range_max_list)
            spark = FWA_Unit()
            spark.position = pos
            # spark.fitness = self.cal_fitfunction(pos)
            self.all_list.append(spark)

    def obtain_spec_spark(self, id):
        """生成特殊火星，不再计算适配度和不确定性"""
        pos = self.unit_list[id].position.copy()
        for d in range(self.dim):
            if np.random.rand() < 0.5:
                # 生成一个服从Levy分布的随机数
                pos[d] += self.levy_flight()
                pos[d] *= np.random.normal(1, 1)
        pos = self.get_out_bound_value_rand(pos, self.range_min_list, self.range_max_list)
        spec_spark = FWA_Unit()
        spec_spark.position = pos
        # spec_spark.fitness = self.cal_fitfunction(pos)
        self.all_list.append(spec_spark)

    def get_fitness_uncertainty(self):
        """计算适配度和不确定性"""
        self.all_list.extend(self.unit_list)
        pos_list = np.array([unit.position for unit in self.all_list])
        values, vars = self.surr.predict_value_variance(pos_list)
        value_min = values[0]
        value_max = values[0]
        var_min = vars[0]
        var_max = vars[0]
        for i in range(len(self.all_list)):
            self.all_list[i].fitness = values[i]
            self.all_list[i].uncertainty = vars[i]
            if values[i] < value_min:
                value_min = values[i]
            if values[i] > value_max:
                value_max = values[i]
            if vars[i] < var_min:
                var_min = vars[i]
            if vars[i] > var_max:
                var_max = vars[i]
        self.value_max = value_max
        self.value_min = value_min
        self.var_max = var_max
        self.var_min = var_min
        
    def select_sparks(self):
        """
        选择火星
        """
        WT = self.W(self.iter_num_main)
        R_max = self.value_max
        R_min = self.value_min
        U_max = self.var_max
        U_min = self.var_min
        R_range = R_max - R_min
        U_range = U_max - U_min

        # 计算每个个体的选择概率
        value_list = []
        best_idx = 0
        denominator = 0
        for i in range(len(self.all_list)):
            unit = self.all_list[i]
            R_norm = ((unit.fitness - R_min) / R_range) if R_range > 0 else 0
            U_norm = ((unit.uncertainty - U_min) / U_range) if U_range > 0 else 0
            numerator = R_norm + WT * U_norm
            if math.isnan(numerator):
                print("存在NaN，此时 R_norm: ", R_norm, " U_norm: ", U_norm)
                numerator = 0
            unit.value = numerator # 存储综合值
            value_list.append(numerator)
            denominator += numerator
            if numerator > self.all_list[best_idx].value:
                best_idx = i

        # 精英选择，并从all_list中删除
        unit_best = self.all_list[best_idx]
        self.unit_list[0].fitness = unit_best.fitness
        self.unit_list[0].position = unit_best.position
        self.unit_list[0].value = unit_best.value
        del self.all_list[best_idx]
        del value_list[best_idx]

        # 轮盘赌，确保概率和为1
        value_list_64 = np.asarray(value_list).astype('float64')
        sum1 = np.sum(value_list_64)
        probabilities = value_list_64 / sum1

        # 轮盘赌选择火星，同时避免重复
        try:
            chosen_id = np.random.choice(range(len(self.all_list)), p=probabilities, size=self.size - 1, replace=False)
        except ValueError:
            print("概率和不为1，此时 probabilities: ", probabilities)
            chosen_id = np.random.choice(range(len(self.all_list)), size=self.size - 1, replace=False)
        # 从1开始，0已经是精英了
        for i in range(self.size - 1):
            k = i + 1
            self.unit_list[k].fitness = self.all_list[chosen_id[i]].fitness
            self.unit_list[k].position = self.all_list[chosen_id[i]].position
            self.unit_list[k].value = self.all_list[chosen_id[i]].value
        self.all_list = []

        # 更新CF信息
        self.get_cf_amplitude(unit_best)
        self.CF = unit_best
