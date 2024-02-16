"""
代理辅助的烟花算法实现1
- 添加了EFWA的振幅下限
- 添加了EFWA的位移方法，每个维度在振幅内随机（FWA_Base已实现）
- 添加了dynFWA的动态振幅
- 修改突变算子，不再生成在原点附近，服从Levy分布
- 修改选择策略为：精英选择、其他轮盘赌
- 修改映射策略为：镜像法
"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import math
import numpy as np
from SAEA.algs.fwa_surr.FWA_Surr_Base import FWA_Surr
from EA.algs.algorithm_fireworks.FWA_Unit import FWA_Unit
from SAEA.utils.common1 import avoid_duplicates_roulette


class FWA_Surr_Impl1(FWA_Surr):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list, 
                 alpha=0.6, beta=0.5, gamma=0.3, delta=0.5):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list, alpha, beta, gamma, delta)
        self.name = 'FWA_Surr_Impl1'
        """烟花振幅最小值"""
        self.amplitude_min = None
        self.amplitude_min_init = (range_max_list[0] - range_min_list[0]) * 0.02
        self.amplitude_min_final = (range_max_list[0] - range_min_list[0]) * 0.001
        """Core Firework 核心烟花，指当前的全局最优烟花"""
        self.CF = None
        self.CF_amplitude = None
        self.expansion_factor = 1.2
        """CF振幅扩张因子"""
        self.reduction_factor = 0.9
        """CF振幅下降因子"""
        

    def init(self):
        super().init()

        # 初始化最小振幅
        self.amplitude_min = self.amplitude_min_init

        # 初始化CF
        cf_idx = 0
        for i in range(self.size):
            if self.unit_list[i].fitness > self.unit_list[cf_idx].fitness:
                cf_idx = i
        tmp = self.unit_list[0]
        self.unit_list[0] = self.unit_list[cf_idx]
        self.unit_list[cf_idx] = tmp
        self.CF = self.unit_list[0]
        self.CF_amplitude = self.get_spark_amplitude(0)
    
    def update_firework(self, iter):
        """更新烟花信息， 新增了更新振幅下限"""
        super().update_firework(iter)
        self.update_amplitude_min(iter)
    
    def obtain_spark(self, id):
        """生成正常火星"""
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
            spark.fitness = self.cal_fitfunction(pos)
            self.all_list.append(spark)
        
    
    def get_spark_amplitude(self, id):
        """计算烟花振幅，动态振幅，有下限"""
        value_max = max(unit.fitness for unit in self.unit_list)
        value_sum = sum(unit.fitness for unit in self.unit_list)
        value = self.unit_list[id].fitness
        amplitude = self.amplitude_max * (value_max - value + np.finfo(float).eps) / (self.size * value_max - value_sum + np.finfo(float).eps)
        amplitude = max(amplitude, self.amplitude_min)
        return amplitude
    
    def select_sparks(self):
        """
        选择火星，考虑不确定性，标准化适配度和不确定性
        改进：
        - 精英选择，其他个体轮盘赌
        - 更新CF信息
        """
        self.all_list.extend(self.unit_list)
        WT = self.W(self.iter_num_main)
        R_max = self.all_list[0].fitness
        R_min = self.all_list[-1].fitness
        U_max = self.get_unit_uncertainty(self.all_list[0])
        U_min = self.get_unit_uncertainty(self.all_list[-1])
        for i in range(len(self.all_list)):
            if self.all_list[i].fitness > R_max:
                R_max = self.all_list[i].fitness
            if self.all_list[i].fitness < R_min:
                R_min = self.all_list[i].fitness
            U = self.get_unit_uncertainty(self.all_list[i])
            if U > U_max:
                U_max = U
            if U < U_min:
                U_min = U


        # R_values = [unit.fitness for unit in self.all_list]
        # U_values = [self.get_unit_uncertainty(unit) for unit in self.all_list]
        # R_max, R_min = max(R_values), min(R_values)
        # U_max, U_min = max(U_values), min(U_values)
        # R_range = R_max - R_min
        R_range = R_max - R_min
        U_range = U_max - U_min

        # 计算每个个体的选择概率
        value_list = []
        best_idx = 0
        denominator = 0
        for i in range(len(self.all_list)):
            unit = self.all_list[i]
            R_norm = ((unit.fitness - R_min) / R_range) if R_range > 0 else 0
            U_norm = ((self.get_unit_uncertainty(unit) - U_min) / U_range) if U_range > 0 else 0
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
        # probabilities = [v / denominator for v in value_list]
        # p = np.asarray(p).astype('float64')
        probabilities = value_list_64 / sum1

        # 轮盘赌选择火星，同时避免重复
        chosen_id = np.random.choice(range(len(self.all_list)), p=probabilities, size=self.size - 1, replace=False)
        # chosen_id = avoid_duplicates_roulette(range(len(self.all_list)), probabilities, self.size - 1)
        # 从1开始，0已经是精英了
        for i in range(self.size - 1):
            k = i + 1
            self.unit_list[k].fitness = self.all_list[chosen_id[i]].fitness
            self.unit_list[k].position = self.all_list[chosen_id[i]].position
            self.unit_list[k].value = self.all_list[chosen_id[i]].value

        # for i in range(self.size - 1):
        #     # print(i)
        #     chosen_id = avoid_duplicates_roulette(range(len(self.all_list)), probabilities)[0]
            # chosen_id = np.random.choice(range(len(self.all_list)), p=probabilities)
        #     k = i + 1
        #     self.unit_list[k].fitness = self.all_list[chosen_id].fitness
        #     self.unit_list[k].position = self.all_list[chosen_id].position
        #     self.unit_list[k].value = self.all_list[chosen_id].value
        #     del self.all_list[chosen_id]
        #     del probabilities[chosen_id]

        self.all_list = []

        # 更新CF信息
        self.get_cf_amplitude(unit_best)
        self.CF = unit_best

    def get_cf_amplitude(self, unit_best):
        """计算CF振幅"""
        if self.CF.fitness < unit_best.fitness:
            # unit_best有改进
            # 计算CF和unit_best的距离
            distance = np.sqrt(np.sum((self.CF.position - unit_best.position) ** 2))
            if distance > self.CF_amplitude:
                # 距离大于CF振幅，重新初始化
                self.CF_amplitude = self.get_spark_amplitude(0)
            else:
                # 距离小于CF振幅，扩张CF振幅
                self.CF_amplitude *= self.expansion_factor
        else:
            # unit_best无改进，缩小CF振幅
            self.CF_amplitude *= self.reduction_factor
        
    def obtain_spec_spark(self, id):
        """生成特殊火星"""
        pos = self.unit_list[id].position.copy()
        for d in range(self.dim):
            if np.random.rand() < 0.5:
                # 生成一个服从Levy分布的随机数
                pos[d] += self.levy_flight()
                pos[d] *= np.random.normal(1, 1)
        pos = self.get_out_bound_value_rand(pos, self.range_min_list, self.range_max_list)
        spec_spark = FWA_Unit()
        spec_spark.position = pos
        spec_spark.fitness = self.cal_fitfunction(pos)
        self.all_list.append(spec_spark)

    def get_out_bound_value_rand(self, position, min_list=None, max_list=None):
        """镜像法处理越界值"""
        min_list = self.range_min_list if min_list is None else min_list
        max_list = self.range_max_list if max_list is None else max_list
        position_tmp = position.copy()


        # 对于超过最大值的元素，减去最小值，然后对范围取模，最后最大值减去结果
        above_max = position > max_list
        position_tmp[above_max] = position[above_max] - min_list[above_max]
        position_tmp[above_max] = position_tmp[above_max] % (max_list[above_max] - min_list[above_max])
        position_tmp[above_max] = max_list[above_max] - position_tmp[above_max]

        # 对于超过最小值的元素，减去最小值，然后对范围取模，最后最小值加上结果
        below_min = position < min_list
        position_tmp[below_min] = position[below_min] - min_list[below_min]
        position_tmp[below_min] = position_tmp[below_min] % (max_list[below_min] - min_list[below_min])
        position_tmp[below_min] = min_list[below_min] + position_tmp[below_min]

        # # 确保反射后的值仍在范围内
        # position_tmp = np.clip(position_tmp, min_list, max_list)

        return position_tmp


    def levy_flight(self, beta=1.5, alpha_u=1, alpha_v=1):
        """Levy飞行"""
        u = np.random.normal(0, alpha_u, 1)
        v = np.random.normal(0, alpha_v, 1)
        step = u / math.pow(abs(v), (1 / beta))
        return step

    def update_amplitude_min(self, iter):
        """更新振幅下限"""
        A_i = self.amplitude_min_init
        A_f = self.amplitude_min_final
        t = iter
        t_max = self.iter_max
        self.amplitude_min = A_i - (A_i - A_f) * (math.sqrt(t * (2 * t_max - t))) / t_max

    def toStr(self):
        str = super().toStr()
        str += f"amplitude_min: {self.amplitude_min}\n"
        str += f"expansion_factor: {self.expansion_factor}\n"
        str += f"reduction_factor: {self.reduction_factor}\n"
        return str
