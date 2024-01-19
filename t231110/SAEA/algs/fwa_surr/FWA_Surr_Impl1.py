"""
代理辅助的烟花算法实现1
- 添加了EFWA的振幅下限
- 添加了EFWA的位移方法，每个维度在振幅内随机
- 添加了dynFWA的动态振幅
- 修改突变算子，不再生成在原点附近，服从Levy分布
- 修改选择策略为：精英选择、其他轮盘赌
"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from SAEA.algs.fwa_surr.FWA_Surr_Base import FWA_Surr

class FWA_Surr_Impl1(FWA_Surr):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list, alpha=0.6, beta=0.5):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list, alpha, beta)
        self.name = 'FWA_Surr_Impl1'
        self.amplitude_min = 0.1
        """Core Firework 核心烟花，指当前的全局最优烟花"""
        self.CF = None
        """Prev Core Firework 上一代的全局最优烟花，用于计算动态振幅"""
        self.CF_prev = None

    def init(self):
        super().init()

        # 初始化CF
        self.CF = self.unit_list[0]
        for unit in self.unit_list:
            if unit.fitness > self.CF.fitness:
                self.CF = unit
    
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
        pass

    
