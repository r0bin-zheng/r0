"""
代理模型辅助的差分进化算法基类
"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from EA.algs.differential_evolution.DE_Base import DE_Base

class DE_Surr_Base(DE_Base):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list)
        self.name = 'DE_Surr_Base'
        self.surr = None
        self.use_variances = False

    def init(self):
        """初始化种群，由于SAEA框架中的一部分，初始化种群工作将在主算法完成"""
        self.position_best = np.zeros(self.dim)
        self.value_best_history = []
        self.position_best_history = []
        self.value_best = -np.finfo(np.float64).max

    def choose(self):
        """
        选择操作，根据代理模型预测结果选择
        由于使用代理模型，使用批量预测代替单个预测来提升效率
        """
        pos_new_list = []
        idx_list = []
        for i in range(self.size):
            pos = self.unit_list[i].position_new
            if pos is not None:
                pos_new_list.append(pos)
                idx_list.append(i)
        fitness_list = self.surr.predict(np.array(pos_new_list))
        self.cal_fit_num += len(pos_new_list)
        for i in range(len(idx_list)):
            new_value = fitness_list[i]
            if new_value > self.unit_list[idx_list[i]].fitness:
                self.unit_list[idx_list[i]].fitness = new_value
                self.unit_list[idx_list[i]].position = self.unit_list[idx_list[i]].position_new.copy()
