"""代理模型辅助的遗传算法基类"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from EA.algs.algorithm_genetic.GA_Base import GA_Base


class GA_Surr_Base(GA_Base):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list, 
                 is_cal_max=True, fitfunction=None, silent=False,
                 cross_rate=0.8, alter_rate=0.05, cross_strategy=2, surr=None):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list,
                         is_cal_max, fitfunction, silent,
                         cross_rate, alter_rate, cross_strategy)

        # 算法信息
        self.name = 'GA_Surr_Base'
        self.use_variances = False

        # 参数
        self.surr = surr

    # def init(self, unit_list=[]):
    #     """初始化unit_list和temp_unit_list"""
    #     super().init(unit_list=unit_list)

    #     self.position_best = np.zeros(self.dim)
    #     self.value_best_history = []
    #     self.position_best_history = []
    #     self.value_best = -np.finfo(np.float64).max

    # def set_unit_list(self, unit_list, surr):
    #     """设置种群列表，GA还需要设置temp_unit_list"""
    #     super().set_unit_list(unit_list, surr)
    #     self.temp_unit_list = [unit for unit in self.unit_list]

    def cal_value(self):
        """使用代理模型计算适配度"""
        X = []
        for i in range(self.size):
            X.append(self.unit_list[i].position)
        y = self.surr.predict(np.array(X))
        for i in range(self.size):
            self.unit_list[i].fitness = y[i]
            self.cal_fit_num += 1
            
