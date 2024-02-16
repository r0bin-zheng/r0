"""代理模型辅助的粒子群算法基类"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from EA.algs.algorithm_particle_swarm.PSO_Base import PSO_Base


class PSO_Surr_Base(PSO_Base):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list)
        self.name = 'PSO_Surr_Base'
        self.surr = None
        self.use_variances = False

    def init(self):
        """初始化种群，由于SAEA框架中的一部分，初始化种群工作将在self.set_unit_list完成"""
        self.position_best = np.zeros(self.dim)
        self.value_best_history = []
        self.position_best_history = []
        self.value_best = -np.finfo(np.float64).max

    def set_unit_list(self, unit_list, surr):
        """设置种群列表，PSO还需要设置unit.velocity"""
        super().set_unit_list(unit_list, surr)
        for i in range(len(self.unit_list)):
            unit = self.unit_list[i]
            unit.position_best = unit.position
            unit.velocity = np.random.uniform(-self.velocity_max_list, self.velocity_max_list)
        self.curr_cycle_end = np.random.randint(3, 30)

    def update(self, iter):
        super().update(iter)
        X = np.array([unit.position for unit in self.unit_list])
        y = self.surr.predict(X)
        self.cal_fit_num += self.size
        for i in range(self.size):
            if y[i] > self.unit_list[i].fitness:
                self.unit_list[i].fitness = y[i]
                self.unit_list[i].position_best = self.unit_list[i].position

    def update_position(self, id):
        """
        更新位置，去除更新适配度的操作
        由于使用代理模型，使用外部的批量预测代替单个预测来提升效率
        """
        unit = self.unit_list[id]
        position_new = unit.position + unit.velocity
        unit.position = self.get_out_bound_value_rand(position_new, self.range_min_list, self.range_max_list)
        # fitness = self.cal_fitfunction(unit.position)
        # if fitness > unit.fitness:
        #     unit.fitness = fitness
        #     unit.position_best = unit.position
            
