"""粒子群算法基类"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

import numpy as np
from algs.base.alg import Algorithm_Impl
from algs.algorithm_particle_swarm.PSO_Unit import PSO_Unit


class PSO_Base(Algorithm_Impl):
    """
    粒子群算法基类

    注意: PSO算法中Unit的fitness不是position的函数值，而是position_best的函数值
    """
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list)
        self.name = 'PSO'
        self.velocity_max_list = (np.array(range_max_list) - np.array(range_min_list)) * 0.1
        """速度限制"""
        self.C1 = 2
        """自我认知学习因子"""
        self.C2 = 2
        """社会认知学习因子"""
        self.W = 1
        """惯性权重"""
        self.w_strategy = 5
        """
        惯性权重策略
        - 1: 固定
        - 2: 线性递减
        - 3: 非线性递减
        - 4: 随机惯性
        - 5: 随机周期线性下降
        """
        self.curr_cycle_start = 1
        """当前周期的开始时间"""
        self.curr_cycle_end = None
        """当惯性权重策略为“随机周期线性下降时”，下一次更新周期的迭代数"""

    def init(self):
        super().init()
        self.unit_list = []
        for i in range(self.size):
            unit = PSO_Unit()
            unit.position = np.random.uniform(self.range_min_list, self.range_max_list)
            unit.position_best = unit.position
            unit.velocity = np.random.uniform(-self.velocity_max_list, self.velocity_max_list)
            unit.fitness = self.cal_fitfunction(unit.position)
            self.unit_list.append(unit)
        self.curr_cycle_end = np.random.randint(3, 30)

    def update(self, iter):
        super().update(iter)
        self.update_w(iter)
        for i in range(self.size):
            self.update_velocity(i)
            self.update_position(i)

    def update_velocity(self, id):
        unit = self.unit_list[id]
        velocity_new = self.W * unit.velocity + \
                       self.C1 * np.random.rand(self.dim) * (unit.position_best - unit.position) + \
                       self.C2 * np.random.rand(self.dim) * (self.position_best - unit.position)
        velocity_new = self.get_out_bound_value(velocity_new, -self.velocity_max_list, self.velocity_max_list)
        unit.velocity = velocity_new

    def update_position(self, id):
        unit = self.unit_list[id]
        position_new = unit.position + unit.velocity
        unit.position = self.get_out_bound_value_rand(position_new, self.range_min_list, self.range_max_list)
        fitness = self.cal_fitfunction(unit.position)
        if fitness > unit.fitness:
            unit.fitness = fitness
            unit.position_best = unit.position

    def update_w(self, iter):
        if self.w_strategy == 2:
            """线性递减"""
            self.W = (self.iter_max - iter) / self.iter_max

        elif self.w_strategy == 3:
            """非线性递减"""
            # 初始值是1，最终值接近0
            self.W = 1 - (iter / self.iter_max) ** 2
        elif self.w_strategy == 4:
            """随机惯性"""
            self.W = np.random.rand() * 2

        elif self.w_strategy == 5:
            """随机周期线性下降"""
            if iter >= self.curr_cycle_end:
                self.W = 1
                self.curr_cycle_end = np.random.randint(3, 30) + iter
                self.curr_cycle_start = iter
            else:
                self.W = (self.curr_cycle_end - iter) / (self.curr_cycle_end - self.curr_cycle_start)
        # print("惯性权重：", self.W)
        return self.W
