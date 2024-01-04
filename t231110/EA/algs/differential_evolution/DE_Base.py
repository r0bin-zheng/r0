"""差分进化算的基类"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

import numpy as np
from algs.base.alg import Algorithm_Impl
from algs.differential_evolution.DE_Unit import DE_Unit

class DE_Base(Algorithm_Impl):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list)
        self.name = 'DE'
        self.cross_rate = 0.3
        self.alter_factor = 0.5
        self.unit_list = []
        self.unit_class = DE_Unit

    def init(self):
        super().init()
        for i in range(self.size - len(self.unit_list)):
            unit = DE_Unit()
            unit.position = np.random.uniform(self.range_min_list, self.range_max_list)
            unit.position_new = unit.position.copy()
            unit.fitness = self.cal_fitfunction(unit.position)
            self.unit_list.append(unit)
        # self.print_unit_list()

    def update(self, iter):
        super().update(iter)
        self.altered()
        self.cross()
        self.choose()

    def altered(self):
        for i in range(self.size):
            rand_id_list = np.random.permutation(self.size)
            r1, r2, r3 = rand_id_list[:3]
            if r1 == i:
                r1 = rand_id_list[3]
            if r2 == i:
                r2 = rand_id_list[3]
            if r3 == i:
                r3 = rand_id_list[3]

            new_pos = self.unit_list[r1].position + self.alter_factor * (self.unit_list[r2].position - self.unit_list[r3].position)
            new_pos = self.get_out_bound_value(new_pos)
            self.unit_list[i].position_new = new_pos

    def cross(self):
        for i in range(self.size):
            cur_dim = np.random.randint(self.dim)
            for d in range(self.dim):
                r = np.random.uniform(0, 1)
                if r >= self.cross_rate and d != cur_dim:
                    self.unit_list[i].position_new[d] = self.unit_list[i].position[d]

    def choose(self):
        for i in range(self.size):
            new_value = self.cal_fitfunction(self.unit_list[i].position_new)
            if new_value > self.unit_list[i].fitness:
                self.unit_list[i].fitness = new_value
                self.unit_list[i].position = self.unit_list[i].position_new.copy()

    def print_unit_list(self):
        print('unit_list:')
        for i in range(self.size):
            print(self.unit_list[i])
