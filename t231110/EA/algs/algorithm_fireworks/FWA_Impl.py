"""烟花算法实现"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

import numpy as np
from algs.algorithm_fireworks.FWA_Base import FWA_Base


class FWA_Impl(FWA_Base):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list)

        """选择策略：根据距离distance、根据适应度fitness、随机选择random"""
        self.select_strategy = 'fitness'

    def select_sparks(self):
        self.all_list.extend(self.unit_list)
        values = np.array([unit.fitness for unit in self.all_list])
        max_value_id = np.argmax(values)

        if self.select_strategy == 'distance':
            distances = np.abs(values[:, None] - values)
            dist_sum = np.sum(distances, axis=1)
            rates = dist_sum / np.sum(dist_sum)
        elif self.select_strategy == 'fitness':
            min_value = np.min(values)
            adjusted_values = values - min_value
            rates = adjusted_values / np.sum(adjusted_values)
        elif self.select_strategy == 'random':
            rates = np.ones(len(self.all_list)) / len(self.all_list)

        for i in range(self.size):
            if i == 0:
                self.unit_list[0].position = self.all_list[max_value_id].position
                self.unit_list[0].fitness = self.all_list[max_value_id].fitness
            else:
                chosen_id = np.random.choice(range(len(self.all_list)), p=rates)
                self.unit_list[i].position = self.all_list[chosen_id].position
                self.unit_list[i].fitness = self.all_list[chosen_id].fitness

        self.all_list = []