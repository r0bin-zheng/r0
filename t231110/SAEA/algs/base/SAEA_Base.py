"""SAEA算法基类"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import time
import numpy as np
from EA.algs.base.alg import Algorithm_Impl
from smt.sampling_methods import LHS
from SAEA.algs.base.SAEA_Unit import SAEA_Unit
from SAEA.utils.surrogate_model import SurrogateFactory

class SAEA_Base:
    def __init__(self, dim, size, surr_type, fit_max, range_min_list, range_max_list, surr_setting=None):
        self.name = 'SAEA'
        self.dim = dim
        self.size = size
        self.is_cal_max = fit_max
        self.range_min_list = range_min_list
        self.range_max_list = range_max_list

        self.cal_fit_num = 0
        self.unit_list = []
        self.surr_factory = SurrogateFactory(surr_type, surr_setting)
        self.surr = None
        self.fitfunction = None

    def run(self):
        self.start_time = time.time()
        self.init()
        self.iteration()
        self.end_time = time.time()
        self.print_result()
        self.save()


    def init(self):
        self.unit_new = []
        self.unit_new_history = []
        self.cal_fit_num = 0
        self.init_sampling()

    def iteration(self):
        iter = 0
        while self.cal_fit_num < self.is_cal_max:
            self.update(iter)
            self.print_iter()
            self.save_iter()
            iter += 1

    def init_sampling(self):
        xlimts = []
        for d in range(self.dim):
            lb = self.range_min_list[d]
            ub = self.range_max_list[d]
            xlimts.append([lb, ub])
        xlimts = np.array(xlimts)

        sampling = LHS(xlimits=xlimts)
        init_pop = sampling(self.size)

        for i in range(self.size):
            unit = SAEA_Unit()
            unit.position = init_pop[i]
            unit.fitness = self.cal_fitfunction(unit.position)
            self.unit_list.append(unit)

    def update(self, iter):
        self.create_surrogate()
        self.optimize_surrogate()

    def create_surrogate(self):
        X = []
        y = []
        for unit in self.unit_list:
            X.append(unit.position)
            y.append(unit.fitness)
        self.surr = self.surr_factory.get_sm(np.array(X), np.array(y))

    def optimize_surrogate(self):
        pass

    def cal_fitfunction(self, position):
        if self.fitfunction is None:
            value = 0
        else:
            value = self.fitfunction(position) if self.is_cal_max else -self.fitfunction(position)
        self.cal_fit_num += 1
        return value
    
    def cal_surr_fitfunction(self, position):
        if self.surr is None:
            value = 0
        else:
            value = self.surr(position)
        return value
    
    def print_iter(self):
        pass

    def save_iter(self):
        pass