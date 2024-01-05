import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from EA.algs.algorithm_fireworks.FWA_Base import FWA_Base
from EA.algs.algorithm_fireworks.FWA_Impl import FWA_Impl
from EA.algs.differential_evolution.DE_Base import DE_Base
from EA.algs.differential_evolution.DE_Impl import DE_Impl
from SAEA.algs.fwa_surr.FWA_Surr_Base import FWA_Surr


alg_dict = {
    "DE": DE_Base,
    "DE_Impl": DE_Impl,
    "FWA": FWA_Base,
    "FWA_Impl": FWA_Impl,
    "FWA_Surr": FWA_Surr,
}

def get_alg(name):
    return alg_dict[name]


class Ea_Factory:
    """
    算法工厂类
    """
    def __init__(self, alg_name, dim, size, iter_max, range_min_list, range_max_list, is_cal_max):
        self.alg_name = alg_name
        self.dim = dim
        self.size = size
        self.iter_max = iter_max
        self.range_min_list = range_min_list
        self.range_max_list = range_max_list
        self.is_cal_max = is_cal_max

    def get_alg_with_surr(self, surr):
        Alg_Class = get_alg(self.alg_name)
        alg_obj = Alg_Class(self.dim, self.size, self.iter_max, self.range_min_list, self.range_max_list)
        alg_obj.fitfunction = surr.fit_one
        alg_obj.is_cal_max = True
        return alg_obj
