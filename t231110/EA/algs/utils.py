import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from EA.algs.algorithm_fireworks.FWA_Base import FWA_Base
from EA.algs.algorithm_fireworks.FWA_Impl import FWA_Impl
from EA.algs.differential_evolution.DE_Base import DE_Base
from EA.algs.differential_evolution.DE_Impl import DE_Impl
from EA.algs.algorithm_particle_swarm.PSO_Base import PSO_Base
from EA.algs.algorithm_genetic.GA_Base import GA_Base
from SAEA.algs.fwa_surr.FWA_Surr_Base import FWA_Surr
from SAEA.algs.fwa_surr.FWA_Surr_Impl1 import FWA_Surr_Impl1
from SAEA.algs.fwa_surr.FWA_Surr_Impl2 import FWA_Surr_Impl2
from SAEA.algs.de_surr.DE_Surr_Base import DE_Surr_Base


alg_dict = {
    "PSO": PSO_Base,

    "GA": GA_Base,

    "DE": DE_Base,
    "DE_Impl": DE_Impl,
    "DE_Surr_Base": DE_Surr_Base,

    "FWA": FWA_Base,
    "FWA_Impl": FWA_Impl,
    "FWA_Surr": FWA_Surr,
    "FWA_Surr_Impl1": FWA_Surr_Impl1,
    "FWA_Surr_Impl2": FWA_Surr_Impl2,
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

    def get_alg_with_surr(self, surr, range_min_list=None, range_max_list=None):
        range_min_list = range_min_list if range_min_list is not None else self.range_min_list
        range_max_list = range_max_list if range_max_list is not None else self.range_max_list
        Alg_Class = get_alg(self.alg_name)
        alg_obj = Alg_Class(self.dim, self.size, self.iter_max, range_min_list, range_max_list)
        alg_obj.fitfunction = surr.predict_one
        alg_obj.is_cal_max = True
        return alg_obj
