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
from SAEA.algs.ga_surr.GA_Surr_Base import GA_Surr_Base
from SAEA.algs.pso_surr.PSO_Surr_Base import PSO_Surr_Base


alg_dict = {
    "PSO": PSO_Base,
    "PSO_Surr_Base": PSO_Surr_Base,

    "GA": GA_Base,
    "GA_Surr_Base": GA_Surr_Base,

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
    def __init__(self, alg_name, static_args={}):
        self.alg_name = alg_name
        self.static_args = static_args
        self.alg_class = get_alg(alg_name)

    def create(self, dynamic_args={}):
        """获取算法对象"""
        args = self.static_args.copy()
        args.update(dynamic_args)
        return self.alg_class(**args)

    def create_with_surr(self, surr, dynamic_args={}):
        """获取带有代理模型的算法对象"""
        ea = self.create(dynamic_args)
        ea.surr = surr
        ea.fitfunction = surr.predict_one
        return ea

        # Alg_Class = get_alg(self.alg_name)
        # size = size if size is not None else self.size
        # range_min_list = range_min_list if range_min_list is not None else self.range_min_list
        # range_max_list = range_max_list if range_max_list is not None else self.range_max_list
        
        # alg_obj = Alg_Class(dim=self.dim, 
        #                     size=self.size, 
        #                     iter_max=self.iter_max, 
        #                     range_max_list=range_min_list, 
        #                     range_max_list=range_max_list,
        #                     surr=surr)
        # alg_obj.fitfunction = surr.predict_one
        # alg_obj.is_cal_max = True
        # return alg_obj
