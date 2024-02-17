import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from EA.algs.utils import Ea_Factory
from SAEA.algs.base.SAEA_Base import SAEA_Base
from SAEA.algs.fwa_surr.FWA_SAEA_Unit import FWA_SAEA_Unit

class FWA_SAEA(SAEA_Base):
    def __init__(self, dim, init_size, pop_size, surr_type, ea_type, fit_max,
                 iter_max, range_min_list, range_max_list, is_cal_max, surr_setting):
        ea_type = "FWA_Surr_Impl2"
        super().__init__(dim, init_size, pop_size, surr_type, ea_type, fit_max,
                         iter_max, range_min_list, range_max_list, is_cal_max, surr_setting)

        self.name = 'FWA_SAEA'
        self.unit_class = FWA_SAEA_Unit

    def init_ea_factory(self):
        ea_args = {
            "dim": self.dim,
            "size": self.pop_size,
            "iter_max": self.iter_max,
            "range_min_list": self.range_min_list,
            "range_max_list": self.range_max_list,
            "is_cal_max": True,
            "silent": True,
            'iter_max_main': self.fit_max - self.init_size,
            'w_strategy_str': self.fws,
        }
        self.ea_factory = Ea_Factory(self.ea_type, ea_args)

    def get_ea_dynamic_args(self):
        return {'iter_num_main': self.iter_num}
