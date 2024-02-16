import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.algs.base.SAEA_Base import SAEA_Base
from SAEA.algs.ga_surr.GA_SAEA_Unit import GA_SAEA_Unit

class GA_SAEA_Base(SAEA_Base):
    def __init__(self, dim, init_size, pop_size, surr_type, ea_type, fit_max,
                 iter_max, range_min_list, range_max_list, is_cal_max, surr_setting):
        ea_type = "GA_Surr_Base"
        super().__init__(dim, init_size, pop_size, surr_type, ea_type, fit_max,
                         iter_max, range_min_list, range_max_list, is_cal_max, surr_setting)
        self.name = 'GA_SAEA_Base'
        self.unit_class = GA_SAEA_Unit
        