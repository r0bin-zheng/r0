"""
FD_HSAEA_Base.py
基于烟花、差分进化算法的层次代理模型辅助进化算法基类
"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.algs.base.HSAEA_Base import HSAEA_Base


class FD_HSAEA_Base(HSAEA_Base):
    def __init__(self, dim, init_size, pop_size, surr_types, ea_types, fit_max,
                 iter_max, range_min_list, range_max_list, is_cal_max, surr_setting,
                 selection_strategy=None, fws="NegativeCorrelation"):
        super().__init__(dim, init_size, pop_size, surr_types, ea_types, fit_max,
                         iter_max, range_min_list, range_max_list, is_cal_max, surr_setting,
                         selection_strategy)
        self.name = 'FD_HSAEA_Base'
        self.fws = fws
        """FWA_Surr的不确定性权重计算策略"""

    def get_optimizer(self):
        """
        获取优化器
        """
        if self.use_local == False:
            ea = super().get_optimizer()
            ea.set_fws(self.fws)

    def toStr(self):
        str = super().toStr()
        str += "fws: {}\n".format(self.fws)
        return str


        
