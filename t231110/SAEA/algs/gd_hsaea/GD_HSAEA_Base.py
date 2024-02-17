"""全局GA局部DE的HSAEA算法基类"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.algs.base.HSAEA_Base import HSAEA_Base


class GD_HSAEA_Base(HSAEA_Base):
    def __init__(self, dim, init_size, pop_size, surr_types, ea_types, fit_max, iter_max, 
                 range_min_list, range_max_list, is_cal_max, surr_setting=None,
                 selection_strategy_str="MixedNonlinearly", pop_size_local=30):
        super().__init__(dim, init_size, pop_size, surr_types, ea_types, fit_max, iter_max, 
                         range_min_list, range_max_list, is_cal_max, surr_setting,
                         selection_strategy_str, pop_size_local)
        
        # 算法信息
        self.name = 'GD_HSAEA_Base'
        """算法名称"""
        self.ea_type_global = "GA_Surr_Base"
        """全局进化算法设置"""
        self.ea_type_local = "DE_Surr_Base"
        """局部进化算法设置"""

