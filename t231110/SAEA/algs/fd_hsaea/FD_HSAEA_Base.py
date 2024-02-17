"""
FD_HSAEA_Base.py
基于烟花、差分进化算法的层次代理模型辅助进化算法基类
"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.algs.base.HSAEA_Base import HSAEA_Base
from EA.algs.utils import Ea_Factory


class FD_HSAEA_Base(HSAEA_Base):
    def __init__(self, dim, init_size, pop_size, surr_types, ea_types, fit_max, iter_max, 
                 range_min_list, range_max_list, is_cal_max, surr_setting=None,
                 selection_strategy_str="MixedNonlinearly", pop_size_local=30, 
                 fws="NegativeCorrelation"):
        super().__init__(dim, init_size, pop_size, surr_types, ea_types, fit_max, iter_max, 
                         range_min_list, range_max_list, is_cal_max, surr_setting,
                         selection_strategy_str, pop_size_local)
        
        # 算法信息
        self.name = 'FD_HSAEA_Base'
        """算法名称"""
        self.ea_type_global = "FWA_Surr_Impl2"
        """全局进化算法设置"""
        self.ea_type_local = "DE_Surr_Base"
        """局部进化算法设置"""

        # 参数
        self.fws = fws
        """FWA_Surr的不确定性权重计算策略"""

    def init_ea_factory(self):
        global_args = {
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
        local_args = {
            "dim": self.dim,
            "size": self.pop_size_local,
            "iter_max": self.iter_max,
            "is_cal_max": True,
            "silent": True,
        }
        self.ea_factory_global = Ea_Factory(self.ea_type_global, static_args=global_args)
        self.ea_factory_local = Ea_Factory(self.ea_type_local, static_args=local_args)
        self.ea_factory = self.ea_factory_global

    def get_ea_dynamic_args(self):
        if self.use_local == False:
            """FWA_Surr需要设置额外参数iter_num_main"""
            return {
                'iter_num_main': self.iter_num,
            }
        else:
            return {
                "size": min(len(self.pop_local), self.pop_size),
                "range_min_list": self.range_min_list_local,
                "range_max_list": self.range_max_list_local,
            }
