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

        self.global_ea_info = ""
        """全局EA信息"""
        self.local_ea_info = ""
        """局部EA信息"""
        self.ea_info_flag = [True, True]

    def get_optimizer(self):
        """
        获取优化器
        """
        ea = super().get_optimizer()
        if self.use_local == False:
            ea.set_w_strategy(self.fws)
            if self.ea_info_flag[0]:
                self.global_ea_info = ea.toStr()
                self.ea_info_flag[0] = False
        else:
            if self.ea_info_flag[1]:
                self.local_ea_info = ea.toStr()
                self.ea_info_flag[1] = False
        return ea

    def toStr(self):
        str = super().toStr()
        str += "fws: {}\n".format(self.fws)
        str += "\n"
        str += "global_ea_info\n"
        str += self.global_ea_info
        str += "\n"
        str += "local_ea_info\n"
        str += self.local_ea_info
        return str


        
