"""
代理模型辅助的差分进化算法基类
"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from EA.algs.differential_evolution.DE_Base import DE_Base

class DE_Surr_Base(DE_Base):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list)
        self.name = 'DE_Surr_Base'
        self.surr = None
