"""差分算法实现"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

from algs.differential_evolution.DE_Base import DE_Base

class DE_Impl(DE_Base):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list)
        