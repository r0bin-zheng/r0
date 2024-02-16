"""遗传算法实现类"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

from algs.algorithm_genetic.GA_Base import GA_Base


class GA_Impl(GA_Base):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list)
        self.name = 'GA_Impl'
        
