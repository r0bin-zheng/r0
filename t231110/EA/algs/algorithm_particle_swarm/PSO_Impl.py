"""PSO算法实现类"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

from algs.algorithm_particle_swarm.PSO_Base import PSO_Base


class PSO_Impl(PSO_Base):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list)
        self.name = 'PSO_Impl'
        
