"""粒子群算法个体类"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

from algs.base.unit import Unit

class PSO_Unit(Unit):
    def __init__(self):
        super().__init__()
        self.velocity = None
        self.position_best = None
        