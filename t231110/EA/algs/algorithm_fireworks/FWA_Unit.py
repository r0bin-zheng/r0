"""烟花算法个体类"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

from algs.base.unit import Unit

class FWA_Unit(Unit):
    def __init__(self):
        super().__init__()
        """个体不确定性，用于代理模型辅助算法中"""
        self.uncertainty = None
        """个体值，有时候判断不使用适配度，而是使用值"""
        self.value = None
        