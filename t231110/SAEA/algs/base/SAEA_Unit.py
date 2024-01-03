"""SAEA个体类"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from EA.algs.base.unit import Unit

class SAEA_Unit(Unit):
    """
    SAEA个体类
    对于SAEA算法，个体类需要记录真实适应度和新位置
    self.position: 位置
    self.fitness: 代理模型预测的适应度
    self.fitness_true: 真实适应度
    """
    def __init__(self):
        super().__init__()
        self.fitness_true = None
        # self.position_new = None
