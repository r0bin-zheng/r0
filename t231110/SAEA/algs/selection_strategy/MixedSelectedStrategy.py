"""
混合了概率和状态的选择策略
"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from SAEA.algs.selection_strategy.base import BaseSelectionStrategy


class MixedSelectedStrategy(BaseSelectionStrategy):
    """
    混合了概率和状态的选择策略
    """
    def __init__(self, list, rate, use_status, changing_type=None, iter_max=None):
        super().__init__(list)
        self.rate = rate
        """list中每个选项的被选中的概率"""
        self.use_status = use_status
        """list中的每个选项是否使用状态选择"""
        self.prev_idx = 0
        """上一次选择的索引"""
        self.changing_type = changing_type
        """
        变化类型
        类型: String
        可选值: 
        - "NoChange" 不改变比例
        - "FirstStageDecreasesLinearly" 首阶段概率线性下降
        - "FirstStageDecreasesNonlinearly" 首阶段概率非线性下降
        """
        self.p_max_first = 0.8
        """首阶段的最大概率"""
        self.p_min_first = 0.1
        """首阶段的最小概率"""
        self.iter_max = iter_max
        """算法的最大迭代次数，用于计算概率变化时量化算法进度"""

    def check(self):
        """
        检查比例之和是否为1，每个选项是否都有比例，大小是否都大于0小于等于1
        """
        sum = 0.0
        for r in self.rate:
            if r <= 0 or r > 1:
                return False
            sum += r
        if sum != 1 or len(self.list) != len(self.rate):
            return False
        return True


    def select(self, flag=None, rate_new=None, iter=None):
        """
        选择

        如果上一次选项使用状态选择，则若成功继续选择上一次的选项
        若失败退回到概率选择

        如果上一次选项使用概率选择，则进行概率选择
        """
        self.select_times += 1
        if self.use_status[self.prev_idx] and flag:
            return self.list[self.prev_idx]
        
        # 更新概率
        if self.changing_type is None or self.changing_type == "NoChange":
            self.rate = rate_new if rate_new is not None else self.rate
        elif iter is not None:
            if self.changing_type == "FirstStageDecreasesLinearly":
                # 公式：p = (iter_max - iter) / iter_max * (p_max - p_min) + p_min
                p_first = (self.iter_max - iter) / self.iter_max * (self.p_max_first - self.p_min_first) + self.p_min_first
            elif self.changing_type == "FirstStageDecreasesNonlinearly":
                # 公式：p = ((iter_max - iter) / iter_max)^k * (p_max - p_min) + p_min
                k = 1 # k值越大，下降越快；k=1时为线性下降
                p_first = ((self.iter_max - iter) / self.iter_max) ** k * (self.p_max_first - self.p_min_first) + self.p_min_first
            self.rate = np.array([p_first, 1 - p_first])
        else:
            self.rate = rate_new if rate_new is not None else self.rate

        if not self.check():
            raise ValueError("rate error")
        
        self.prev_idx = np.random.choice(range(len(self.list)), p=self.rate)
        return self.list[self.prev_idx]
