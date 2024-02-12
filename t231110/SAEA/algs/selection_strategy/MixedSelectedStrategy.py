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
    def __init__(self, list, rate, use_status):
        super().__init__(list)
        self.rate = rate
        """list中每个选项的被选中的概率"""
        self.use_status = use_status
        """list中的每个选项是否使用状态选择"""
        self.prev_idx = 0

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


    def select(self, flag=None, rate_new=None):
        """
        选择

        如果上一次选项使用状态选择，则若成功继续选择上一次的选项
        若失败退回到概率选择

        如果上一次选项使用概率选择，则进行概率选择
        """
        self.select_times += 1
        if self.use_status[self.prev_idx] and flag:
            return self.list[self.prev_idx]
        
        self.rate = rate_new if rate_new is not None else self.rate
        if not self.check():
            raise ValueError("rate error")
        
        self.prev_idx = np.random.choice(range(len(self.list)), p=self.rate)
        return self.list[self.prev_idx]
