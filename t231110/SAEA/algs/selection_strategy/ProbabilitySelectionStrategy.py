"""
基于概率的选择策略
"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from SAEA.algs.selection_strategy.base import BaseSelectionStrategy


class ProbabilitySelectionStrategy(BaseSelectionStrategy):
    """
    基于概率的选择策略，每个选项会按照rate的顺序和比例被选中
    """
    def __init__(self, list, rate, max_select_times=None):
        super().__init__(list)
        self.rate = rate
        """list中每个选项的被选中的概率"""
        self.max_select_times = max_select_times
        """最大选择次数，None时表示无限制"""


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


    def select(self, rate_new=None):
        """
        基于rate的概率进行选择
        """
        self.rate = rate_new if rate_new is not None else self.rate
        if not self.check():
            raise ValueError("rate error")
        
        self.select_times += 1
        return np.random.choice(self.list, p=self.rate)
