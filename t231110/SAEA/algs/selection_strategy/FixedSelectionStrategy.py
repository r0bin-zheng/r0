"""
固定比例选择策略
"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.algs.selection_strategy.base import BaseSelectionStrategy

class FixedSelectionStrategy(BaseSelectionStrategy):
    """
    固定比例选择策略，每个选项会按照rate的顺序和比例被选中
    """
    def __init__(self, list, rate, max_select_times):
        super().__init__(list)
        self.rate = rate
        """list中每个选项的被选中比例"""
        self.max_select_times = max_select_times
        """最大选择次数"""

        if not self.check():
            raise ValueError("rate error")

    def check(self):
        """
        检查比例之和是否为1，每个选项是否都有比例，大小是否都大于0小于等于1
        """
        sum = 0.0
        for r in self.rate:
            if r < 0 or r > 1:
                return False
            sum += r
        if sum != 1 or len(self.list) != len(self.rate):
            return False
        return True


    def select(self, flag=None, iter=None):
        """
        选择
        """
        self.select_times += 1
        cur_rate = self.select_times / self.max_select_times
        sum = 0.0
        for i in range(len(self.list)):
            sum += self.rate[i]
            if cur_rate <= sum:
                return self.list[i]
