"""
基于状态的选择策略
"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.algs.selection_strategy.base import BaseSelectionStrategy


class StatusSelectionStrategy(BaseSelectionStrategy):
    """
    基于状态的选择策略
    """
    def __init__(self, list):
        # 第一次会选择list的第一个，后面会根据状态选择
        super().__init__(list)
        self.prev_idx = 0
        """上一次选择的状态"""
    
        
    def select(self, flag=None):
        """
        选择，如果上一次成功选择了，返回上一次选择的状态，否则返回下一个状态
        """
        if flag is None or flag:
            return self.list[self.prev_idx]
        else:
            self.prev_idx = (self.prev_idx + 1) % len(self.list)
            return self.list[self.prev_idx]