"""
选择策略基类
"""

class BaseSelectionStrategy(object):
    """
    选择策略基类
    使用字符串列表记录策略名称
    """
    def __init__(self, list):
        self.list = list

        # 运行时参数
        self.select_times = 0
        """已选择次数"""
        self.max_select_times = None
        """最大选择次数，None时表示无限制"""

    def reset(self):
        """
        重置选择次数
        """
        self.select_times = 0

    def select(self):
        pass
    