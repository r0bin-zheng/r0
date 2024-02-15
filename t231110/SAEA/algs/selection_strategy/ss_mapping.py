"""
选择策略映射
"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.algs.selection_strategy.MixedSelectedStrategy import MixedSelectedStrategy
from SAEA.algs.selection_strategy.FixedSelectionStrategy import FixedSelectionStrategy

SS_DICT = {
    "JustGlobal": {
        "class": FixedSelectionStrategy,
        "default_args": {
            "list": ["Global", "Local"],
            "rate": [1.0, 0.0],
            "max_select_times": 100
        },
    },
    "JustLocal": {
        "class": FixedSelectionStrategy,
        "default_args": {
            "list": ["Global", "Local"],
            "rate": [0.0, 1.0],
            "max_select_times": 100
        },
    },
    "HalfHalf": {
        "class": FixedSelectionStrategy,
        "default_args": {
            "list": ["Global", "Local"],
            "rate": [0.5, 0.5],
            "max_select_times": 100
        },
    },
    "MixedLinearly": {
        "class": MixedSelectedStrategy,
        "default_args": {
            "list": ["Global", "Local"],
            "rate": [0.8, 0.2],
            "use_status": [False, True],
            "changing_type": "FirstStageDecreasesLinearly",
            "iter_max": 100,
            "p_max_first": 0.8,
            "p_min_first": 0.1
        },
    },
    "MixedNonlinearly": {
        "class": MixedSelectedStrategy,
        "default_args": {
            "list": ["Global", "Local"],
            "rate": [0.8, 0.2],
            "use_status": [False, True],
            "changing_type": "FirstStageDecreasesNonlinearly",
            "iter_max": 100,
            "p_max_first": 0.8,
            "p_min_first": 0.1,
            "k": 2
        },
    },
}


def get_selection_strategy(str, args):
    """
    获取选择策略
    """
    if str in SS_DICT:
        class_ = SS_DICT[str]["class"]
        default_args = SS_DICT[str]["default_args"]
        _args = default_args.copy()
        if args is not None:
            for k, v in _args.items():
                if k in args:
                    _args[k] = v
        ss = class_(**_args)
    else:
        raise Exception("选择策略不存在")
    return ss
