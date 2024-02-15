"""不确定性权重计算策略"""


WT_DICT = {
    # 不考虑不确定性
    "NoUncertainty": {
        "w_strategy": 4,
        "delta": 0.0
    },

    # 常数权重
    "Constant": {
        "w_strategy": 4,
        "delta": 0.5
    },

    # 不确定性权重线性下降
    "LinearDecrease": {
        "w_strategy": 2,
        "beta": 1.0
    },

    # 不确定性权重指数下降
    "ExponentialDecrease": {
        "w_strategy": 1,
        "alpha": 0.5
    },

    # 不确定性和代理模型集合的相关系数负相关
    "NegativeCorrelation": {
        "w_strategy": 3,
        "gamma": 0.3
    }
    
}