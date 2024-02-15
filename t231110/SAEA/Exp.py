"""实验类"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import time
import random
import string
import os
import numpy as np
from EA.algs.base.problem import get_problem_detail
from SAEA.algs.get_alg import get_alg
from SAEA.algs.selection_strategy.ss_mapping import get_selection_strategy

class Exp:
    def __init__(self, problem_name, alg_name, id=None,
                 dim=2, init_size=100, pop_size=60,
                 fit_max=200, iter_max=60,
                 surr_types=[["smt_kplsk"], ["smt_kplsk"]],
                 ea_types=["FWA_Surr_Impl2", "DE_Surr_Base"],
                 selection_strategy="MixedNonlinearly",
                 ss_args={
                    "list": ["Global", "Local"],
                    "rate": [0.8, 0.2],
                    "use_status": [False, True],
                    "changing_type": "FirstStageDecreasesNonlinearly",
                    "iter_max": 100,
                    "p_max_first": 0.8,
                    "p_min_first": 0.1,
                    "k": 2
                }, fws="NegativeCorrelation"):
        self.id = self.get_id() if id is None else id

        # 问题
        self.problem_name = problem_name
        self.problem = None

        # SAEA算法参数
        self.alg_name = alg_name
        self.alg = None
        self.dim = dim
        """维度"""
        self.init_size = init_size
        """初始化采样数量"""
        self.pop_size = pop_size
        """种群大小"""
        self.fit_max = fit_max
        """适应度评价次数上限"""
        self.iter_max = iter_max
        """EA迭代次数上限"""
        self.surr_types = surr_types
        """代理模型类型，二维数组，第一维为全局代理模型，第二维为局部代理模型"""
        self.ea_types = ea_types
        """进化算法类型，二维数组，第一维为全局进化算法，第二维为局部进化算法"""

        # HSAEA算法参数
        self.selection_strategy = selection_strategy
        """选择策略"""
        self.ss_args = ss_args
        """选择策略参数"""
        
        # FWA_Surr算法参数
        self.fws = fws
        """不确定性权重计算策略"""

        self.start_time = None
        self.end_time = None
        self.save_path = f'./exp/{self.id}'

    def init(self):
        # 获取当前文件所在目录
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(self.save_path, exist_ok=True)
        os.chdir(self.save_path)

        # 【修改项】维度
        # dim = 10

        lb, ub, dim, fobj = get_problem_detail(self.problem_name, ndim=self.dim)
        print(f'测试函数：{self.problem_name}')
        print(f'搜索空间：{lb} ~ {ub}')
        print(f'维度：{dim}')
        self.problem = fobj

        # 【修改项】参数设置
        # init_size = 100
        # pop_size = 60
        # # surr_type = "kriging"
        # surr_types = [["smt_kplsk"], ["smt_kplsk"]]
        # ea_type = "FWA_Surr_Impl2"
        # ea_types = ["FWA_Surr_Impl2", "DE_Surr_Base"]
        # fit_max = 1000
        # iter_max = 60

        # init_size = 100
        # pop_size = 60
        # # surr_type = "kriging"
        # surr_types = [["my_sklearn_gpr"], ["smt_kplsk"]]
        # # surr_types = [["sklearn_gpr"], ["smt_kplsk"]]
        # ea_type = "FWA_Surr_Impl2"
        # ea_types = ["FWA_Surr_Impl2", "DE_Surr_Base"]
        # fit_max = 176
        # iter_max = 60

        is_cal_max = False

        # 【修改项】选择策略
        if "HSAEA" in self.alg_name:
            """HSAEA选择策略参数"""
            selection_strategy = get_selection_strategy(self.selection_strategy, self.ss_args)

        range_max_list = np.ones(self.dim) * ub
        range_min_list = np.ones(self.dim) * lb
        print("range_max_list: ", range_max_list)
        print("range_min_list: ", range_min_list)

        # smt库的代理模型训练参数
        surr_setting = {
            "print_training": False,
            "print_prediction": False,
            "print_problem": False,
            "print_solver": False,
        }

        alg_class = get_alg(self.alg_name)
        if "HSAEA" in self.alg_name:
            if "FD_HSAEA" in self.alg_name:
                self.alg = alg_class(self.dim, self.init_size, self.pop_size, self.surr_types, 
                                     self.ea_types, self.fit_max, self.iter_max,
                                     range_min_list, range_max_list, is_cal_max, surr_setting,
                                     selection_strategy=selection_strategy, fws=self.fws)
            else:
                self.alg = alg_class(self.dim, self.init_size, self.pop_size, self.surr_types, 
                                 self.ea_types, self.fit_max, self.iter_max,
                                 range_min_list, range_max_list, is_cal_max, surr_setting,
                                 selection_strategy=selection_strategy)
        else:
            self.alg = alg_class(self.dim, self.init_size, self.pop_size, self.surr_types[0], 
                                 self.ea_types[0], self.fit_max, self.iter_max,
                                 range_min_list, range_max_list, is_cal_max, surr_setting)
        self.alg.fitfunction = fobj
        self.alg.fitfunction_name = self.problem_name

    def solve(self):
        self.init()
        self.alg.run()


    def get_id(self):
        """
        根据日期时间生成id

        格式：“YYYYMMDDHHMMSS_XXXXX”，其中“XXXXX”为随机字符串
        """
        date_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        random_str = ''.join(random.sample(string.ascii_letters + string.digits, 5))
        id = f'{date_time}_{random_str}'
        return id