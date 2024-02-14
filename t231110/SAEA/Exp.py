"""实验类"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import time
import random
import string
import os
import traceback
import numpy as np
from EA.algs.base.problem import get_problem_detail
from SAEA.algs.get_alg import get_alg

class Exp:
    def __init__(self, problem_name, alg_name, id=None):
        self.id = self.get_id() if id is None else id

        self.problem_name = problem_name
        self.problem = None

        self.alg_name = alg_name
        self.alg = None

        self.start_time = None
        self.end_time = None
        self.save_path = f'./exp/{self.id}'

    def init(self):
        # 获取当前文件所在目录
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(self.save_path, exist_ok=True)
        os.chdir(self.save_path)

        dim = 10

        lb, ub, dim, fobj = get_problem_detail(self.problem_name, ndim=dim)
        print(f'测试函数：{self.problem_name}')
        print(f'搜索空间：{lb} ~ {ub}')
        print(f'维度：{dim}')
        self.problem = fobj

        # 参数设置
        init_size = 100
        pop_size = 60
        surr_type = "kriging"
        surr_types = [["smt_kplsk"], ["smt_kplsk"]]
        ea_type = "FWA_Surr_Impl2"
        ea_types = ["FWA_Surr_Impl2", "DE_Surr_Base"]
        fit_max = 1000
        iter_max = 60
        # init_size = 100
        # pop_size = 60
        # # surr_type = "kriging"
        # surr_types = [["my_sklearn_gpr"], ["smt_kplsk"]]
        # # surr_types = [["sklearn_gpr"], ["smt_kplsk"]]
        # ea_type = "FWA_Surr_Impl2"
        # ea_types = ["FWA_Surr_Impl2", "DE_Surr_Base"]
        # fit_max = 176
        # iter_max = 60

        range_max_list = np.ones(dim) * ub
        range_min_list = np.ones(dim) * lb
        print("range_max_list: ", range_max_list)
        print("range_min_list: ", range_min_list)

        is_cal_max = False
        surr_setting = {
            "print_training": False,
            "print_prediction": False,
            "print_problem": False,
            "print_solver": False,
        }

        alg_class = get_alg(self.alg_name)
        if "HSAEA" in self.alg_name:
            self.alg = alg_class(dim, init_size, pop_size, surr_types, ea_types, fit_max,
                                 iter_max, range_min_list, range_max_list, is_cal_max, surr_setting)
        else:
            self.alg = alg_class(dim, init_size, pop_size, surr_types, ea_type, fit_max,
                        iter_max, range_min_list, range_max_list, is_cal_max, surr_setting)
        self.alg.fitfunction = fobj
        self.alg.fitfunction_name = self.problem_name

    def solve(self):
        self.init()
        self.alg.run()
        # try:
        #     self.alg.run()
        # except Exception as e:
        #     # print(e)
        #     # traceback.print_exc()


    def get_id(self):
        """根据日期时间生成id"""

        # 格式：YYYYMMDDHHMMSS_XXXXX
        # 其中 XXXXX 为随机字符串
        date_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

        # 随机字符串
        random_str = ''.join(random.sample(string.ascii_letters + string.digits, 5))

        # 拼接
        id = f'{date_time}_{random_str}'
        return id