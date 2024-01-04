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

class Exp:
    def __init__(self, problem_name, alg_name):
        self.id = self.get_id()

        self.problem_name = problem_name
        self.problem = None

        self.alg_name = alg_name
        self.alg = None

        self.start_time = None
        self.end_time = None
        self.save_path = f'./exp/{self.id}'

    def init(self):
        os.mkdir(self.save_path)
        os.chdir(self.save_path)

        lb, ub, dim, fobj = get_problem_detail(self.problem_name)
        print(f'测试函数：{self.problem_name}')
        print(f'搜索空间：{lb} ~ {ub}')
        print(f'维度：{dim}')
        self.problem = fobj

        dim = 2
        init_size = 50
        pop_size = 50
        surr_type = "kriging"
        ea_type = "DE"
        fit_max = 100
        iter_max = 50
        range_max_list = np.ones(dim) * ub
        range_min_list = np.ones(dim) * lb
        is_cal_max = False
        surr_setting = {
            "print_training": False,
            "print_prediction": False,
            "print_problem": False,
            "print_solver": False,
        }

        alg_class = get_alg(self.alg_name)
        self.alg = alg_class(dim, init_size, pop_size, surr_type, ea_type, fit_max,
                        iter_max, range_min_list, range_max_list, is_cal_max, surr_setting)
        self.alg.fitfunction = fobj

    def solve(self):
        self.init()
        self.alg.run()

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