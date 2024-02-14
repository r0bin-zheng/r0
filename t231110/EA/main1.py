"""优化算法main"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

import numpy as np
from algs.base.problem import get_problem_detail
from algs.utils import get_alg


def get_id():
    """根据日期时间生成id"""

    # 格式：YYYYMMDDHHMMSS_XXXXX
    # 其中 XXXXX 为随机字符串

    import time
    import random
    import string

    # 日期时间
    date_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

    # 随机字符串
    random_str = ''.join(random.sample(string.ascii_letters + string.digits, 5))

    # 拼接
    id = f'{date_time}_{random_str}'
    return id


def run_task(problem="F12022", alg="DE", dim=2):

    # 在exp文件夹下生成一个以id命名的文件夹
    id = get_id()
    import os
    os.mkdir(f'exp/{id}')
    os.chdir(f'exp/{id}')

    # 选择测试函数
    function_name = problem
    lb, ub, dim, fobj = get_problem_detail(function_name, ndim=10)
    print(f'测试函数：{function_name}')
    print(f'搜索空间：{lb} ~ {ub}')
    print(f'维度：{dim}')

    # 算法实例化
    size = 50
    iter_max = 200
    range_max_list = np.ones(dim) * ub
    range_min_list = np.ones(dim) * lb

    # 实例化算法类
    Alg_Class = get_alg(alg)
    base = Alg_Class(dim, size, iter_max, range_min_list, range_max_list)
    base.is_cal_max = False
    base.fitfunction = fobj

    # 运行
    base.run()
    base.save_result()
    print("call fitness func times: ", base.cal_fit_num)

    # 绘制2D和3D图像
    base.draw2_gif(10, True, base.name)
    # base.draw3_gif(10, True, base.name)


def main():
    # problem = "Griewank"
    # problem = "Ackley01"
    problem = "F62005"

    ndim = 10
    alg = "DE"
    # alg = "FWA"
    run_task(problem, alg, ndim)


main()