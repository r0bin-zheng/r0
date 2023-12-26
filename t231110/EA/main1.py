"""优化算法main"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

import numpy as np
from algs.base.problem import get_problem_detail
from algs.utils import get_alg


def run_task(problem="F12022", alg="DE", dim=2):

    # 选择测试函数
    function_name = problem
    lb, ub, dim, fobj = get_problem_detail(function_name)
    print(f'测试函数：{function_name}')
    print(f'搜索空间：{lb} ~ {ub}')
    print(f'维度：{dim}')

    # 算法实例化
    size = 50
    iter_max = 300
    range_max_list = np.ones(dim) * ub
    range_min_list = np.ones(dim) * lb

    # 实例化差分进化算法类
    Alg_Class = get_alg(alg)
    base = Alg_Class(dim, size, iter_max, range_min_list, range_max_list)
    base.is_cal_max = False
    base.fitfunction = fobj

    # 运行
    base.run()
    print("call fitness func times: ", base.cal_fit_num)

    # 绘制2D和3D图像
    base.draw2_gif(10, True, base.name)
    # base.draw3_gif(10, True, base.name)


def main():
    problem = "F12022"
    ndim = 2
    alg = "DE"
    run_task(problem, alg, ndim)


main()