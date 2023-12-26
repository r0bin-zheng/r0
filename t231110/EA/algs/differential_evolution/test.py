import numpy as np
import matplotlib.pyplot as plt
from DE_Base import DE_Base
from algs.base.problem import get_problem_detail

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

# 假设 Get_Functions_details 是一个可用的函数，返回函数细节
# from some_module import Get_Functions_details, DE_Base

# 清除之前的数据（在Python中通常不需要）

# 选择测试函数
function_name = 'F12022'
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
base = DE_Base(dim, size, iter_max, range_min_list, range_max_list)
base.is_cal_max = False
base.fitfunction = fobj

# 运行
base.run()
print("call fitness func times: ", base.cal_fit_num)

# 绘制2D和3D图像（确保已经实现了 draw2_gif 和 draw3_gif 方法）
base.draw2_gif(10, True, base.name)
# base.draw3_gif(10, True, base.name)

# # 绘制图像
# plt.figure(figsize=(10, 5))

# # 绘制搜索空间
# plt.subplot(1, 2, 1)
# # 假设 func_plot 是一个可用的函数，用于绘制函数的图像
# # func_plot(Function_name)
# plt.title('Parameter space')
# plt.xlabel('x_1')
# plt.ylabel('x_2')
# # plt.zlabel(f'{function_name}( x_1 , x_2 )')

# # 绘制目标空间
# plt.subplot(1, 2, 2)
# plt.semilogy(base.value_best_history, color='r')
# plt.title('Objective space')
# plt.xlabel('Iteration')
# plt.ylabel('Best score obtained so far')
# plt.grid(True)
# plt.legend([base.name])

# print(f'The best solution obtained by {base.name} is {base.position_best}')
# print(f'The best optimal value of the objective function found by {base.name} is {base.value_best}')

# plt.tight_layout()
# plt.show()
