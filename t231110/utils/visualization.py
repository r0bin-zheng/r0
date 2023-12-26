# 结果可视化
# 优化算法的指标主要体现：
# 1. 精准度：解的质量、分布情况、
# 2. 效率：花费时间、收敛曲线

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from common import *

# ---------通用---------

# 箱型图展示多次运行单个指标的分布，展示稳定性

# 绘制代理模型的预测曲线与真实曲线的对比
# 需要数据：代理模型、上下界
# 适用于一维变量的输入
def plot_surrogate_1d_with_x2(problem, surrogate, xl, xu, samples_x=None, samples_y=None, x2_default=0, save_path=None):
    # 绘制图像，横坐标为输入变量，纵坐标为输出变量
    # x2_default为第二维变量的默认值
    X = np.linspace(xl, xu, 100)
    X_1 = np.c_[X, np.ones_like(X)*x2_default]
    X_2 = np.c_[np.ones_like(X)*x2_default, X]
    y_1 = surrogate.predict_values(X_1)
    y_2 = surrogate.predict_values(X_2)
    y_true_1 = problem.evaluate(X_1)
    y_true_2 = problem.evaluate(X_2)

    # 以平滑曲线展示
    # 如果有采样点，也展示采样点
    if samples_x is not None and samples_y is not None:
        # 获取采样点的第一维数据
        samples_x_1 = samples_x[:, 0]
        plt.scatter(samples_x_1, samples_y, c="r", label="samples")
        samples_x_2 = samples_x[:, 1]

    plt.plot(X, y_1, label="surrogate")
    plt.plot(X, y_true_1, label="true")
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.title("surrogate_1")
    plt.legend()

    # 保存图像
    if save_path is not None:
        plt.savefig(save_path + "/surrogate_1.png")
        plt.close()
    else:
        plt.show()
    
    # 如果有采样点，也展示采样点
    if samples_x is not None and samples_y is not None:
        # 获取采样点的第二维数据
        samples_x_2 = samples_x[:, 1]
        plt.scatter(samples_x_2, samples_y, c="r", label="samples")
    
    plt.plot(X, y_2, label="surrogate")
    plt.plot(X, y_true_2, label="true")
    plt.xlabel("x2")
    plt.ylabel("y")
    plt.title("surrogate_2")
    plt.legend()

    # 保存图像
    if save_path is not None:
        plt.savefig(save_path + "/surrogate_2.png")
        plt.close()
    else:
        plt.show()


# --------单目标--------

# 最优解收敛曲线
# 需要数据：总迭代次数，每次迭代的最优解
def plot_best_sln_curve(best_sln_list, n_epoch, save_path=None):
    # 绘制图像，横坐标为迭代次数，纵坐标为最优解
    plt.plot(np.arange(n_epoch), best_sln_list)
    plt.xlabel("epoch")
    plt.ylabel("best_sln")
    plt.title("best_sln_curve")
    # 保存图像
    if save_path is not None:
        plt.savefig(save_path + "/best_sln_curve.png")
        plt.close()
    else:
        plt.show()

# 二维变量的解决方案在输入空间的分布
# 需要数据：解集、上下界
def plot_data_2d(X, xl, xu, save_path=None):
    # 绘制图像，横坐标为第一维变量，纵坐标为第二维变量
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(xl, xu)
    plt.ylim(xl, xu)
    plt.title("data_2d")
    # 保存图像
    if save_path is not None:
        plt.savefig(save_path + "/data_2d.png")
        plt.close()
    else:
        plt.show()

# 三维变量的解决方案在输入空间的分布
# 需要数据：解集

# 二维变量+一维输出的解决方案的分布
# 需要数据：解集、上下界
def plot_data_2d_1f(X, y, xl, xu, save_path=None, file_name=None):
    # 绘制图像，横坐标为第一维变量，纵坐标为第二维变量，颜色为第三维变量
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(xl, xu)
    plt.ylim(xl, xu)
    plt.title(file_name if file_name is not None else "data_2d_1f")
    # 保存图像
    if save_path is not None:
        save_file_name = save_path + "/" + ("data_2d_1f" if file_name is None else file_name) + ".png"
        plt.savefig(save_file_name)
        plt.close()
    else:
        plt.show()

# 箱型图展示多维点的分布
# 需要数据：解集

# gif展示解集的生成过程
def plot_gif(Xs, num, xl, xu, save_path):
    for i in range(num):
        plot_data_2d_1f(Xs["X_"+str(i)], Xs["y_"+str(i)], xl, xu, save_path, "data_2d_1f_"+str(i))
    create_gif(save_path, "data_2d_1f", "data_2d_1f_", 0, num, 0.5)

def plot_gif_from_file(load_path, range, xl, xu, save_path):
    data = np.load(load_path)
    plot_gif(data, range, xl, xu, save_path)

# --------多目标--------

# IGD收敛曲线
# 需要数据：每一次迭代的IGD、总迭代次数
def plot_igd(igds, n_epoch, save_path=None):
    # 绘制图像，横坐标为迭代次数，纵坐标为IGD
    plt.plot(np.arange(n_epoch), igds)
    plt.xlabel("epoch")
    plt.ylabel("IGD")
    plt.title("IGD_curve")
    # 保存图像
    if save_path is not None:
        plt.savefig(save_path + "/IGD_curve.png")
        plt.close()
    else:
        plt.show()

# HV收敛曲线
# 需要数据：每一次迭代的HV、总迭代次数

# 二目标问题的解决方案在输出空间的分布
# 需要数据：解集

# 三目标问题的解决方案在输出空间的分布
# 需要数据：解集
