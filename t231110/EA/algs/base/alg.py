"""优化算法基类"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import io
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from EA.algs.base.unit import Unit

class Algorithm_Impl:
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list):

        # 维度
        self.dim = dim
        # 种群大小
        self.size = size
        # 最大迭代次数
        self.iter_max = iter_max
        # 解空间下界和上界
        self.range_min_list = range_min_list
        self.range_max_list = range_max_list
        # 默认为最大化
        self.is_cal_max = True
        # 适应度函数（需要外部定义）
        self.fitfunction = None
        # 适应度函数调用计数
        self.cal_fit_num = 0
        # 是否静默
        self.silence = False
        self.unit_class = Unit

        # 初始化
        self.position_best = np.zeros(self.dim)
        self.value_best = -np.finfo(np.float64).max
        self.value_best_history = []
        self.position_best_history = []

    def run(self):
        start_time = time.time()
        self.init()
        self.iteration()
        end_time = time.time()
        self.time_cost = end_time - start_time
        if not self.silence:
            # self.print_result()
            print(f"运行时间: {end_time - start_time}")

    def init(self):
        self.position_best = np.zeros(self.dim)
        self.value_best_history = []
        self.position_best_history = []
        self.value_best = -np.finfo(np.float64).max

    def iteration(self):
        for iter in range(1, self.iter_max + 1):
            self.update(iter)

    def update(self, iter):
        for i in range(self.size):
            if self.unit_list[i].fitness > self.value_best:
                self.value_best = self.unit_list[i].fitness
                self.position_best = self.unit_list[i].position
            self.unit_list[i].save()

        if not self.silence:
            print(f"第 {iter} 代")
        if self.is_cal_max:
            self.value_best_history.append(self.value_best)
            if not self.silence:
                print(f"最优值= {self.value_best}")
        else:
            self.value_best_history.append(-self.value_best)
            if not self.silence:
                print(f"最优值= {-self.value_best}")

        self.position_best_history.append(self.position_best)
        if not self.silence:
            print(f"最优解= {self.position_best}")

    def cal_fitfunction(self, position):
        """
        计算适应度函数值，对于最大化问题，返回值不变；对于最小化问题，返回值取相反数
        """
        if self.fitfunction is None:
            value = 0
        else:
            value = self.fitfunction(position) if self.is_cal_max else -self.fitfunction(position)
        self.cal_fit_num += 1
        return value

    def get_out_bound_value(self, position, min_list=None, max_list=None):
        min_list = self.range_min_list if min_list is None else min_list
        max_list = self.range_max_list if max_list is None else max_list
        position_tmp = np.clip(position, min_list, max_list)
        return position_tmp

    def get_out_bound_value_rand(self, position, min_list=None, max_list=None):
        min_list = self.range_min_list if min_list is None else min_list
        max_list = self.range_max_list if max_list is None else max_list
        position_tmp = position.copy()
        position_rand = np.random.uniform(min_list, max_list, position.shape)
        position_tmp[position < min_list] = position_rand[position < min_list]
        position_tmp[position > max_list] = position_rand[position > max_list]
        return position_tmp
    
    def set_unit_list(self, unit_list):
        self.unit_list = []
        for i in range(len(unit_list)):
            if i >= self.size:
                break
            unit = self.unit_class()
            unit.position = unit_list[i].position.copy()
            # print("position: ", unit.position, "shape: ", unit.position.shape)
            unit.fitness = self.fitfunction(X=unit.position)
            # print("fitness: ", unit.fitness)
            self.unit_list.append(unit)
    
    def save_result(self):
        import time
        """保存self.value_best_history、self.position_best_history、self.value_best、self.position_best到result.txt"""
        with open('result.txt', 'w') as f:

            # str_time = time.strftime("%Y-%m-%d %H:%M:%S", self.time)

            # f.write(
            #     f"--------------------------------------Info--------------------------------------\n")
            # f.write(f"ID: {self.id}\n")
            # f.write(f"Path: {self.path}\n")
            # f.write(f"Time: {str_time}\n")
            # f.write(f"Success: {self.success}\n")

            # f.write(
            #     f"--------------------------------------Problem--------------------------------------\n")
            # f.write(f"Problem: {self.problem_name}\n")
            # f.write(f"Number of objects: {self.n_obj}\n")
            # f.write(f"Number of variables: {self.n_var}\n")

            # f.write(
            #     f"--------------------------------------Algorithm--------------------------------------\n")
            # f.write(f"Algorithm choice: {self.alg}\n")
            # f.write(f"Phase list: {self.phase_list} ({phases})\n")
            # f.write(f"Phase strategy: {strategy_list[self.strategy]}\n")
            # f.write(f"First phase rate: {self.rate}\n")
            # f.write(f"Population size: {self.mu}\n")
            # f.write(f"Maximum number of function evaluations: {self.max_fe}\n")

            f.write(
                f"--------------------------------------Result--------------------------------------\n")
            f.write(f"value_best: {self.value_best}\n")
            f.write(f"position_best: {self.position_best}\n")

            f.write(f"history:\n")
            for i in range(len(self.value_best_history)):
                f.write(f"{i + 1}: {self.position_best_history[i]} {self.value_best_history[i]}\n")
    
    def draw2_gif1(self, step, is_save, name):
        if self.dim < 2:
            print('维度太低，无法绘制图像')
            return
        if step < 1:
            step = 1

        images = []

        for i in range(1, self.iter_max + 1):
            # print("draw 2d gif iter: ", i)
            if i % step > 0 and i > 1:
                continue

            plt.figure(figsize=(8, 6))
            for s in range(self.size):
                # print(self.unit_list[s].position_history_list)
                cur_position = self.unit_list[s].position_history_list[i-1, :]
                plt.scatter(cur_position[0], cur_position[1], 10, 'b')

            range_size_x = self.range_max_list[0] - self.range_min_list[0]
            range_size_y = self.range_max_list[1] - self.range_min_list[1]
            plt.axis([self.range_min_list[0] - 0.2 * range_size_x, self.range_max_list[0] + 0.2 * range_size_x,
                    self.range_min_list[1] - 0.2 * range_size_y, self.range_max_list[1] + 0.2 * range_size_y])
            plt.text(self.range_min_list[0], self.range_max_list[1], str(i), fontsize=20)
            plt.gca().set_aspect('equal', adjustable='box')

            if is_save:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                images.append(img)
                if images:
                    filename = f"{name}_2d.gif"
                    images[0].save(filename, save_all=True, append_images=images[1:], loop=0, duration=200)
                buf.close()

            plt.close()

        print("draw 2d gif done.")
        # if is_save and images:
        #     filename = f"{name}_2d.gif"
        #     images[0].save(filename, save_all=True, append_images=images[1:], loop=0, duration=100)

    def draw2_gif(self, step, is_save, name):
        if self.dim < 2:
            print('维度太低，无法绘制图像')
            return
        if step < 1:
            step = 1

        images = []

        # 创建绘制函数深度图的网格数据
        x = np.linspace(self.range_min_list[0], self.range_max_list[0], 100)
        y = np.linspace(self.range_min_list[1], self.range_max_list[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.fitfunction(np.array([X[i, j], Y[i, j]]))

        for i in range(1, self.iter_max + 1):
            if i % step > 0 and i > 1:
                continue

            plt.figure(figsize=(8, 6))
            # 绘制函数的深度图
            plt.imshow(Z, extent=[self.range_min_list[0], self.range_max_list[0], 
                                self.range_min_list[1], self.range_max_list[1]],
                    origin='lower', cmap='viridis', alpha=0.7)
            plt.colorbar()  # 显示颜色条

            for s in range(self.size):
                cur_position = self.unit_list[s].position_history_list[i-1, :]
                plt.scatter(cur_position[0], cur_position[1], 10, 'b')  # 保持原来的散点样式

            # 设置图形范围和比例
            range_size_x = self.range_max_list[0] - self.range_min_list[0]
            range_size_y = self.range_max_list[1] - self.range_min_list[1]
            plt.axis([self.range_min_list[0] - 0.2 * range_size_x, self.range_max_list[0] + 0.2 * range_size_x,
                    self.range_min_list[1] - 0.2 * range_size_y, self.range_max_list[1] + 0.2 * range_size_y])
            plt.text(self.range_min_list[0], self.range_max_list[1], str(i), fontsize=20)
            plt.gca().set_aspect('equal', adjustable='box')

            if is_save:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                images.append(img)
                if images:
                    filename = f"{name}_2d.gif"
                    images[0].save(filename, save_all=True, append_images=images[1:], loop=0, duration=300)
                buf.close()

            plt.close()

        # if is_save and images:
        #     filename = f"{name}_2d.gif"
        #     images[0].save(filename, save_all=True, append_images=images[1:], loop=0, duration=200)
            
        print("draw 2d gif done.")


    def draw3_gif(self, step, is_save, name):
        if self.dim < 3:
            print('维度太低，无法绘制三维图像')
            return
        if step < 1:
            step = 1

        images = []

        for i in range(1, self.iter_max + 1):
            if i % step > 0 and i > 1:
                continue

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            for s in range(self.size):
                cur_position = self.unit_list[s].position_history_list[i - 1, :]
                ax.scatter(cur_position[0], cur_position[1], cur_position[2], 10, 'b')

            range_size_x = self.range_max_list[0] - self.range_min_list[0]
            range_size_y = self.range_max_list[1] - self.range_min_list[1]
            range_size_z = self.range_max_list[2] - self.range_min_list[2]
            ax.set_xlim([self.range_min_list[0] - 0.1 * range_size_x, self.range_max_list[0] + 0.1 * range_size_x])
            ax.set_ylim([self.range_min_list[1] - 0.1 * range_size_y, self.range_max_list[1] + 0.1 * range_size_y])
            ax.set_zlim([self.range_min_list[2] - 0.1 * range_size_z, self.range_max_list[2] + 0.1 * range_size_z])
            ax.text(self.range_min_list[0], self.range_max_list[1], self.range_max_list[2], str(i), fontsize=20)

            if is_save:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                images.append(img)
                buf.close()

            plt.close()

        if is_save and images:
            filename = f"{name}_3d.gif"
            images[0].save(filename, save_all=True, append_images=images[1:], loop=0, duration=100)

