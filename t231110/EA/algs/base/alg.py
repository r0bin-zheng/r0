"""优化算法基类"""

import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

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

        # 初始化
        self.position_best = np.zeros(self.dim)
        self.value_best = -np.finfo(np.float64).max
        self.value_best_history = []
        self.position_best_history = []

    def run(self):
        import time
        start_time = time.time()
        self.init()
        self.iteration()
        end_time = time.time()
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

        print(f"第 {iter} 代")
        if self.is_cal_max:
            self.value_best_history.append(self.value_best)
            print(f"最优值= {self.value_best}")
        else:
            self.value_best_history.append(-self.value_best)
            print(f"最优值= {-self.value_best}")

        self.position_best_history.append(self.position_best)
        print(f"最优解= {self.position_best}")



    def cal_fitfunction(self, position):
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
    
    
    def draw2_gif(self, step, is_save, name):
        if self.dim < 2:
            print('维度太低，无法绘制图像')
            return
        if step < 1:
            step = 1

        images = []

        for i in range(1, self.iter_max + 1):
            print("draw 2d gif iter: ", i)
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
                    images[0].save(filename, save_all=True, append_images=images[1:], loop=0, duration=100)
                buf.close()

            plt.close()

        # if is_save and images:
        #     filename = f"{name}_2d.gif"
        #     images[0].save(filename, save_all=True, append_images=images[1:], loop=0, duration=100)


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

