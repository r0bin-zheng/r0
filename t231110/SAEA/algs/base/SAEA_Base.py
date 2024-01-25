"""SAEA算法基类"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import io
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from EA.algs.base.alg import Algorithm_Impl
from EA.algs.utils import get_alg, Ea_Factory
from smt.sampling_methods import LHS
from pymoo.operators.sampling.lhs import LHS as LHS_pymoo
from SAEA.algs.base.SAEA_Unit import SAEA_Unit
from SAEA.utils.surrogate_model import SurrogateFactory
from SAEA.surrogates.factory import SurrogateModelFactory


class SAEA_Base:
    """
    代理模型辅助进化算法基类
    """
    def __init__(self, dim, init_size, pop_size, surr_types, ea_type, fit_max, iter_max, range_min_list, range_max_list, is_cal_max, surr_setting=None):
        self.name = 'SAEA'
        self.dim = dim
        self.init_size = init_size
        self.pop_size = pop_size
        self.fit_max = fit_max
        self.iter_max = iter_max
        self.range_min_list = range_min_list
        self.range_max_list = range_max_list
        self.is_cal_max = is_cal_max
        self.silence = False
        self.save_flag = True

        self.iter_num = 0
        self.cal_fit_num = 0
        self.unit_list = []
        self.surr_type = surr_types
        """
        代理模型类型
        type: list
        """
        # self.surr_factory = SurrogateFactory([surr_type], surr_setting)
        self.surr_factory = SurrogateModelFactory(surr_types, surr_setting)
        self.ea_type = ea_type
        self.ea_factory = Ea_Factory(ea_type, dim, pop_size, iter_max, range_min_list, range_max_list, is_cal_max)
        self.surr = None
        self.surr_history = []
        self.fitfunction_name = None
        self.fitfunction = None
        self.unit_new = []

        self.value_best = None
        self.value_best_history = []
        self.position_best = None
        self.time_cost = None

    def run(self):
        self.start_time = time.time()
        self.init()
        self.iteration()
        self.end_time = time.time()
        self.time_cost = self.end_time - self.start_time
        self.print_result()
        self.save()


    def init(self):
        self.unit_new = []
        self.unit_new_history = []
        self.cal_fit_num = 0
        self.iter_num = 0
        self.position_best = np.zeros(self.dim)
        self.value_best = -np.finfo(np.float64).max
        self.init_sampling()

    def iteration(self):
        while self.cal_fit_num < self.fit_max:
            self.update()
            self.print_iter()
            self.save_iter()
            self.iter_num += 1

    def init_sampling(self):
        xlimts = []
        for d in range(self.dim):
            lb = self.range_min_list[d]
            ub = self.range_max_list[d]
            xlimts.append([lb, ub])
        xlimts = np.array(xlimts)
        sampling = LHS(xlimits=xlimts)
        init_pop = sampling(self.init_size)
        for i in range(self.init_size):
            unit = SAEA_Unit()
            unit.position = init_pop[i]
            unit.fitness_true = self.cal_fitfunction(unit.position)
            self.unit_list.append(unit)
            if unit.fitness_true > self.value_best:
                self.value_best = unit.fitness_true
                self.position_best = unit.position.copy()

    def update(self):
        self.create_surrogate()
        self.optimize()
        self.merge()
        self.update_status()

    def create_surrogate(self):
        X = []
        y = []
        for unit in self.unit_list:
            X.append(unit.position)
            y.append(unit.fitness_true)
        X= np.array(X)
        y = np.array(y)
        self.surr = self.surr_factory.create(X, y)
        self.surr_history.append(self.surr)

    def optimize(self):
        ea = self.ea_factory.get_alg_with_surr(self.surr)
        selected_unit = self.select()
        ea.silence = True
        ea.set_unit_list(selected_unit)
        ea.run()
        unit = SAEA_Unit()
        unit.position = ea.position_best
        unit.fitness = ea.value_best
        self.unit_new = [unit]
        print("unit_new position: ", unit.position, " fitness: ", unit.fitness)
        ea.draw2_gif(10, True, f"SAEA_iter{self.iter_num}")

    def merge(self):
        for ind in self.unit_new:
            ind.fitness_true = self.cal_fitfunction(ind.position)
            self.unit_list.append(ind)
            if ind.fitness_true > self.value_best:
                self.value_best = ind.fitness_true
                self.position_best = ind.position.copy()
        self.value_best_history.append(self.value_best)
    
    def update_status(self):
        pass

    def select(self):
        """
        选择数量为pop_size的个体
        默认策略为选择fitness最大的个体
        """
        unit_copy = self.unit_list.copy()
        unit_copy.sort(key=lambda x: x.fitness_true, reverse=True)
        return unit_copy[:self.pop_size]

    def cal_fitfunction(self, position):
        if self.fitfunction is None:
            value = 0
        else:
            value = self.fitfunction(position) if self.is_cal_max else -self.fitfunction(position)
        self.cal_fit_num += 1
        return value
    
    def cal_surr_fitfunction(self, position):
        if self.surr is None:
            value = 0
        else:
            # value = self.surr.fit_one(position)
            value = self.surr.predict_one(position.reshape(1, -1))[0]
        return value
    
    def print_iter(self):
        if self.silence:
            return
        print('迭代次数：', self.iter_num, ' 最优解：', self.position_best, ' 最优值：', self.value_best)

    def save_iter(self):
        pass

    def print_result(self):
        print('最优解：', self.position_best)
        print('最优值：', self.value_best if self.is_cal_max else -self.value_best)
        print('时间：', self.time_cost)

    def save(self):
        if not self.save_flag:
            return
        self.save_result()
        self.draw_fit_2d_gif(10, True, self.name)
        self.draw_surr_2d_gif(10, True, self.name)
        self.draw_value_best()

    def save_result(self):
        with open('result.txt', 'w') as f:
            f.write(f"--------------------------------------Info--------------------------------------\n")
            f.write(f"alg: {self.name}\n")
            f.write(f"problem: {self.fitfunction_name}\n")
            f.write(f"dim: {self.dim}\n")
            f.write(f"init_size: {self.init_size}\n")
            f.write(f"pop_size: {self.pop_size}\n")
            f.write(f"fit_max: {self.fit_max}\n")
            # f.write(f"iter_max: {self.iter_max}\n")
            f.write(f"range_min_list: {self.range_min_list}\n")
            f.write(f"range_max_list: {self.range_max_list}\n")
            f.write(f"is_cal_max: {self.is_cal_max}\n")
            f.write(f"surr_type: {self.surr_type}\n")
            f.write(f"ea_type: {self.ea_type}\n")

            f.write(
                f"--------------------------------------Result--------------------------------------\n")
            f.write(f"value_best: {self.value_best}\n")
            f.write(f"position_best: {self.position_best}\n")
            f.write(f"time_cost: {self.time_cost}\n")
            f.write(f"pop:\n")
            for i in range(len(self.unit_list)):
                f.write(f"#{i} {self.unit_list[i].position} -- {self.unit_list[i].fitness_true if self.is_cal_max else -self.unit_list[i].fitness_true}\n")
        print("save result at result.txt")
    
    def draw_fit_2d_gif(self, step, is_save, name=None):
        """绘制迭代过程中真实背景和种群个体"""
        if self.dim < 2:
            print('维度太低，无法绘制图像')
            return
        if step < 1:
            step = 1

        name = name if name else self.name
        images = []

        # 创建绘制函数深度图的网格数据
        x = np.linspace(self.range_min_list[0], self.range_max_list[0], 100)
        y = np.linspace(self.range_min_list[1], self.range_max_list[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.fitfunction(np.array([X[i, j], Y[i, j]]))

        for i in range(1, self.iter_num + 1):
            if i % step > 0 and i > 1:
                continue

            plt.figure(figsize=(8, 6))
            # 绘制函数的深度图
            plt.imshow(Z, extent=[self.range_min_list[0], self.range_max_list[0], 
                                self.range_min_list[1], self.range_max_list[1]],
                    origin='lower', cmap='viridis', alpha=0.7)
            plt.colorbar()  # 显示颜色条

            point_num = self.init_size + i
            for s in range(point_num):
                cur_position = self.unit_list[s].position
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
            
        print("draw pop fitness 2d gif at ", name + "_2d.gif")

    def draw_surr_2d_gif(self, step, is_save, name=None):
        """绘制迭代过程中代理模型背景和种群个体"""
        if self.dim < 2:
            print('维度太低，无法绘制图像')
            return
        if step < 1:
            step = 1

        name = name if name else self.name
        images = []

        # 创建绘制函数深度图的网格数据
        x = np.linspace(self.range_min_list[0], self.range_max_list[0], 100)
        y = np.linspace(self.range_min_list[1], self.range_max_list[1], 100)
        X, Y = np.meshgrid(x, y)

        for i in range(1, self.iter_num + 1):
            if i % step > 0 and i > 1:
                continue

            Z = np.zeros_like(X)
            t = 1 if self.is_cal_max else -1
            for ii in range(X.shape[0]):
                for jj in range(X.shape[1]):
                    Z[ii, jj] = self.surr_history[i-1].predict_one(np.array([X[ii, jj], Y[ii, jj]])) * t

            plt.figure(figsize=(8, 6))
            # 绘制函数的深度图
            plt.imshow(Z, extent=[self.range_min_list[0], self.range_max_list[0], 
                                self.range_min_list[1], self.range_max_list[1]],
                    origin='lower', cmap='viridis', alpha=0.7)
            plt.colorbar()  # 显示颜色条

            point_num = self.init_size + i
            for s in range(point_num):
                cur_position = self.unit_list[s].position
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
                    filename = f"{name}_surr_2d.gif"
                    images[0].save(filename, save_all=True, append_images=images[1:], loop=0, duration=300)
                buf.close()

            plt.close()
            
        print("draw pop surr 2d gif at ", name + "_surr_2d.gif")

    def draw_value_best(self):
        """
        绘制value_best收敛曲线
        """
        value_best_history_copy = self.value_best_history.copy()
        if self.is_cal_max == False:
            value_best_history_copy = -np.array(value_best_history_copy)

        plt.figure(figsize=(8, 6))
        plt.plot(value_best_history_copy)
        plt.xlabel('iter')
        plt.ylabel('value_best')
        plt.title('value_best')
        if self.save_flag:
            plt.savefig(f"{self.name}_value_best.png")
        else:
            plt.show()

        print("draw value_best at ", self.name + "_value_best.png")