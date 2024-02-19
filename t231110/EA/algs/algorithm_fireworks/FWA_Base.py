"""差分进化算的基类"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110/EA")

import numpy as np
from algs.algorithm_fireworks.FWA_Unit import FWA_Unit
from algs.base.alg import Algorithm_Impl

class FWA_Base(Algorithm_Impl):
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list,
                 is_cal_max=True, fitfunction=None, silent=False,
                 m=5, a=0.25, b=0.9, spec_num=5, amplitude_max=30):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list, 
                         is_cal_max=is_cal_max, fitfunction=fitfunction, silent=silent)
        # 算法信息
        self.name = 'FWA'
        self.unit_class = FWA_Unit

        # 参数
        self.m = m
        """正常火星数量基数，生成正常火星的数量: m*b ~ m*a"""
        self.a = a
        """正常火星数量下限倍数"""
        self.b = b
        """正常火星数量上限倍数"""
        self.spec_num = spec_num
        """生成特殊火星的数量"""
        self.amplitude_max = amplitude_max
        """烟花振幅最大值"""

        # 中间值
        self.all_list = []
        """所有火星，包括烟花、正常火星、特殊火星"""
        self.unit_list = []
        """正常火星列表"""

    def init(self, unit_list=[]):
        """初始化unit_list和其中的position、fitness"""
        super().init(unit_list=unit_list)
        if len(unit_list) == 0:
            self.unit_list = []
            for i in range(self.size):
                unit = FWA_Unit()
                unit.position = np.random.uniform(self.range_min_list, self.range_max_list, self.dim)
                unit.fitness = self.cal_fitfunction(unit.position)
                self.unit_list.append(unit)
        else:
            self.unit_list = unit_list
    
    def update(self, iter):
        super().update(iter)
        for i in range(self.size):
            self.obtain_spark(i)

        for num in range(self.spec_num):
            rand_id = np.random.randint(0, self.size)
            self.obtain_spec_spark(rand_id)

        self.select_sparks()

    def obtain_spark(self, id):
        """生成正常火星"""
        num_sparks = self.get_spark_num(id)
        amplitude = self.get_spark_amplitude(id)
        for n in range(num_sparks):
            pos = self.unit_list[id].position.copy()
            for d in range(self.dim):
                if np.random.rand() < 0.5:
                    pos[d] += amplitude * np.random.uniform(-1, 1)
                    # pos[d] = self.get_out_bound_value_rand(pos[d], self.range_min_list[d], self.range_max_list[d])
            pos = self.get_out_bound_value_rand(pos, self.range_min_list, self.range_max_list)
            spark = FWA_Unit()
            spark.position = pos
            spark.fitness = self.cal_fitfunction(pos)
            self.all_list.append(spark)

    def obtain_spec_spark(self, id):
        """生成特殊火星"""
        pos = self.unit_list[id].position.copy()
        for d in range(self.dim):
            if np.random.rand() < 0.5:
                pos[d] *= np.random.normal(1, 1)
                # pos[d] = self.get_out_bound_value_rand(pos[d], self.range_min_list[d], self.range_max_list[d])
        pos = self.get_out_bound_value_rand(pos, self.range_min_list, self.range_max_list)
        spec_spark = FWA_Unit()
        spec_spark.position = pos
        spec_spark.fitness = self.cal_fitfunction(pos)
        self.all_list.append(spec_spark)

    def select_sparks(self):
        """选择火星"""
        self.all_list.extend(self.unit_list)
        values = np.array([unit.fitness for unit in self.all_list])
        # 按照适应度值的大小进行排序，并选择最优的size个个体
        sort_index = np.argsort(-values)
        # if self.is_cal_max:
        #     sort_index = np.argsort(-values)
        # else:
        #     sort_index = np.argsort(values)
        for i in range(self.size):
            self.unit_list[i].position = self.all_list[sort_index[i]].position
            self.unit_list[i].fitness = self.all_list[sort_index[i]].fitness
        # self.all_list = [self.all_list[i] for i in sort_index]
        # self.unit_list = self.all_list[:self.size]
        self.all_list = []


        # max_value_id = np.argmax(values)

        # distances = np.abs(values[:, None] - values)
        # dist_sum = np.sum(distances, axis=1)
        # rates = dist_sum / np.sum(dist_sum)

        # for i in range(self.size):
        #     if i == 0:
        #         self.unit_list[0].position = self.all_list[max_value_id].position
        #         self.unit_list[0].fitness = self.all_list[max_value_id].fitness
        #     else:
        #         chosen_id = np.random.choice(range(len(self.all_list)), p=rates)
        #         self.unit_list[i].position = self.all_list[chosen_id].position
        #         self.unit_list[i].fitness = self.all_list[chosen_id].fitness

        self.all_list = []

    def get_spark_num(self, id):
        """计算火星数量"""
        value_min = min(unit.fitness for unit in self.unit_list)
        value_sum = sum(unit.fitness for unit in self.unit_list)
        value = self.unit_list[id].fitness
        num = round(self.m * (value - value_min + np.finfo(float).eps) / 
                    (value_sum - self.size * value_min + np.finfo(float).eps))
        num = max(round(self.a * self.m), min(num, round(self.b * self.m)))
        return num

    def get_spark_amplitude(self, id):
        """计算烟花振幅"""
        value_max = max(unit.fitness for unit in self.unit_list)
        value_sum = sum(unit.fitness for unit in self.unit_list)
        value = self.unit_list[id].fitness
        amplitude = self.amplitude_max * (value_max - value + np.finfo(float).eps) / (self.size * value_max - value_sum + np.finfo(float).eps)
        return amplitude

    def get_best_id(self):
        """找到具有最大适应度值的个体的索引"""
        values = np.array([unit.fitness for unit in self.unit_list])
        best_id = np.argmax(values)
        return best_id
    
    def toStr(self):
        str = super().toStr()
        str += f"m: {self.m}\n"
        str += f"a: {self.a}\n"
        str += f"b: {self.b}\n"
        str += f"spec_num: {self.spec_num}\n"
        str += f"amplitude_max: {self.amplitude_max}\n"
        return str
