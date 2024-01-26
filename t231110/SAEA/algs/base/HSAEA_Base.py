"""
hierarchical surrogate assisted evolutionary algorithm
分层代理模型辅助进化算法
"""

import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from EA.algs.utils import get_alg, Ea_Factory
from SAEA.algs.base.SAEA_Base import SAEA_Base
from SAEA.algs.base.SAEA_Unit import SAEA_Unit
from SAEA.utils.surrogate_model import SurrogateFactory
from SAEA.surrogates.factory import SurrogateModelFactory

class HSAEA_Base(SAEA_Base):
    """
    层次代理模型辅助进化算法基类
    
    参数
    ----------
    surr_types: list(二维数组)

    """
    def __init__(self, dim, init_size, pop_size, surr_types, ea_types, fit_max,
                 iter_max, range_min_list, range_max_list, is_cal_max, surr_setting):
        ea_type = ea_types[0]
        surr_type = surr_types[0]
        super().__init__(dim, init_size, pop_size, surr_type, ea_type, fit_max,
                         iter_max, range_min_list, range_max_list, is_cal_max, surr_setting)
        self.name = 'HSAEA_Base'
        self.pop_size_local = init_size
        self.pop_local = []
        
        self.surr_type_global = surr_types[0]
        """
        本地代理模型类型
        type: list
        """

        self.surr_type_local = surr_types[1]
        """
        全局代理模型类型
        type: list
        """

        self.surr_factory_global = self.surr_factory
        self.surr_factory_local = SurrogateModelFactory(surr_types[1], surr_setting)
        
        self.ea_type_global = ea_types[0]
        """全局进化算法设置"""

        self.ea_type_local = ea_types[1]
        """局部进化算法设置"""

        self.ea_factory_global = self.ea_factory
        self.ea_factory_local = Ea_Factory(ea_types[1], dim, pop_size, iter_max, range_min_list, range_max_list, is_cal_max)
        
        self.use_local = False
        """使用局部代理的flag"""

        self.range_min_list_local = range_min_list
        self.range_max_list_local = range_max_list

    def update(self):
        print("use local surr model" if self.use_local else "use global surr model")
        return super().update()

    def create_surrogate(self):
        X = []
        y = []
        for unit in self.unit_list:
            X.append(unit.position)
            y.append(unit.fitness_true)
        X= np.array(X)
        y = np.array(y)
        if self.use_local:
            # 选择欧式距离最靠近最优解的前1/5个点
            dist = np.linalg.norm(X - self.position_best, axis=1)
            idx = np.argsort(dist)
            idx = idx[:int(len(idx) / 5)]
            self.pop_size_local = len(idx)
            self.pop_local = []
            for i in range(self.pop_size_local):
                unit = SAEA_Unit()
                unit.position = X[idx[i]]
                unit.fitness = y[idx[i]]
                self.pop_local.append(unit)
            X = X[idx]
            y = y[idx]
            self.surr = self.surr_factory_local.create(X, y)
            # 更新边界, range_min_list_local为X每个维度的最小值，range_max_list_local为X每个维度的最大值
            self.range_min_list_local = np.min(X, axis=0)
            self.range_max_list_local = np.max(X, axis=0)
        else:
            self.surr = self.surr_factory_global.create(X, y)
        self.surr_history.append(self.surr)

    # def optimize(self):
    #     if self.use_local:
    #         ea = self.ea_factory_local.get_alg_with_surr(self.surr, self.range_min_list_local, self.range_max_list_local)
    #     else:
    #         ea = self.ea_factory_global.get_alg_with_surr(self.surr)
    #     ea.surr = self.surr
    #     ea.size = len(self.unit_list)
    #     ea.iter_num_main = self.iter_num
    #     ea.iter_max_main = self.fit_max - self.init_size
    #     ea.silence = True
        # selected_unit = self.select()
    #     ea.set_unit_list(selected_unit)
    #     ea.run()
    #     unit = SAEA_Unit()
    #     unit.position = ea.position_best
    #     unit.fitness = ea.value_best
    #     self.unit_new = [unit]
    #     print("unit_new position: ", unit.position, " fitness: ", unit.fitness)
    #     ea.draw2_gif(10, True, f"HSAEA_iter{self.iter_num}")

    def optimize(self):
        optimizer = self.get_optimizer()
        optimizer.run()
        unit = self.get_best_unit(optimizer)
        self.unit_new = [unit]
        print("unit_new position: ", unit.position, " fitness: ", unit.fitness)
        if self.draw_gif_flag:
            optimizer.draw2_gif(10, True, f"SAEA_iter{self.iter_num}")

    def get_optimizer(self):
        """获取优化器（进化算法）"""
        if self.use_local:
            ea = self.ea_factory_local.get_alg_with_surr(self.surr, self.range_min_list_local, self.range_max_list_local)
        else:
            ea = self.ea_factory_global.get_alg_with_surr(self.surr)
        ea.surr = self.surr
        if self.use_local:
            ea.size = min(self.pop_size_local, self.pop_size)
        else:
            ea.size = self.pop_size 
        ea.iter_num_main = self.iter_num
        ea.iter_max_main = self.fit_max - self.init_size
        ea.silence = True
        selected_unit = self.pop_local if self.use_local else self.unit_list
        self.init_ea_population(selected_unit, ea)
        # ea.set_unit_list(selected_unit)
        return ea

    def init_ea_population(self, pop, ea):
        """初始化进化算法种群"""
        for i in range(len(pop)):
            if i > ea.size:
                break
            ind = pop[i]
            unit = ea.unit_class()
            unit.position = ind.position.copy()
            unit.fitness = ind.fitness
            ea.unit_list.append(unit)

        # 预测每个个体的值和方差
        pos_list = [unit.position for unit in ea.unit_list]
        if self.surr.predict_value_variance and ea.use_variances:
            # 如果代理模型支持预测方差且进化算法使用方差，则预测每个个体的值和方差
            value_list, var_list = self.surr.predict_value_variance(pos_list)
            for i in range(len(ea.unit_list)):
                ea.unit_list[i].fitness = value_list[i]
                ea.unit_list[i].uncertainty = var_list[i]
        else:
            # 否则只预测每个个体的值
            value_list = self.surr.predict(pos_list)
            for i in range(len(ea.unit_list)):
                ea.unit_list[i].fitness = value_list[i]
    
    def get_best_unit(self, optimizer):
        """添加去重功能"""
        unit_list = optimizer.unit_list
        unit_new = SAEA_Unit()
        if optimizer.use_variances:
            unit_list.sort(key=lambda x: x.value, reverse=True)
        else:
            unit_list.sort(key=lambda x: x.fitness, reverse=True)
        flag = False
        for unit in unit_list:
            pos_new = unit.position
            for unit_exist in self.unit_list:
                pos_exist = unit_exist.position
                if np.allclose(pos_new, pos_exist):
                    continue
                else:
                    unit_new.position = unit.position
                    unit_new.fitness = unit.fitness
                    flag = True
                    break
            if flag:
                break
        return unit_new

    def update_status(self):
        value_best = self.value_best_history[-1]
        if len(self.value_best_history) < 2:
            value_best_prev = value_best
        else:
            value_best_prev = self.value_best_history[-2]
        prev_use_local = self.use_local

        if self.use_local:
            # 如果局部代理模型的最优值大于上一代的最优值，则继续使用局部代理模型
            if value_best > value_best_prev:
                self.use_local = True
            else:
                self.use_local = False
        else:
            # 如果全局代理模型的最优值大于上一代的最优值，则继续使用全局代理模型
            if value_best > value_best_prev:
                self.use_local = False
            else:
                self.use_local = True
        

