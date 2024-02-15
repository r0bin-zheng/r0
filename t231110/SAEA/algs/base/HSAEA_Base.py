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
from SAEA.surrogates.factory import SurrogateModelFactory
from SAEA.algs.selection_strategy.MixedSelectedStrategy import MixedSelectedStrategy

class HSAEA_Base(SAEA_Base):
    """
    层次代理模型辅助进化算法基类
    
    参数
    ----------
    surr_types: list(二维数组)

    """
    def __init__(self, dim, init_size, pop_size, surr_types, ea_types, fit_max,
                 iter_max, range_min_list, range_max_list, is_cal_max, surr_setting,
                 selection_strategy=None):
        ea_type = ea_types[0]
        surr_type = surr_types[0]
        super().__init__(dim, init_size, pop_size, surr_type, ea_type, fit_max,
                         iter_max, range_min_list, range_max_list, is_cal_max, surr_setting)
        self.name = 'HSAEA_Base'
        self.pop_size_local = 30
        """局部代理模型训练样本数量，当数字大于1时为数量，当数字小于等于1时为比例"""
        self.pop_local = []
        """局部代理模型的训练样本"""
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
        """全局代理模型工厂"""
        self.surr_factory_local = SurrogateModelFactory(surr_types[1], surr_setting)
        """局部代理模型工厂"""    
        self.ea_type_global = ea_types[0]
        """全局进化算法设置"""
        self.ea_type_local = ea_types[1]
        """局部进化算法设置"""
        self.ea_factory_global = self.ea_factory
        """全局进化算法工厂"""
        self.ea_factory_local = Ea_Factory(ea_types[1], dim, pop_size, iter_max, range_min_list, range_max_list, is_cal_max)
        """局部进化算法工厂"""
        self.use_local = False
        """使用局部代理的flag"""
        self.range_min_list_local = range_min_list
        """设计空间下界"""
        self.range_max_list_local = range_max_list
        """设计空间上界"""
        self.global_p_init = 0.8
        """全局搜索的初始概率"""
        self.selection_strategy = MixedSelectedStrategy(
            ["Global", "Local"],
            [self.global_p_init, 1.0 - self.global_p_init],
            [False, True],
            changing_type="FirstStageDecreasesNonlinearly",
            iter_max=(fit_max - init_size)
        ) if selection_strategy is None else selection_strategy
        """选择策略"""

    def create_surrogate(self):
        """
        构建代理模型
        """
        # 获取全部个体样本
        X, y = [], []
        for unit in self.unit_list:
            X.append(unit.position)
            y.append(unit.fitness_true)
        X, y = np.array(X), np.array(y)

        # 根据use_local构建训练样本
        if self.use_local:
            # 计算每个个体到全局最优的距离，取最近的pop_size_local个个体作为局部代理模型的训练样本
            dist = np.linalg.norm(X - self.position_best, axis=1)
            idx = np.argsort(dist)
            if self.pop_size_local > 1:
                idx = idx[:self.pop_size_local]
            else:
                idx = idx[:min(int(len(idx) * self.pop_size_local), len(idx))]
            self.pop_size_local = len(idx)
            self.pop_local = []
            for i in idx:
                unit = SAEA_Unit()
                unit.position = X[i]
                unit.fitness = y[i]
                self.pop_local.append(unit)
            X, y = X[idx], y[idx]

            # 更新边界和局部代理模型
            self.range_min_list_local = np.min(X, axis=0)
            self.range_max_list_local = np.max(X, axis=0)
            self.surr = self.surr_factory_local.create(X, y)
        else:
            self.surr = self.surr_factory_global.create(X, y)
        self.surr_history.append(self.surr)

    def optimize(self):
        """
        优化代理模型，更新unit_new
        """
        # 获取优化器（进化算法）进行优化
        optimizer = self.get_optimizer()
        optimizer.run()

        # 更新unit_new
        unit = self.get_best_unit(optimizer)
        self.unit_new = [unit]
        print("unit_new position: ", unit.position, " fitness: ", unit.fitness)
        if self.draw_gif_flag:
            optimizer.draw2_gif(10, True, f"SAEA_iter{self.iter_num}")

    def get_optimizer(self):
        """
        获取优化器（进化算法）
        """
        if self.use_local:
            ea = self.ea_factory_local.get_alg_with_surr(self.surr, self.range_min_list_local, self.range_max_list_local)
        else:
            ea = self.ea_factory_global.get_alg_with_surr(self.surr)
        ea.surr = self.surr
        if self.use_local:
            ea.size = min(len(self.pop_local), self.pop_size)
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
        """
        使用选择策略更新use_local
        """
        # 获取本次和上次最佳值
        value_best = self.value_best_history[-1]
        if len(self.value_best_history) < 2:
            value_best_prev = value_best
        else:
            value_best_prev = self.value_best_history[-2]
        
        phase = self.selection_strategy.select(flag=(value_best > value_best_prev), iter=self.iter_num)
        self.use_local = phase == "Local"

        print("[Surrogate] ", "LOCAL" if self.use_local else "GLOBAL")
        
        
        # if self.use_local:
        #     # 如果局部代理模型的最优值大于上一代的最优值，则继续使用局部代理模型
        #     if value_best > value_best_prev:
        #         self.use_local = True
        #     else:
        #         self.use_local = False
        # else:
        #     # 如果全局代理模型的最优值大于上一代的最优值，则继续使用全局代理模型
        #     if value_best > value_best_prev:
        #         self.use_local = False
        #     else:
        #         self.use_local = True
    
    def toStr(self):
        # 打印选择策略的类名称
        select_strategy_class = self.selection_strategy.__class__.__name__
        # return super().toStr() + f" HSAEA_Base: use_local={self.use_local}\n\
        #     selection_strategy={select_strategy_class}\n\
        #     surr_type_global={self.surr_type_global}\n\
        #     surr_type_local={self.surr_type_local}\n\
        #     ea_type_global={self.ea_type_global}\n\
        #     ea_type_local={self.ea_type_local}\n\
        #     pop_size_local={self.pop_size_local}\n\
        #     range_min_list_local={self.range_min_list_local}\n\
        #     range_max_list_local={self.range_max_list_local}\n\
        #     global_p_init={self.global_p_init}\n\
        #     selection_strategy={select_strategy_class}\n"
        str = super().toStr()
        str += f"HSAEA_Base: use_local={self.use_local}\n"
        str += f"surr_type_global={self.surr_type_global}\n"
        str += f"surr_type_local={self.surr_type_local}\n"
        str += f"ea_type_global={self.ea_type_global}\n"
        str += f"ea_type_local={self.ea_type_local}\n"
        str += f"pop_size_local={self.pop_size_local}\n"
        str += f"range_min_list_local={self.range_min_list_local}\n"
        str += f"range_max_list_local={self.range_max_list_local}\n"
        str += f"global_p_init={self.global_p_init}\n"
        str += f"selection_strategy={select_strategy_class}\n"
        return str
