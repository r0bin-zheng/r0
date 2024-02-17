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
from SAEA.algs.selection_strategy.ss_mapping import get_selection_strategy

class HSAEA_Base(SAEA_Base):
    """
    层次代理模型辅助进化算法基类
    
    参数
    ----------
    surr_types: list(二维数组)

    """
    def __init__(self, dim, init_size, pop_size, surr_types, ea_types, fit_max, iter_max, 
                 range_min_list, range_max_list, is_cal_max, surr_setting=None,
                 selection_strategy_str="MixedNonlinearly", pop_size_local=30):
        ea_type = ea_types[0]
        surr_type = surr_types[0]
        super().__init__(dim, init_size, pop_size, surr_type, ea_type, fit_max, iter_max, 
                         range_min_list, range_max_list, is_cal_max, surr_setting)
        
        # 算法信息
        self.name = 'HSAEA_Base'

        # 参数
        self.selection_strategy_str = selection_strategy_str
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
        self.ea_type_global = ea_types[0]
        """全局进化算法设置"""
        self.ea_type_local = ea_types[1]
        """局部进化算法设置"""
        self.pop_size_local = pop_size_local
        """局部代理模型训练样本数量，当数字大于1时为数量，当数字小于等于1时为比例"""

        # 中间值
        self.pop_local = []
        """局部代理模型的训练样本"""
        self.surr_factory = None
        """当前代理模型工厂"""
        self.surr_factory_global = None
        """全局代理模型工厂"""
        self.surr_factory_local = None
        """局部代理模型工厂"""
        self.ea_factory_global = None
        """全局进化算法工厂"""
        self.ea_factory_local = None
        """局部进化算法工厂"""
        self.use_local = False
        """使用局部代理的flag"""
        self.range_min_list_local = range_min_list
        """设计空间下界"""
        self.range_max_list_local = range_max_list
        """设计空间上界"""
        self.selection_strategy = None
        """选择策略"""
        self.global_ea_info = ""
        """全局EA信息"""
        self.local_ea_info = ""
        """局部EA信息"""
        self.ea_info_flag = [True, True]

    def init(self):
        """初始化"""
        super().init()
        self.init_selection_strategy()

    def init_selection_strategy(self):
        self.selection_strategy = get_selection_strategy(self.selection_strategy_str)

    def init_surr_factory(self):
        super().init_surr_factory()
        self.surr_factory_global = self.surr_factory
        self.surr_factory_local = SurrogateModelFactory(self.surr_type_local, self.surr_setting)
        self.surr_factory = self.surr_factory_global

    def init_ea_factory(self):
        global_args = {
            "dim": self.dim,
            "size": self.pop_size,
            "iter_max": self.iter_max,
            "range_min_list": self.range_min_list,
            "range_max_list": self.range_max_list,
            "is_cal_max": True,
            "silent": True,
        }
        local_args = {
            "dim": self.dim,
            "size": self.pop_size_local,
            "iter_max": self.iter_max,
            "is_cal_max": True,
            "silent": True,
        }
        self.ea_factory_global = Ea_Factory(self.ea_type_global, static_args=global_args)
        self.ea_factory_local = Ea_Factory(self.ea_type_local, static_args=local_args)
        self.ea_factory = self.ea_factory_global
    
    def create_surrogate(self):
        """构建代理模型"""
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
        self.surr = self.surr_factory.create(X, y)
        self.surr_history.append(self.surr)

    def optimize(self):
        """
        优化代理模型，更新unit_new
        """
        # 获取优化器（进化算法）进行优化
        optimizer, pop = self.get_optimizer()
        optimizer.run(unit_list=pop)

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
        ea_args = self.get_ea_dynamic_args()
        ea = self.ea_factory.create_with_surr(self.surr, dynamic_args=ea_args)
        if self.use_local == False:
            if self.ea_info_flag[0]:
                self.global_ea_info = ea.toStr()
                self.ea_info_flag[0] = False
        else:
            if self.ea_info_flag[1]:
                self.local_ea_info = ea.toStr()
                self.ea_info_flag[1] = False

        ea_pop = self.init_ea_population(ea)
        return ea, ea_pop
    
    def get_ea_dynamic_args(self):
        if self.use_local:
            return {
                "size": min(len(self.pop_local), self.pop_size),
                "range_min_list": self.range_min_list_local,
                "range_max_list": self.range_max_list_local,
            }
        else:
            return {}

    # def init_ea_population(self, pop, ea):
    #     """初始化进化算法种群"""
    #     for i in range(len(pop)):
    #         if i > ea.size:
    #             break
    #         ind = pop[i]
    #         unit = ea.unit_class()
        #     unit.position = ind.position.copy()
        #     unit.fitness = ind.fitness
        #     ea.unit_list.append(unit)

        # # 预测每个个体的值和方差
        # pos_list = [unit.position for unit in ea.unit_list]
        # if self.surr.predict_value_variance and ea.use_variances:
        #     # 如果代理模型支持预测方差且进化算法使用方差，则预测每个个体的值和方差
        #     value_list, var_list = self.surr.predict_value_variance(pos_list)
        #     for i in range(len(ea.unit_list)):
        #         ea.unit_list[i].fitness = value_list[i]
        #         ea.unit_list[i].uncertainty = var_list[i]
        # else:
        #     # 否则只预测每个个体的值
        #     value_list = self.surr.predict(pos_list)
        #     for i in range(len(ea.unit_list)):
        #         ea.unit_list[i].fitness = value_list[i]
    
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

        if self.use_local:
            self.surr_factory = self.surr_factory_local
            self.ea_factory = self.ea_factory_local
        else:
            self.surr_factory = self.surr_factory_global
            self.ea_factory = self.ea_factory_global
    
    def toStr(self):
        # 打印选择策略的类名称
        select_strategy_class = self.selection_strategy.__class__.__name__
        str = super().toStr()
        str += f"use_local: {self.use_local}\n"
        str += f"surr_type_global: {self.surr_type_global}\n"
        str += f"surr_type_local: {self.surr_type_local}\n"
        str += f"ea_type_global: {self.ea_type_global}\n"
        str += f"ea_type_local: {self.ea_type_local}\n"
        str += f"pop_size_local: {self.pop_size_local}\n"
        str += f"range_min_list_local: {self.range_min_list_local}\n"
        str += f"range_max_list_local: {self.range_max_list_local}\n"
        str += f"selection_strategy_str: {self.selection_strategy_str}\n"
        str += f"selection_strategy: {select_strategy_class}\n"
        str += "\n"

        str += "global_ea_info\n"
        str += self.global_ea_info
        str += "\n"

        str += "local_ea_info\n"
        str += self.local_ea_info

        return str
