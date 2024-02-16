"""烟花算法，结合代理模型改进，基类"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import numpy as np
from EA.algs.algorithm_fireworks.FWA_Base import FWA_Base
from SAEA.algs.fwa_surr.wt_strategy import WT_DICT


class FWA_Surr(FWA_Base):
    """
    结合代理模型(surrogate model)的烟花算法基类
    
    eg.
    ```
    alg = FWA_Surr(dim=2, size=50, iter_max=50, range_min_list=[-100, -100], range_max_list=[100, 100], is_cal_max=False, iter_max_main=50)
    alg.surr = surr
    alg.w_strategy = 2
    ```
    """
    def __init__(self, dim, size, iter_max, range_min_list, range_max_list, 
                 w_strategy_str="NegativeCorrelation", alpha=0.6, beta=0.5, gamma=0.3, delta=0.5):
        super().__init__(dim, size, iter_max, range_min_list, range_max_list)
        self.name = 'FWA_Surr'
        self.surr = None
        """代理模型"""

        # 算法参数
        self.iter_num = 0
        """循环迭代次数"""
        self.iter_num_main = None
        """主循环迭代次数"""
        self.iter_max_main = None
        """主循环迭代次数最大值"""
        self.w_strategy_str = w_strategy_str
        """权重因子计算策略字符串"""
        self.w_strategy = None
        """
        权重因子计算策略

        类型：int
        - 1: 随迭代次数指数衰减
        - 2: 随迭代次数线性衰减
        - 3: 和皮尔逊相关系数呈负相关
        - 4: 固定不确定性的权重
        """
        self.alpha = alpha
        """权重因子计算策略1的参数，控制基于指数下降的不确定性的权重"""
        self.beta = beta
        """权重因子计算策略2的参数，控制基于线性下降的不确定性的权重"""
        self.gamma = gamma
        """
        权重因子计算策略3的参数，控制基于皮尔逊相关系数的不确定性的权重
        最大值0.5，最小值0.0
        """
        self.delta = delta
        """权重因子计算策略4的参数，固定不确定性的权重"""

        # 中间值，非参数
        self.WT = None
        """权重因子"""
        self.pop_fitness_max = None
        """该变量用于当前种群适应度最大值"""
        self.pop_fitness_min = None
        """该变量用于当前种群适应度最小值"""
        self.pop_uncertainty_max = None
        """该变量用于当前种群不确定性最大值"""
        self.pop_uncertainty_min = None
        """该变量用于当前种群不确定性最小值"""

        self.use_variances = True
        """用于指示该算法是否使用代理模型的不确定性"""

    def init(self):
        """初始化种群，由于SAEA框架中的一部分，初始化种群工作将在主算法完成"""
        self.set_w_strategy()
        self.position_best = np.zeros(self.dim)
        self.value_best_history = []
        self.position_best_history = []
        self.value_best = -np.finfo(np.float64).max
    
    def set_w_strategy(self, w_strategy_str=None):
        """初始化不确定性权重计算策略"""
        key = w_strategy_str if w_strategy_str is not None else self.w_strategy_str
        self.w_strategy_str = key
        wt_setting = WT_DICT[key]
        self.w_strategy = wt_setting["w_strategy"]
        if self.w_strategy == 1:
            self.alpha = wt_setting["alpha"]
        elif self.w_strategy == 2:
            self.beta = wt_setting["beta"]
        elif self.w_strategy == 3:
            self.gamma = wt_setting["gamma"]
        elif self.w_strategy == 4:
            self.delta = wt_setting["delta"]

    def update(self, iter):
        self.iter_num = iter

        # 更新烟花信息
        self.update_firework(iter)
        self.statistics_unit_list()
        
        # 生成火星
        self.denominator = None
        # print("生成火星")
        for i in range(self.size):
            self.obtain_spark(i)
        
        # 生成特殊火星
        # print("生成特殊火星")
        for num in range(self.spec_num):
            rand_id = np.random.randint(0, self.size)
            self.obtain_spec_spark(rand_id)

        # 选择火星
        # print("选择火星")
        self.select_sparks()
        # print("选择火星结束")
    
    def update_firework(self, iter):
        """更新烟花信息"""
        # wt = self.W(iter)
        # fitness_list = [unit.fitness for unit in self.unit_list]
        # uncertainty_list = [self.get_unit_uncertainty(unit) for unit in self.unit_list]
        # # 标准化适应度和不确定性
        # fitness_range = max(fitness_list) - min(fitness_list)
        # uncertainty_range = max(uncertainty_list) - min(uncertainty_list)
        # normalized_fitness_list = [(unit.fitness - min(fitness_list)) / fitness_range if fitness_range > 0 else 0 for unit in self.unit_list]
        # normalized_uncertainty_list = [(self.get_unit_uncertainty(unit) - min(uncertainty_list)) / uncertainty_range if uncertainty_range > 0 else 0 for unit in self.unit_list]

        if iter == 1:
            for i in range(self.size):
                if self.unit_list[i].fitness > self.value_best:
                    self.value_best = self.unit_list[i].fitness
                    self.position_best = self.unit_list[i].position
                self.unit_list[i].save()
        else:
            if iter == 2:
                self.value_best = self.unit_list[0].value
            for i in range(self.size):
                if self.unit_list[i].value > self.value_best:
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
    
    def get_spark_num2(self, id):
        """
        计算火星数量，考虑不确定性。
        <公式>
        这是烟花算法中，求正常火星数量的公式，适应度越高，数量越多
        在这个公式中，a和b是预先设定的参数，用于控制火星的数量。这两个参数的设定通常基于启发式或问题的特定需求。a和b的值都小于 1（a<b<1），这确保了火星数量的调整保持在合理的范围内。
        a 通常是一个较小的数，用于确保至少有一定数量的火星。
        b 则是一个较大的数，用于限制火星数量的上限，以避免生成过多的火星
        """

        T = self.iter_num_main
        f_max = max(unit.fitness for unit in self.unit_list)
        f_i = self.unit_list[id].fitness
        U_i = self.get_unit_uncertainty(self.unit_list[id])

        numerator = (f_max - f_i) * self.W(T) * U_i
        denominator = sum((f_max - unit.fitness) * self.W(T) * self.get_unit_uncertainty(unit) for unit in self.unit_list)

        num = round(self.m * (numerator / (denominator + np.finfo(float).eps)))
        num = max(round(self.a * self.m), min(num, round(self.b * self.m)))
        return num
    
    def get_spark_num(self, id):
        """计算火星数量，考虑不确定性，标准化适配度和不确定性"""
        WT = self.W(self.iter_num_main)
        f_i = self.unit_list[id].fitness
        U_i = self.get_unit_uncertainty(self.unit_list[id])
        
        # 最小-最大标准化适应度和不确定性
        f_range = self.pop_fitness_max - self.pop_fitness_min
        U_range = self.pop_uncertainty_max - self.pop_uncertainty_min

        normalized_f_i = (f_i - self.pop_fitness_min) / f_range if f_range > 0 else 0
        normalized_U_i = (U_i - self.pop_uncertainty_min) / U_range if U_range > 0 else 0
        
        # 计算分子和分母
        numerator = normalized_f_i + WT * normalized_U_i
        if self.denominator == None:
            self.denominator = sum((unit.fitness - self.pop_fitness_min) / f_range if f_range > 0 else 0 + WT * (self.get_unit_uncertainty(unit) - self.pop_uncertainty_min) / U_range if U_range > 0 else 0 for unit in self.unit_list)

        # 计算并返回火星数量
        num = round(self.m * (numerator / (self.denominator + np.finfo(float).eps)))
        num = max(round(self.a * self.m), min(num, round(self.b * self.m)))
        return num

    def select_sparks2(self):
        """选择火星，考虑不确定性，原始版本"""
        T = self.iter_num_main
        self.all_list.extend(self.unit_list)
        f_max = max(unit.fitness for unit in self.all_list)

        R = np.array([unit.fitness for unit in self.all_list])
        U = np.array([self.get_unit_uncertainty(unit) for unit in self.all_list])

        # 对R中的值进行调整，以保证都为正数
        min_R = np.min(R)
        adjusted_R = R - min_R

        rates = (adjusted_R * self.W(T) * U) / np.sum(adjusted_R * self.W(T) * U)

        for i in range(self.size):
            chosen_id = np.random.choice(range(len(self.all_list)), p=rates)
            self.unit_list[i].position = self.all_list[chosen_id].position
            self.unit_list[i].fitness = self.all_list[chosen_id].fitness

        self.all_list = []

    def select_sparks(self):
        """选择火星，考虑不确定性，标准化适配度和不确定性"""
        self.all_list.extend(self.unit_list)
        WT = self.W(self.iter_num_main)
        R_max = max(unit.fitness for unit in self.all_list)
        R_min = min(unit.fitness for unit in self.all_list)
        R_range = R_max - R_min
        U_max = max(self.get_unit_uncertainty(unit) for unit in self.all_list)
        U_min = min(self.get_unit_uncertainty(unit) for unit in self.all_list)
        U_range = U_max - U_min

        # 计算每个个体的选择概率
        probabilities = []
        for unit in self.all_list:
            R_norm = ((unit.fitness - R_min) / R_range) if R_range > 0 else 0
            U_norm = ((self.get_unit_uncertainty(unit) - U_min) / U_range) if U_range > 0 else 0
            numerator = R_norm + WT * U_norm
            unit.value = numerator # 存储综合值
            probabilities.append(numerator)

        denominator = sum(probabilities) + np.finfo(float).eps  # 为了防止除以零
        probabilities = [p / denominator for p in probabilities]

        # 根据计算出的概率选择火星
        for i in range(self.size):
            chosen_id = np.random.choice(range(len(self.all_list)), p=probabilities)
            self.unit_list[i].position = self.all_list[chosen_id].position
            self.unit_list[i].fitness = self.all_list[chosen_id].fitness
            self.unit_list[i].value = self.all_list[chosen_id].value
            self.unit_list[i].uncertainty = self.get_unit_uncertainty(self.unit_list[i])

        self.all_list = []

    def get_uncertainty(self, X):
        """获取给定解的不确定性"""
        std = self.surr.predict_variances_one(X)
        return std
    
    def get_unit_uncertainty(self, unit):
        """获取给定单元的不确定性"""
        if unit.uncertainty is not None:
            return unit.uncertainty
        unit.uncertainty = self.get_uncertainty(unit.position)
        return unit.uncertainty

    def W(self, T):
        """计算权重因子"""
        if self.WT is None:
            if self.w_strategy == 1:
                """计算权重因子，随迭代次数指数衰减"""
                self.WT = 1 / (1 + self.alpha * T)

            elif self.w_strategy == 2:
                """计算权重因子，随迭代次数线性衰减"""
                self.WT = max(0, self.beta * (1 - T / self.iter_max_main))

            elif self.w_strategy == 3:
                """计算权重因子，和皮尔逊相关系数呈负相关"""
                n_samples = 20 # 随机采样个数
                X = np.random.uniform(self.range_min_list, self.range_max_list, (n_samples, self.dim))
                y = self.surr.predict_each_model(X)
                corr_matrix = np.corrcoef(y)  # 计算相关系数矩阵
                n = y.shape[0]
                k = 1  # 从对角线上方一行开始
                upper_triangle_indices = np.triu_indices(n=n, k=k) # 获取上三角的索引
                unique_corr_values = corr_matrix[upper_triangle_indices]
                avg_unique_corr = np.mean(unique_corr_values) # 计算代理模型集合的平均相关系数
                self.WT = self.gamma * (1 - avg_unique_corr)
                # print("WT: ", self.WT, " avg_unique_corr: ", avg_unique_corr)
            
            elif self.w_strategy == 4:
                """计算权重因子，固定不确定性的权重"""
                self.WT = self.delta

            # print("WT: ", self.WT)
        return self.WT
        
    def statistics_unit_list(self):
        """统计unit_list的信息"""
        fitness_max = np.finfo(float).min
        fitness_min = np.finfo(float).max
        uncertainty_max = np.finfo(float).min
        uncertainty_min = np.finfo(float).max
        for unit in self.unit_list:
            if unit.fitness > fitness_max:
                fitness_max = unit.fitness
            if unit.fitness < fitness_min:
                fitness_min = unit.fitness
            if self.get_unit_uncertainty(unit) > uncertainty_max:
                uncertainty_max = self.get_unit_uncertainty(unit)
            if self.get_unit_uncertainty(unit) < uncertainty_min:
                uncertainty_min = self.get_unit_uncertainty(unit)
        self.pop_fitness_max = fitness_max
        self.pop_fitness_min = fitness_min
        self.pop_uncertainty_max = uncertainty_max
        self.pop_uncertainty_min = uncertainty_min

    def toStr(self):
        str = super().toStr()
        str += f"iter_num: {self.iter_num}\n"
        str += f"iter_num_main: {self.iter_num_main}\n"
        str += f"iter_max_main: {self.iter_max_main}\n"
        str += f"w_strategy_str: {self.w_strategy_str}\n"
        str += f"w_strategy: {self.w_strategy}\n"
        str += f"alpha: {self.alpha}\n"
        str += f"beta: {self.beta}\n"
        str += f"gamma: {self.gamma}\n"
        str += f"delta: {self.delta}\n"
        str += f"WT: {self.WT}\n"
        # str += f"pop_fitness_max: {self.pop_fitness_max}\n"
        # str += f"pop_fitness_min: {self.pop_fitness_min}\n"
        # str += f"pop_uncertainty_max: {self.pop_uncertainty_max}\n"
        # str += f"pop_uncertainty_min: {self.pop_uncertainty_min}\n"
        return str
