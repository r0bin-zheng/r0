import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

from SAEA.algs.base.SAEA_Base import SAEA_Base
from SAEA.algs.fwa_surr.FWA_SAEA_Unit import FWA_SAEA_Unit

class FWA_SAEA(SAEA_Base):
    def __init__(self, dim, init_size, pop_size, surr_type, ea_type, fit_max,
                 iter_max, range_min_list, range_max_list, is_cal_max, surr_setting):
        ea_type = "FWA_Surr_Impl2"

        super().__init__(dim, init_size, pop_size, surr_type, ea_type, fit_max,
                         iter_max, range_min_list, range_max_list, is_cal_max, surr_setting)

        self.name = 'FWA_SAEA'
        self.m = 40
        self.a = 0.2
        self.b = 0.8
        self.spec_num = 5
        self.amplitude_max = range_max_list[0] * 0.6
        self.all_list = []

    def optimize(self):
        ea = self.ea_factory.get_alg_with_surr(self.surr)
        # selected_unit = self.select()
        # for unit in selected_unit:
        #     unit.uncertainty = None
        ea.surr = self.surr
        ea.size = len(self.unit_list)
        ea.iter_num_main = self.iter_num
        ea.iter_max_main = self.fit_max - self.init_size
        ea.silence = True
        ea.set_unit_list(self.unit_list)
        ea.run()
        unit = FWA_SAEA_Unit()
        unit.position = ea.position_best
        unit.fitness = ea.value_best
        self.unit_new = [unit]
        print("unit_new position: ", unit.position, " fitness: ", unit.fitness)
        ea.draw2_gif(10, True, f"FWA_SAEA_iter{self.iter_num}")
