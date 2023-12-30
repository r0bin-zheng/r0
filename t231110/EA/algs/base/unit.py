"""个体基类"""

import numpy as np

class Unit:
    def __init__(self):
        self.position = None
        self.fitness = None
        self.position_history_list = np.array([])
    
    def save(self):
        if self.position_history_list.size == 0:
            self.position_history_list = np.append(self.position_history_list, self.position)
        else:
            self.position_history_list = np.vstack((self.position_history_list, self.position))
    
    def __str__(self):
        return f"position={self.position}, fitness={self.fitness}"
