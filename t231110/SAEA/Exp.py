"""实验类"""
import time
import random
import string
import os

class Exp:
    def __init__(self, problem, alg):
        self.id = None
        self.problem = None
        self.alg = None
        self.start_time = None
        self.end_time = None
        self.save_path = None

    def init(self):
        self.id = self.get_id()
        self.save_path = f'./exp/{self.id}'

    def solve(self):
        os.mkdir(self.save_path)
        os.chdir(self.save_path)

        self.start_time = time.time()
        # ...
        self.end_time = time.time()
        
        pass

    def get_id():
        """根据日期时间生成id"""

        # 格式：YYYYMMDDHHMMSS_XXXXX
        # 其中 XXXXX 为随机字符串
        date_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

        # 随机字符串
        random_str = ''.join(random.sample(string.ascii_letters + string.digits, 5))

        # 拼接
        id = f'{date_time}_{random_str}'
        return id