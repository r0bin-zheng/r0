import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import time
import random
import string
import datetime

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


def get_time_str():
    """获取当前时间字符串，格式：YYYY-MM-DD HH:MM:SS"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def roulette_selection(weights):
    """
    Perform a single roulette wheel selection. 

    轮盘赌选择
    """
    cumulative_weights = [sum(weights[:i+1]) for i in range(len(weights))]
    total = cumulative_weights[-1]
    pick = random.uniform(0, total)
    for i, weight in enumerate(cumulative_weights):
        if pick <= weight:
            return i

def avoid_duplicates_roulette(items, weights, n=1):
    """ 
    Perform roulette selection n times, 
    avoiding duplicates by setting the weight to 0.

    轮盘赌选择，避免重复

    eg.
    >>> items = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    >>> weights = [10, 20, 30, 40, 50]
    >>> selected = avoid_duplicates_roulette(items, weights, 3)
    >>> selected
    ['elderberry', 'date', 'cherry']
    """
    selected_items = []
    for _ in range(n):
        index = roulette_selection(weights)
        selected_items.append(items[index])
        weights[index] = 0  # Set the weight of the selected item to 0
    return selected_items
