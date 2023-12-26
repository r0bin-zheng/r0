# 根据日期时间生成id
import os
import time
import datetime
import imageio
import numpy as np

# 根据实验id创建文件夹，并返回文件夹绝对路径和id
def create_exp_folder(keyword, path=None):
    exp_id = get_exp_id_plus(keyword)
    target_dir = path if path is not None else os.path.dirname(os.path.abspath(__file__))
    exp_path = target_dir + "//" + exp_id
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
        print("mkdir: ", exp_path)
    return exp_id, exp_path

# 使用时间戳生成实验id
def get_exp_id():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

# 使用时间戳、关键字生成实验id
def get_exp_id_plus(keyword):
    return time.strftime("%Y%m%d-%H%M%S", time.localtime()) + "_" + keyword

# 多个图像合成一个gif
def create_gif(path, gif_name, file_prefix, start, end, fps=1):
    gif_file_name = path + "/" + gif_name + ".gif"
    frames = []
    for i in range(start, end):
        filename = path + "/" + file_prefix + str(i) + '.png'
        frames.append(imageio.imread(filename))

    # 保存为gif格式的图
    imageio.mimsave(gif_file_name, frames, 'GIF', duration=200)

# 保存单个numpy数组
def save_numpy_array(arr, name, path=None):
    np.save(((path+"/") if path is not None else "")+name+".npy", arr)

# 读取单个numpy数组
def load_numpy_array(name, path=None):
    return np.load(((path+"/") if path is not None else "")+name+".npy")

# 保存字典
def save_dict(dict, name, path=None):
    np.save(((path+"/") if path is not None else "")+name+".npy", dict)

# 读取字典
def load_dict(name, path=None):
    return np.load(((path+"/") if path is not None else "")+name+".npy", allow_pickle=True).item()

