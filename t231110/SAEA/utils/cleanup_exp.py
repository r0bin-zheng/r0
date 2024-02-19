"""清理空文件夹"""


import os
import shutil

def remove_empty_folders(path):
    # 遍历指定目录
    for root, dirs, files in os.walk(path, topdown=False):
        # 对于每一个目录，检查是否为空
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            # 如果目录为空，则删除
            if not os.listdir(dir_path):
                shutil.rmtree(dir_path)
                print(f"删除空目录：{dir_path}")

# 设置起始路径
start_path = "/home/robin/projects/t231101/t231110/EA/exp/"
remove_empty_folders(start_path)
