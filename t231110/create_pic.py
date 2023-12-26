# 绘制图像
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 参数设置
save_path = "./pic"

# 导入npy数据
X = np.load(os.path.join(save_path, "X.npy"))
y = np.load(os.path.join(save_path, "y.npy"))

# 打印数据
print("X: ", X)
print("y: ", y)

# 绘制图像
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')
plt.show()


