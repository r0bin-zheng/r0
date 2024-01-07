import numpy as np
import matplotlib.pyplot as plt

from smt.surrogate_models import KRG

xt = np.array([0.0, 1.0, 1.5, 2.0, 3.0, 4.0])
yt = np.array([0.0, 1.0, 25.0, 100.5, 0.9, 1.0])

sm = KRG(theta0=[1e-2])
sm.set_training_values(xt, yt)
sm.train()

num = 100
x = np.linspace(0.0, 4.0, num)
# x中添加0.0 1.0 2.0 3.0 4.0，且确保按序排列
x = np.sort(np.hstack((x, xt)))
# 去重
x = np.unique(x)
num = len(x)
y = sm.predict_values(x)
# estimated variance
s2 = sm.predict_variances(x)
for i in range(num):
    print(x[i], y[i], s2[i])
# derivative according to the first variable
dydx = sm.predict_derivatives(xt, 0)
fig, axs = plt.subplots(1)

# add a plot with variance
axs.plot(xt, yt, "o")
axs.plot(x, y)
axs.fill_between(
    np.ravel(x),
    np.ravel(y - 3 * np.sqrt(s2)),
    np.ravel(y + 3 * np.sqrt(s2)),
    color="lightgrey",
)
axs.set_xlabel("x")
axs.set_ylabel("y")
axs.legend(
    ["Training data", "Prediction", "Confidence Interval 99%"],
    loc="lower right",
)

plt.show()