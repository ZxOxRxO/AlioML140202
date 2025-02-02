
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)


def f(x1, x2):
    return 2*x2 + 5*x1**3 + 123*x2**5


X1 = np.random.uniform(0, 10, 600)
X2 = np.random.uniform(0, 10, 600)

y_true = f(X1, X2)
noise = np.random.normal(0, 10, 600)
y_noisy = y_true + noise


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1, X2, y_noisy, c=y_noisy, cmap='viridis')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y_noisy')


data = pd.DataFrame({'X1': X1,  "X2" : X2, 'y_noisy': y_noisy})



data.to_excel('2feactherGDProblem.xlsx', index=False)
plt.show()

