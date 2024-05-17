import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

def f(x1 , x2  ):
    return  5*x1 + 459



X1 = np.random.uniform(0, 10, 50)
X2 = np.random.uniform(0, 10, 50)
y_true = f(X1, X2)
noise = np.random.normal(0,2, 50)
y_noisy = y_true + noise


noise = np.random.standard_t(10, 50)
y_noisy = y_true + noise



data = pd.DataFrame({'X': X1, 'y_noisy': y_noisy})


data.to_excel('./LinearData.xlsx', index=False)





df = pd.read_excel('LinearData.xlsx')
X = df[['X']]
y = df['y_noisy']

# plt.plot(df['X'], df['y_noisy'])
fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(X, y, cmap='viridis')

plt.xlabel('X')
plt.ylabel('y_noisy')
plt.title('X1 vs y_noisy')
plt.show()

