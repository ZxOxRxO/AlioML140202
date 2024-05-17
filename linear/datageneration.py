import numpy as np
import pandas as pd

np.random.seed(0)

def f(x1 , x2  ):
    return  5*x1 + 459



X1 = np.random.uniform(0, 10, 100)
X2 = np.random.uniform(0, 10, 100)
y_true = f(X1, X2)
noise = np.random.normal(0, 10, 100)
y_noisy = y_true + noise


noise = np.random.standard_t(10, 100)
y_noisy = y_true + noise



data = pd.DataFrame({'X': X1, 'y_noisy': y_noisy})


data.to_excel('./linear/LinearData.xlsx', index=False)