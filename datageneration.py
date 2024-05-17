import numpy as np
import pandas as pd

np.random.seed(0)  # for reproducibility

# Define the function f(x)
def f(x):
    return  3*(x**4) + 19.4*(x**6) +  2*x + 5*x + 10

# Generate noisy data with Gaussian noise for the function f(x)
X = np.linspace(-10, 13 , 200)
y_true = f(X)
# noise = np.random.normal(0, 20000, 100)  # using Gaussian noise with mean 0 and standard deviation 10
# y_noisy = y_true + noise

noise = np.random.standard_t(10, 200)  # using Gaussian noise with mean 0 and variance 10
y_noisy = y_true + noise


# Create a DataFrame to store the noisy data
data = pd.DataFrame({'X': X, 'y_noisy': y_noisy})

# Save the data to an Excel file
data.to_excel('generatedData.xlsx', index=False)