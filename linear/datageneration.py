import numpy as np
import pandas as pd

np.random.seed(0)  # for reproducibility

# Define the function f(x)
def f(x1 , x2 , x3 ):
    return  3*(x1**4) + 19.4*(x2**6) +  2*x3 + 5*x1 + 10

# Generate noisy data with Gaussian noise for the function f(x)

X1 = np.random.uniform(0, 10, 100)  # Feature 1 values
X2 = np.random.uniform(0, 10, 100)  # Feature 2 values
X3 = np.random.uniform(0, 10, 100)  # Feature 2 values
y_true = f(X1, X2)
noise = np.random.normal(0, 10, 100)  # Gaussian noise
y_noisy = y_true + noise
# noise = np.random.normal(0, 20000, 100)  # using Gaussian noise with mean 0 and standard deviation 10
# y_noisy = y_true + noise

noise = np.random.standard_t(10, 200)  # using Gaussian noise with mean 0 and variance 10
y_noisy = y_true + noise


# Create a DataFrame to store the noisy data
data = pd.DataFrame({'X': X, 'y_noisy': y_noisy})

# Save the data to an Excel file
data.to_excel('generatedData.xlsx', index=False)