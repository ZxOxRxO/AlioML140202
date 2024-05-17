# import numpy as np
# from sklearn.preprocessing import StandardScaler
#
# # Define the multi-feature function f(x1, x2)
# def f(x1, x2):
#     return 5*x1 + 3*x2 + 7
#
# # Generate noisy data for the multi-feature function
# np.random.seed(0)
# X1 = np.random.uniform(0, 10, 100)  # Feature 1 values
# X2 = np.random.uniform(0, 10, 100)  # Feature 2 values
# y_true = f(X1, X2)
# noise = np.random.normal(0, 10, 100)  # Gaussian noise
# y_noisy = y_true + noise
#
# # Normalize the data using StandardScaler
# scaler = StandardScaler()
# X = np.column_stack((X1, X2))  # Combine feature columns
# X_normalized = scaler.fit_transform(X)
#
# # Apply Gradient Descent (Sample code)
# # Gradient Descent code would depend on the specific implementation and optimization problem
# # Here's a simple illustration applying GD to the function f(x1, x2)
# # For simplicity, we use the formula theta = theta - alpha * (grad/mse)
# # where alpha is the learning rate and grad/mse is the gradient of the mean squared error
#
# alpha = 0.01
# theta = np.zeros(3)  # Coefficients for intercept, x1, x2
#
# for i in range(1000):
#     y_pred = np.dot(np.column_stack((np.ones(X_normalized.shape[0]), X_normalized)), theta)
#     error = y_pred - y_noisy
#     grad_mse = np.dot(X_normalized.T, error) / X_normalized.shape[0 ]
#
#     # Update the coefficients
#     theta = theta - alpha * grad_mse
#
#     # Show the response
#     print("Coefficients after Gradient Descent:")
#     for i, coef in enumerate(theta):
#         print(f"Theta{i}: {coef}")





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)  # for reproducibility

# Define the function f(x)
def f(x1, x2):
    return 3*(x1**4) + 19.4*(x2**6) + 2*x2 + 5*x2**3 + 5*x1 + 10

# Generate noisy data with Gaussian noise for the function f(x)
X1 = np.random.uniform(0, 10, 1000)  # Feature 1 values
X2 = np.random.uniform(0, 10, 1000)  # Feature 2 values

y_true = f(X1, X2)
noise = np.random.normal(0, 2000000, 1000)  # Gaussian noise
y_noisy = y_true + noise

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1, X2, y_noisy, c=y_noisy, cmap='viridis')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y_noisy')



data = pd.DataFrame({'X1': X1,  "X2" : X2, 'y_noisy': y_noisy})

# Save the data to an Excel file
data.to_excel('2feactherGDProblem.xlsx', index=False)
plt.show()

