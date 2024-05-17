import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)  # for reproducibility

# Define the function f(x)
def f(x1, x2, x3):
    return 3*(x1**4) + 19.4*(x2**6) + 2*x2 + 0.2*x2**5 + 5*x1 + 10

# Generate noisy data with Gaussian noise for the function f(x)
X1 = np.random.uniform(0, 5, 100)  # Feature 1 values
X2 = np.random.uniform(0, 5, 100)  # Feature 2 values

y_true = f(X1, X2)
noise = np.random.normal(0, 200, 100)  # Gaussian noise
y_noisy = y_true + noise

# Create a Pandas DataFrame to store the data
df = pd.DataFrame({'X1': X1, 'X2': X2, 'y_true': y_true, 'y_noisy': y_noisy})

# Save the data to an Excel file
with pd.ExcelWriter('data.xlsx') as writer:
    df.to_excel(writer, index=False)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1, X2, y_noisy, c=y_noisy, cmap='RdYlGn')
ax.set_zlim(-100, 100)  # Set z-axis limits to magnify noise

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y_noisy')

# Create a 2D scatter plot to visualize noise
fig, ax = plt.subplots()
ax.scatter(X1, X2, c=y_noisy, cmap='coolwarm')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('Noise Distribution')

plt.show()