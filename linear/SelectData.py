import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Read the Excel file
dataset = pd.read_excel('./linear/3.xlsx')

# Assume the first column is the feature (X) and the second column is the target (y)
X = dataset.iloc[:, 0].values.reshape(-1, 1)
y = dataset.iloc[:, 1].values.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using Gradient Descent
model.fit(X_train, y_train)

# Estimate the parameters (slope and intercept)
slope = model.coef_[0][0]
intercept = model.intercept_[0]

print(f"Estimated parameters: slope = {slope:.2f}, intercept = {intercept:.2f}")

# Visualize the calculated model on all data
plt.scatter(X, y, label='Original Data')
plt.plot(X, slope * X + intercept, label='Linear Regression Model', color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Model on All Data')
plt.legend()
plt.show()

# Make predictions on the test set
y_pred = model.predict(X_test)

# Visualize the testing results
plt.scatter(X_test, y_test, label='Test Data')
plt.plot(X_test, y_pred, label='Predicted Values', color='green')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Model on Test Data')
plt.legend()
plt.show()