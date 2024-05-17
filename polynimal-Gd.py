import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load your data into a Pandas dataframe
df = pd.read_excel('2feactherGDProblem.xlsx')

# Assuming X1 and X2 are your features, and Y is your target variable
X = df[['X1', 'X2']]
Y = df['y_noisy']

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a polynomial features object with degree 2 (you can adjust this)
poly_features = PolynomialFeatures(degree=5)

# Transform your training data into polynomial features
X_train_poly = poly_features.fit_transform(X_train)

# Create a linear regression object
lr_model = LinearRegression()

# Train the model on the polynomial features
lr_model.fit(X_train_poly, y_train)

# Print the estimated coefficients (parameters)
print("Estimated coefficients:")
print("Intercept:", lr_model.intercept_)
print("Coefficients:", lr_model.coef_)

# You can also get the polynomial coefficients using the `get_params()` method
print("Polynomial coefficients:")
print(poly_features.get_params())


# Evaluate the model on the testing data
y_pred = lr_model.predict(poly_features.transform(X_test))
print("Mean squared error:", np.mean((y_pred - y_test) ** 2))

# You can also use cross-validation to optimize the model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr_model, X_train_poly, y_train, cv=2, scoring='neg_mean_squared_error')
print("Cross-validation scores:", scores)