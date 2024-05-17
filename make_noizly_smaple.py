import numpy as np
import matplotlib as mtp
import pandas as pd


# Data creation

# Create independent variable
x = np.arange(0,100,2) # Produces [0, 100) with steps of 2.

# Use a linear function to obtain the dependent variable
y = 0.3*x + 0.6 # Parameters are arbitrary.

# Noise generation

# Genearte noise with same size as that of the data.
noise = np.random.normal(0,2, len(x)) #  μ = 0, σ = 2, size = length of x or y. Choose μ and σ wisely.

# Add the noise to the data.
y_noised = y + noise


data = pd.DataFrame({'X': x, 'y_noisy': y_noised})

# Save the data to an Excel file
data.to_excel('generatedData.xlsx', index=False)