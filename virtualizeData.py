import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the Excel file
data = pd.read_excel('generatedData.xlsx')

# Extract X and y values from the data
X = data['X']
y = data['y_noisy']

# Create a scatter plot to visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Noisy Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Noisy Data Visualization')
plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig('noisy_data_plot.png')

# Display the plot
plt.show()