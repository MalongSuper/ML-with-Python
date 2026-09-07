# Covariance Matrix and Correlation Matrix with Python
import pandas as pd

sample_data = {'Height': [5.50, 6.00, 5.70, 5.90],
               'Weight': [150, 180, 160, 170],
               'Age': [25, 30, 28, 35]}
# Convert to data frame
df = pd.DataFrame(sample_data)
cov_matrix = df.cov()
corr_matrix = df.corr()
print("Covariance Matrix:\n", cov_matrix)
print("Correlation Matrix:\n", corr_matrix)
