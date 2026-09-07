# Calculate PCA of a dataset
import pandas as pd
import numpy as np

sample_data = {'Student': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
                           'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20'],
               'Height': [160, 161, 159, 162, 170, 169, 168, 172, 171, 167,
                          180, 178, 179, 181, 182, 190, 191, 150, 149, 151],
               'Weight': [55, 56, 54, 57, 70, 72, 68, 75, 74,
                          69, 90, 85, 88, 92, 91, 95, 98, 45, 46, 47],
               'Age': [18, 18, 17, 18, 19, 19, 19, 20, 20, 19,
                       21, 21, 21, 22, 22, 23, 23, 17, 17, 17],
               'First_GPA': [3.0, 3.1, 2.8, 3.2, 3.5, 3.4, 3.3, 3.6, 3.5, 3.2, 3.8, 3.7, 3.8, 3.9, 3.9,
                             4.0, 4.0, 2.2, 2.3, 2.4],
               'Second_GPA': [3.2, 3.3, 3.0, 3.4, 3.6, 3.5, 3.4, 3.7, 3.6, 3.4, 3.9, 3.8,
                              4.0, 4.0, 4.0, 4.0, 4.0, 2.5, 2.4, 2.6],
               'Scholarship': [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]}

# Get the columns and their values
df = pd.DataFrame(sample_data)
df = df.drop(columns=['Student'])
df_col_values = []
for k in df.columns:
    col = df[f'{k}'].tolist()
    df_col_values.append(col)

# Mean and Standard Deviation
df_mean = df.mean(axis=0)
df_mean_list = df_mean.tolist()
df_std = df.std(axis=0)
df_std_list = df_std.tolist()

# Step 1: Standardize Values
z_score = []
for j in range(len(df_mean_list)):
    for i in range(len(df_col_values[0])):
        z = (df_col_values[j][i] - df_mean_list[j]) / df_std_list[j]
        z_score.append(z)

# Reshape z-scores into 20×6 matrix
z_array = np.array(z_score).reshape(len(df), len(df.columns))
print("Z Score:\n", z_array)

# Step 2: Compute the Covariance Matrix (since data is standardized, correlation ≈ covariance)
cov_matrix = np.cov(z_array, rowvar=False)
print("\nCovariance Matrix:\n", cov_matrix)

# Step 3: Eigenvalues and Eigenvectors
eig_value, eig_vector = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eig_value)
print("\nEigenvectors:\n", eig_vector)

# Sort eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eig_value)[::-1]  # Sort in descending order
eig_value_sorted = eig_value[sorted_indices]
eig_vector_sorted = eig_vector[:, sorted_indices]

# Compute the explained variance ratio
explained_variance_ratio = eig_value_sorted / np.sum(eig_value_sorted)
print("\nExplained Variance Ratio:\n", explained_variance_ratio)
# Select top p components (e.g., p=2 for a 2D transformation)
p = 2
feature_vector = eig_vector_sorted[:, :p]
print("\nFeature Vector Matrix:\n", feature_vector)
# Step 5: Transform original data
reduced_data = np.dot(z_array, feature_vector)
print("\nTransformed Data:\n", reduced_data)
