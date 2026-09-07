# Calculate PCA with Scikit-Learn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# Standardize the Data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Perform
pca = PCA(n_components=2)  # Selecting top 2 principal components
principal_components = pca.fit_transform(df_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:\n", explained_variance_ratio)
# Principal Component Matrix
print("\nTransformed Data:\n", principal_components)
