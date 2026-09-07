# Independent Component Analysis
import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv('gender_voice.csv')
# Fill in missing value with mean
df = df.fillna(df.drop(columns=['label']).mean())
# Mapping
df = df.replace({"male": 0, "female": 1})
# Only use three columns
columns = ["meanfreq", "sd", "median"]

# Find X_centered
X = df[columns]
# Column-wise mean
mu = X.mean(axis=0)
# Centered data (demonstration only)
X_centered = X - mu
print("X_c:\n", X_centered)

# FastICA automatically performs whitening, then finds the rotation matrix R.
ica = FastICA(n_components=3, whiten='unit-variance', random_state=0)
# S = RZ -> Independent components
S = ica.fit_transform(X_centered)
print("Independent components (S):\n", S)
# A = Mixing matrix
A = ica.mixing_
print("Mixing matrix A:\n", A)
# Whitening matrix V = D^{-1/2} E^T
V = ica.whitening_
print("Whitening matrix (V):\n", V)
# Whitened data Z = X_c V^T
Z = X_centered.values @ V.T
print("Whitened data (Z):\n", Z)
# Unmixing matrix W = RV
W = ica.components_
print("Unmixing Matrix (W):\n", W)
# Rotation matrix R = W V^T
# Because FastICA uses orthonormal whitening: V^-1 = V^T
R = W @ V.T
print("Rotation matrix (R):\n", R)


# Store independent components
df_ica = pd.DataFrame(S, columns=["IC1", "IC2", "IC3"])
# Scatter plot of ICA components
plt.scatter(S[:, 0], S[:, 1])
plt.xlabel("IC1")
plt.ylabel("IC2")
plt.title("ICA Projection")
plt.show()
