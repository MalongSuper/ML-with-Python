# Gaussian Mixture Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


# Load the dataset
df = pd.read_csv('iris.csv')
target = 'variety'
features = df.columns.drop(target)

X = df[features].values
model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
model.fit(X)
labels = model.predict(X)

plt.figure(figsize=(8, 6))
plt.title("Gaussian Mixture Model Clustering")
plt.scatter(X[:,2], X[:,3], c=labels, cmap='viridis', s=50, edgecolor='k')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.grid(True)
plt.legend()
plt.show()
