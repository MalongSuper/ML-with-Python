# K-Mean Clustering for Housing
# Determine best K using the Elbow Method or Silhouette Method
# Unsupervised Learning: No data split needed
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def elbow_method(X):
    distortions = []
    # Perform K-Means on multiple k
    K = range(2, 16)
    for k in K:
        model = KMeans(n_clusters=k)
        model.fit(X)
        # Calculate their inertia (WCSS)
        distortions.append(model.inertia_)
    # Compute second derivative (discrete approximation)
    diff = np.diff(distortions, 2)
    elbow_point = np.argmin(diff) + 1  # offset due to diff

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, marker='o')
    plt.scatter(K[elbow_point], distortions[elbow_point], color='red', s=200,
                label=f"Elbow at k={K[elbow_point]}")
    plt.grid(True)
    plt.title('Elbow Point')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion (WCSS)')
    plt.legend()
    plt.show()


# Read data
df = pd.read_csv('datasets/housingforKMeans.csv',
                 usecols=['longitude', 'latitude', 'median_house_value'])
# Cluster by geography
X = df[['longitude', 'latitude']]
scaler = StandardScaler()  # Normalize the data
X_scaled = scaler.fit_transform(X)
elbow_method(X)
