# K-Mean Clustering for Housing
# Determine best K using the Elbow Method or Silhouette Method
# Unsupervised Learning: No data split needed
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def silhouette_method(X):
    silhouette_scores = []
    # Perform K-Means on multiple k
    K = range(3, 12)
    for k in K:
        model = KMeans(n_clusters=k)
        model.fit(X)
        # Calculate and append silhouette score
        score = silhouette_score(X, model.labels_, metric='euclidean')
        silhouette_scores.append(score)

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores, marker='o')
    plt.title("Silhouette Method for Optimal k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)

    # Highlight best k
    best_k = K[silhouette_scores.index(max(silhouette_scores))]
    best_score = max(silhouette_scores)
    plt.scatter(best_k, best_score, color='red', s=200, label=f"Best k = {best_k}")
    plt.legend()
    plt.show()


# Read data
df = pd.read_csv('datasets/housingforKMeans.csv',
                 usecols=['longitude', 'latitude', 'median_house_value'])
# Cluster by geography
X = df[['longitude', 'latitude']]
scaler = StandardScaler()  # Normalize the data
X_scaled = scaler.fit_transform(X)
silhouette_method(X)
