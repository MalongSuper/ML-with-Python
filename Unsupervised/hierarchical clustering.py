# Hierarchical Clustering
# Reference: https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage


def agglomerative_clustering(X):
    # Agglomerative Clustering: Using Bottom-Up approach.
    clustering = AgglomerativeClustering(n_clusters=3)
    labels = clustering.fit_predict(X)
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model.fit(X)
    return labels


def divisive_clustering(X, max_clusters=3):
    # Divisive Clustering: Top-Down
    clusters = [X]
    while len(clusters) < max_clusters:
        # choose the largest cluster to split
        cluster_to_split = max(clusters, key=lambda x: len(x))
        clusters.remove(cluster_to_split)

        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(cluster_to_split)

        cluster1 = cluster_to_split[labels == 0]
        cluster2 = cluster_to_split[labels == 1]
        clusters.extend([cluster1, cluster2])

    return clusters


def plot_dendrogram(X):
    plt.figure(figsize=(12, 5))

    # Agglomerative dendrogram
    plt.subplot(1, 2, 1)
    Z = linkage(X, method='ward')
    dendrogram(Z)
    plt.title("Agglomerative Clustering Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")

    # Divisive approximation dendrogram
    plt.subplot(1, 2, 2)
    Z_div = linkage(X, method='ward')
    dendrogram(Z_div)
    plt.title("Divisive Clustering (Top-Down Approximation)")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")

    plt.tight_layout()
    plt.show()


# Execute the function
df = pd.read_csv('datasets/iris.csv')

target = 'variety'
X = df.drop(columns=target).values
agglomerative_clustering(X)
divisive_clustering(X)
plot_dendrogram(X)
