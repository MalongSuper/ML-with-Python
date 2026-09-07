# T-Distributed Stochastic Neighbor Embedding (t-SNE)
import numpy as np
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Compute Distance function
def compute_distances(X):
    sum_X = np.sum(np.square(X), axis=1)
    distances = sum_X[:, None] + sum_X[None, :] - 2 * np.dot(X, X.T)
    return distances


def compute_pij(D, sigma=1.0):
    P = np.exp(-D / (2 * sigma**2))
    np.fill_diagonal(P, 0)
    P = P / np.sum(P)
    return P


def compute_qij(Y):
    sum_Y = np.sum(np.square(Y), axis=1)
    num = 1 / (1 + sum_Y[:, None] + sum_Y[None, :] - 2 * np.dot(Y, Y.T))
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return Q


def kl_divergence(P, Q):
    return np.sum(P * np.log((P + 1e-10) / (Q + 1e-10)))


def update(Y, P, Q, lr=0.1):
    n = Y.shape[0]
    for i in range(n):
        grad = np.zeros(2)
        for j in range(n):
            if i != j:
                diff = (P[i, j] - Q[i, j]) * (Y[i] - Y[j])
                grad += diff
        Y[i] += lr * grad
    return Y


# Iris dataset (ignoring target labels)
X = np.array([[5.1, 3.5, 1.4, 0.2],
              [4.9, 3.0, 1.4, 0.2],
              [4.7, 3.2, 1.3, 0.2],
              [4.6, 3.1, 1.5, 0.2],
              [5.0, 3.6, 1.4, 0.2],
              [7.0, 3.2, 4.7, 1.4],
              [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5],
              [5.5, 2.3, 4.0, 1.3],
              [6.5, 2.8, 4.6, 1.5],
              [6.3, 3.3, 6.0, 2.5],
              [5.8, 2.7, 5.1, 1.9],
              [7.1, 3.0, 5.9, 2.1],
              [6.3, 2.9, 5.6, 1.8],
              [6.5, 3.0, 5.8, 2.2]])


# Normalize the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# Compute Euclidean Distance
D = compute_distances(X)
P = compute_pij(D)

n = X.shape[0]
Y = np.random.randn(n, 2)


for epoch in range(100):
    Q = compute_qij(Y)
    loss = kl_divergence(P, Q)
    Y = update(Y, P, Q)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")



# Load Data
iris = load_iris()

# Initialize and Fit t-SNE
# n_components=2 reduces the 4D data to 2D
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(iris.data)

# Visualize
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=iris.target)
plt.title("t-SNE of Iris Dataset")
plt.show()
