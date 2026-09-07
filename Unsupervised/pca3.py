# Kernel Principal Component Analysis
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np


# Function to calculate the RBF kernel matrix
def rbf_kernel_matrix(matrix_a, gamma=0.1):
    n = len(matrix_a)
    gram_matrix = np.zeros((n, n))  # Initialize the gram matrix
    for i in range(n):
        for j in range(n):
            # Compute Euclidean distance between each point
            p1, p2 = matrix_a[i], matrix_a[j]
            # Use Squared Distance
            squared_distance = np.linalg.norm(p1 - p2) ** 2
            # Plug in the distance to the gram matrix
            gram_matrix[i, j] = np.exp(-gamma * squared_distance)

    return gram_matrix


def main():
    # Test the function with a simple input
    A = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    print("Matrix A:\n", A)
    print("gamma = 0.1:\n", rbf_kernel_matrix(A))
    print("gamma = 1:\n", rbf_kernel_matrix(A, 1))

    # Generate an array with random values
    m, n = eval(input("\nEnter size (m x n): "))
    B = np.random.randint(1, 20, size=(m, n))
    g = float(input("Enter gamma (g): "))
    print("Matrix B:\n", B)
    print(rbf_kernel_matrix(B, g))


main()


''' Extra '''
# Generate a non-linear dataset
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
# Apply Kernel PCA with an RBF kernel
kernel_pca = KernelPCA(kernel='rbf', gamma=15, n_components=2)
X_kernel_pca = kernel_pca.fit_transform(X)

# Plot the original data and the transformed data
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
plt.title('Original Data')
plt.subplot(1, 2, 2)
plt.scatter(X_kernel_pca[:, 0], X_kernel_pca[:, 1], c=y, edgecolors='k', marker='o')
plt.title('Kernel PCA Transformed Data')
plt.show()
