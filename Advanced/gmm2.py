# Gaussian Mixture Model
# Expectation–Maximization algorithm
import pandas as pd
import numpy as np


# Multivariate Gaussian for multiple features
def gaussian_pdf(x, mu, sigma):
    d = len(x)
    det_sigma = np.linalg.det(sigma)  # Determinant of sigma
    inv_sigma = np.linalg.inv(sigma)  # Inverse of sigma
    diff = x - mu  # difference between x and mu
    exponent = 1 / ((2 * np.pi) ** d/2 * (det_sigma ** 0.5))
    # Since it is matrix, we use dot product "@"
    fx = exponent * np.exp(-0.5 * diff.T @ inv_sigma @ diff)
    return fx


def gaussian_mixture(df, K=2):  # If not defined, assume K = 2
    # Convert the DataFrame into a numerical matrix that can be used for
    # probability calculations
    X = df.values
    N = len(X)
    # Number of features
    n_features = X.shape[1]
    # Random means
    mu = X[np.random.choice(len(X), K, replace=False)]
    # Identity covariance matrices
    sigma = [np.eye(n_features) for _ in range(K)]
    # Equal mixing weights
    # The mixing weight pi_k represents the
    # probability that a random data point comes from cluster k.
    # To satisfy this constraint easily, we start with equal probabilities
    pi = np.ones(K) / K
    # Inside this loop, we compute
    # the posterior probability (responsibility) for every cluster.
    numerator = []
    for n in range(N):
        for k in range(K):
            num = pi[k] * gaussian_pdf(X[n], mu[k], sigma[k])
            numerator.append(num)
    return numerator


def expectation_maximization(df, K=2, max_iter=100, tol=1e-4):
    X = df.values
    N, d = X.shape
    responsibilities = []
    # Initial parameters
    mu = X[np.random.choice(N, K, replace=False)]
    sigma = [np.eye(d) for _ in range(K)]
    pi = np.ones(K) / K
    log_likelihoods = []
    for iteration in range(max_iter):
        # E-Step
        responsibilities = np.zeros((N, K))
        for n in range(N):
            numerator = []
            for k in range(K):
                value = pi[k] * gaussian_pdf(X[n], mu[k], sigma[k])
                numerator.append(value)
            denominator = np.sum(numerator)
            for k in range(K):
                responsibilities[n, k] = numerator[k] / denominator

        # M-Step
        Nk = np.sum(responsibilities, axis=0)  # axis=0 for column-wise
        # Update means
        # This moves the cluster center toward the data points that belong more strongly to that cluster
        mu_new = []
        for k in range(K):
            weighted_sum = np.sum(responsibilities[:, k].reshape(-1,1) * X, axis=0)
            mu_k = weighted_sum / Nk[k]
            mu_new.append(mu_k)
        mu = np.array(mu_new)
        # Update covariance
        # This determines the spread and orientation of the Gaussian distribution
        sigma_new = []
        for k in range(K):
            sigma_k = np.zeros((d, d))
            for n in range(N):
                diff = (X[n] - mu[k]).reshape(-1,1)
                sigma_k += responsibilities[n,k] * (diff @ diff.T)
            sigma_k = sigma_k / Nk[k]
            # Regularization to avoid singular matrix
            sigma_k += 1e-6 * np.eye(d)
            sigma_new.append(sigma_k)
        sigma = sigma_new
        # Update mixing weights
        # The mixing weights represent the proportion of the dataset belonging to each cluster
        pi = Nk / N
        # Log-Likelihood
        log_likelihood = 0
        for n in range(N):
            total = 0
            for k in range(K):
                total += pi[k] * gaussian_pdf(X[n], mu[k], sigma[k])
            log_likelihood += np.log(total)
        log_likelihoods.append(log_likelihood)
        # Convergence Check
        if iteration > 0:
            if abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                print(f"Converged at iteration {iteration}")
                break
    return mu, sigma, pi, responsibilities, log_likelihoods


data = {
    "sepal.length": [5.1, 4.9, 4.7, 4.6, 7.0, 6.4, 6.9, 5.5, 6.3, 5.8, 7.1, 6.3],
    "sepal.width":  [3.5, 3.0, 3.2, 3.1, 3.2, 3.2, 3.1, 2.3, 3.3, 2.7, 3.0, 2.9],
    "petal.length": [1.4, 1.4, 1.3, 1.5, 4.7, 4.5, 4.9, 4.0, 6.0, 5.1, 5.9, 5.6],
    "petal.width":  [0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3, 2.5, 1.9, 2.1, 1.8]
}

df = pd.DataFrame(data)

mu, sigma, pi, responsibilities, log_likelihoods = expectation_maximization(df, K=3)
print("Mean:\n", mu)
print("Sigma:\n", sigma)
print("Pi:\n", pi)
print("Responsibilities:\n", responsibilities)
print("Log Likelihoods:\n", log_likelihoods)
# Assign clusters
print("Cluster Assignments:")
clusters = np.argmax(responsibilities, axis=1)
df["Cluster"] = clusters
print(df)
