# Regularization in Machine Learning
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate a dataset with some irrelevant features
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    noise=20,
    random_state=42
)

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create L1 and L2 regularized models
lasso = make_pipeline(
    StandardScaler(),
    Lasso(alpha=1.0, max_iter=10000)
)

ridge = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

# 4. Train both models
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)

# 5. Evaluate their performance
for name, model in [("Lasso (L1)", lasso), ("Ridge (L2)", ridge)]:
    predictions = model.predict(X_test)

    print(f"{name}")
    print("MSE:", round(mean_squared_error(y_test, predictions), 2))
    print("R²:", round(r2_score(y_test, predictions), 3))

# 6. Examine the learned coefficients
lasso_coef = lasso[-1].coef_
ridge_coef = ridge[-1].coef_

results = pd.DataFrame({
    "Feature": [f"Feature {i+1}" for i in range(X.shape[1])],
    "L1 Coefficient": lasso_coef,
    "L2 Coefficient": ridge_coef
})

print("\nLearned Coefficients:")
print(results.round(3).to_string(index=False))

print("\nNumber of zero L1 coefficients:",
      np.sum(np.isclose(lasso_coef, 0)))
