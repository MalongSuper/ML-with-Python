# Elastic Net Regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

sample_data = {'First-Semester Score (X)': [4.5, 5.2, 6.0, 6.4, 7.1, 7.3, 8.0, 8.4, 9.1, 9.5],
               'Second-Semester Score (Y)': [5.1, 6.3, 6.7, 7.5, 7.8, 8.1, 8.6, 9.0, 9.4, 9.8]}

df = pd.DataFrame(sample_data)
# 80% Training, 20% Testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print("Training Data:", train_df)
# Features and Targets
X_train = train_df[['First-Semester Score (X)']]
y_train = train_df['Second-Semester Score (Y)']
X_test = test_df[['First-Semester Score (X)']]
y_test = test_df['Second-Semester Score (Y)']

# l1_ratio: Defines the mix between L1 and L2.
# + l1_ratio = 1: Pure Lasso.
# + l1_ratio = 0: Pure Ridge.
# + 0 < l1_ratio < 1: A hybrid approach that allows for both feature selection
# (setting some coefficients to zero)
# and coefficient shrinkage (grouping correlated variables).
model = ElasticNet(alpha=0.5, l1_ratio=0.5)
model.fit(X_train, y_train)
# Display the Elastic Net Regression equation
A = model.intercept_
B = model.coef_[0]
print(f"Elastic Net Equation: Y = {A} + ({B} * X)")
# Evaluation
y_predict = model.predict(X_test)
# Compute the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_predict)
print(f"Mean Squared Error on the test set: {mse}")
# Compute the R² score
r2 = r2_score(y_test, y_predict)
print(f"R² Score on the test set: {r2}")

# Train the Linear Regression for plotting
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)
x_line = np.linspace(min(df['First-Semester Score (X)'])-0.5,
                     max(df['First-Semester Score (X)'])+0.5, 100)
y_ols = ols_model.intercept_ + ols_model.coef_[0] * x_line
y_en = model.intercept_ + model.coef_[0] * x_line

# Plot
plt.figure(figsize=(8,6))
plt.scatter(df['First-Semester Score (X)'], df['Second-Semester Score (Y)'],
            color="black", label="Data points")
plt.plot(x_line, y_ols, color="blue", label="OLS Regression Line")
plt.plot(x_line, y_en, color="red", linestyle="--", label="Elastic Net Regression Line")

plt.xlabel("First-Semester Score (X)")
plt.ylabel("Second-Semester Score (Y)")
plt.title("Geometric Interpretation of Elastic Net Regression")
plt.legend()
plt.grid(True)
plt.show()
