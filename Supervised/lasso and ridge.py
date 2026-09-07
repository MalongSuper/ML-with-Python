import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

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
# Fit the model
print("Model: Linear Regression")
model1 = LinearRegression()
model1.fit(X_train, y_train)
# Display the linear regression equation
A1 = model1.intercept_
B1 = model1.coef_[0]
print(f"Linear Equation: Y = {A1} + ({B1} * X)")
# Evaluation
y_predict = model1.predict(X_test)
# Compute the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_predict)
print(f"Mean Squared Error on the test set: {mse}")
# Compute the R² score
r2 = r2_score(y_test, y_predict)
print(f"R² Score on the test set: {r2}")

print("\nModel: Ridge Regression")
model2 = Ridge(alpha=0.01)
model2.fit(X_train, y_train)
# Display the Ridge Regression equation
A2 = model2.intercept_
B2 = model2.coef_[0]
print(f"Ridge Equation: Y = {A2} + ({B2} * X)")

print("\nModel: Lasso Regression")
model3 = Lasso(alpha=0.01)
model3.fit(X_train, y_train)
# Display the Ridge Regression equation
A3 = model3.intercept_
B3 = model3.coef_[0]
print(f"Lasso Equation: Y = {A3} + ({B3} * X)")

x_line = np.linspace(min(df['First-Semester Score (X)'])-0.5,
                     max(df['First-Semester Score (X)'])+0.5, 100)
y_ols = A1 + B1 * x_line
y_ridge = A2 + B2 * x_line
y_lasso = A3 + B3 * x_line

# Plot
plt.figure(figsize=(8,6))
plt.scatter(df['First-Semester Score (X)'], df['Second-Semester Score (Y)'],
            color="black", label="Data points")
plt.plot(x_line, y_ols, color="blue", label="OLS Regression Line")
plt.plot(x_line, y_ridge, color="red", linestyle="--", label="Ridge Regression Line")
plt.plot(x_line, y_lasso, color="green", linestyle="--", label="Lasso Regression Line")

plt.xlabel("First-Semester Score (X)")
plt.ylabel("Second-Semester Score (Y)")
plt.title("Regression Lines: OLS, Ridge, and Lasso")
plt.legend()
plt.grid(True)
plt.show()
