import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

sample_data = {'First-Semester Score (X)': [4.5, 5.2, 6.0, 6.4, 7.1, 7.3, 8.0, 8.4, 9.1, 9.5],
               'Second-Semester Score (Y)': [5.1, 6.3, 6.7, 7.5, 7.8, 8.1, 8.6, 9.0, 9.4, 9.8]}

df = pd.DataFrame(sample_data)
# 80% Training, 20% Testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print("Training Data:\n", train_df)
# Features and Targets
X_train = train_df[['First-Semester Score (X)']]
y_train = train_df['Second-Semester Score (Y)']
X_test = test_df[['First-Semester Score (X)']]
y_test = test_df['Second-Semester Score (Y)']
# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# Display the linear regression equation
A = model.intercept_
B = model.coef_[0]
print(f"Linear Equation: Y = {A} + ({B} * X)")
# Evaluation
y_predict = model.predict(X_test)
# Compute the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_predict)
print(f"Mean Squared Error on the test set: {mse}")
# Compute the R² score
r2 = r2_score(y_test, y_predict)
print(f"R² Score on the test set: {r2}")

# Scatter plot of actual data
plt.scatter(df['First-Semester Score (X)'], df['Second-Semester Score (Y)'],
            color='blue', label='Actual Data')
# Regression line
X_range = [[min(df['First-Semester Score (X)'])],
           [max(df['First-Semester Score (X)'])]]
y_range = model.predict(X_range)
plt.plot([x[0] for x in X_range], y_range, color='red', linewidth=2, label='Regression Line')

# Labels and title
plt.xlabel('First-Semester Score (X)')
plt.ylabel('Second-Semester Score (Y)')
plt.title(f'Linear Regression: First vs Second Semester Scores: Y = {A:.2f} + ({B:.2f} * X)')
plt.legend()
plt.grid(True)
plt.show()
