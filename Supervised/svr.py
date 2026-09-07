# Support Vector Regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Outlook":  [0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2],
    "Temp":     [0, 0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 1, 0, 1],
    "Humidity": [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    "Windy":    [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    "Hours Played": [26, 30, 46, 46, 62, 23, 43, 36, 38, 48, 48, 52, 44, 30]
}

df = pd.DataFrame(data)
# Gram Matrix (Inner product)
gram_matrix = np.zeros((len(df), len(df)))
print("Gram Matrix:")
for i in range(len(df)):
    for j in range(len(df)):
        # Append to the respective position
        gram_matrix[i, j] = df.iloc[i, 0:-1].dot(df.iloc[j, 0:-1])
print(gram_matrix)

target = "Hours Played"
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Features and Target
X_train = train_df[df.columns.drop(target)]
y_train = train_df[target]
X_test = test_df[df.columns.drop(target)]
y_test = test_df[target]

# Train SVR model using Linear kernel
model1 = SVR(kernel='linear', C=100, epsilon=5)
model1.fit(X_train, y_train)
# Prediction
y_pred1 = model1.predict(X_test)
print("Model 1: Support Vector Regressor (Linear Kernel; C=100; eps=5)")
print(f"MSE: {mean_squared_error(y_test, y_pred1)}")
print(f"R^2 Score: {r2_score(y_test, y_pred1)}")

# Train SVR model using RBF kernel, change the epsilons
# Train SVR model using Linear kernel
model2 = SVR(kernel='rbf', C=100, gamma=0.5, epsilon=0.5)
model2.fit(X_train, y_train)
# Prediction
y_pred2 = model2.predict(X_test)
print("Model 2: Support Vector Regressor (RBF Kernel; C=100; gamma=0.5; eps=0.5)")
print(f"MSE: {mean_squared_error(y_test, y_pred2)}")
print(f"R^2 Score: {r2_score(y_test, y_pred2)}")
