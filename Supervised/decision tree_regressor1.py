# Decision Tree Regressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data = {
    "Outlook":  [0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2],
    "Temp":     [0, 0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 1, 0, 1],
    "Humidity": [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    "Windy":    [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    "Hours Played": [26, 30, 46, 46, 62, 23, 43, 36, 38, 48, 48, 52, 44, 30]
}

df = pd.DataFrame(data)

target = "Hours Played"
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Features and Target
X_train = train_df[df.columns.drop(target)]
y_train = train_df[target]
X_test = test_df[df.columns.drop(target)]
y_test = test_df[target]

print("Decision Tree Regressor")
model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X_train, y_train)
# Make prediction
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Plot Decision Tree
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=df.columns.drop(target).tolist(), filled=True, rounded=True)
plt.title("Decision Tree Regressor")
plt.show()
