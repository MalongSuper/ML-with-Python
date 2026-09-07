# Random Forest Regressor with Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/detect_dataset.csv')
# Drop unnamed columns
df = df.drop(columns=["Unnamed: 7",  "Unnamed: 8"])
print(df.head(20))
# Split the data into 80% for training and 20% for testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training set shape: {train_df.shape}")
print(f"Testing set shape: {test_df.shape}")
# Features and Target
features = ['Output (S)', 'Ia', 'Ib', 'Ic', 'Va', 'Vb']
# Note: The target for Regression can be a continuous variable
X_train = train_df[features]
y_train = train_df['Vc']
X_test = test_df[features]
y_test = test_df['Vc']
# Train the model with Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Calculate Mean Square Error, R2 Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Testing Demo
sample = X_test.iloc[0]  # Selecting any testing sample
print("Sample data for testing: \n", sample.tolist())
# Predict
predicted_vc = model.predict([sample])
print(f"Predicted Vc: {predicted_vc[0]}")
print(f"Actual Vc: {y_test.iloc[0]}")

# Draw plots
importances = model.feature_importances_
feature_names = features
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()
