# Random Forest Classifier with Python
# Detect faults in a power system
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('detect_dataset.csv')
# Drop unnamed columns
df = df.drop(columns=["Unnamed: 7",  "Unnamed: 8"])
print(df.head(20))
# Split the data into 80% for training and 20% for testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training set shape: {train_df.shape}")
print(f"Testing set shape: {test_df.shape}")
# Features and Target
features = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
# Note: The target for Regression can be a continuous variable
X_train = train_df[features]
y_train = train_df['Output (S)']
X_test = test_df[features]
y_test = test_df['Output (S)']
# Train the model with Random Forest
# if not specified, n_estimators is 100
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")

# Testing Demo
sample = X_test.iloc[0]  # Selecting any testing sample
print("Sample data for testing: \n", sample.tolist())
# Predict
predicted_vc = model.predict([sample])
print(f"Predicted Output(S): {predicted_vc[0]}")
print(f"Actual Output(S): {y_test.iloc[0]}")
