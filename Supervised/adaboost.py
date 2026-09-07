# Adaptive Boosting Classifier (or AdaBoost)
# Iris Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report


# Prepare the data
print("Data Preparation")
df = pd.read_csv('iris.csv')
print(df[df['variety'] == 'Setosa'].head(10))
print(df[df['variety'] == 'Versicolor'].head(10))
print(df[df['variety'] == 'Virginica'].head(10))

# Training and Testing Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training:\n{train_df}")
print(f"Testing:\n{test_df}")

# Features and Target
features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
X_train = train_df[features]
y_train = train_df['variety']
X_test = test_df[features]
y_test = test_df['variety']

model = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=42)
model.fit(X_train, y_train)
print("Training Complete!!")

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")
# Classification Report: Precision, Recall, F1-Score, Support
classification_report = classification_report(y_test, y_pred)
print(f"Classification Report: \n {classification_report}")

# Sample prediction with testing data
sample = X_test.iloc[0]
print("Sample data for testing:\n", sample)
predicted_val = model.predict([sample])
print(f"Predicted: {predicted_val[0]}")
print(f"Actual: {y_test.iloc[0]}")

sample = X_test.iloc[1]
print("Sample data for testing:\n", sample)
predicted_val = model.predict([sample])
print(f"Predicted: {predicted_val[0]}")
print(f"Actual: {y_test.iloc[1]}")

sample = X_test.iloc[2]
print("Sample data for testing:\n", sample)
predicted_val = model.predict([sample])
print(f"Predicted: {predicted_val[0]}")
print(f"Actual: {y_test.iloc[2]}")
