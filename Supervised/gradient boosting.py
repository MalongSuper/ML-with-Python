# Gradient Boosting Classifier with Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Define the dataset
data = {
    'Age': ['≤30', '≤30', '31…40', '>40', '>40', '>40', '31…40', '≤30', '≤30', '>40', '≤30', '31…40', '31…40', '>40'],
    'Income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium',
               'medium', 'medium', 'high', 'medium'],
    'Student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'Credit_Rating': ['Fair', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair',
                      'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair', 'Excellent'],
    'Buy_XBOX': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

# Prepare the data
df = pd.DataFrame(data)

# Convert Categorical Data to Numeric
# For example: no = 0, yes = 1
df['Age'] = df['Age'].replace({'≤30': 0, '31…40': 1, '>40': 2})
df['Income'] = df['Income'].replace({'low': 0, 'medium': 1, 'high': 2})
df['Student'] = df['Student'].replace({'no': 0, 'yes': 1})
df['Credit_Rating'] = df['Credit_Rating'].replace({'Fair': 0, 'Excellent': 1})
df['Buy_XBOX'] = df['Buy_XBOX'].replace({'no': 0, 'yes': 1})
# 80% training and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training:\n{train_df}")
print(f"Testing:\n{test_df}")
# Features and Target
features = ['Age', 'Income', 'Student', 'Credit_Rating']
X_train = train_df[features]
y_train = train_df['Buy_XBOX']
X_test = test_df[features]
y_test = test_df['Buy_XBOX']

# Train the model with Gradient Boosting
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Get staged predictions (i.e., predictions after each boosting round)
staged_predicts = list(model.staged_predict_proba(X_test))
# Print first 20 stages
for i, predicts in enumerate(staged_predicts[:20]):
    print(f"Stage {i+1}: Predicted probabilities:\n{predicts}")

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")
# Classification Report: Precision, Recall, F1-Score, Support
classification_report = classification_report(y_test, y_pred)
print(f"Classification Report: \n {classification_report}")

# Sample prediction with testing data
sample = X_test.iloc[0]
print("Sample data for testing: \n", sample)
predicted_val = model.predict([sample])
print(f"Predicted XBOX: {predicted_val[0]}")
print(f"Actual XBOX: {y_test.iloc[0]}")

# Optional: Feature Importance
importances = model.feature_importances_
feature_names = features

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Gradient Boosting Feature Importance")
plt.show()
