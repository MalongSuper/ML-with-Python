# Decision Trees with Python
# Optional: Calculate Accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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
# Train the model with decision tree
# If criterion is not specified, criterion by default is Gini
model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
model.fit(X_train, y_train)
# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")
# Training and Testing set score
print(f"Training set score: {model.score(X_train, y_train)}")
print(f"Testing set score: {model.score(X_test, y_test)}")
# Draw decision tree
plt.figure(figsize=(12, 6))
tree.plot_tree(model, feature_names=features, class_names=["No", "Yes"])
plt.show()
