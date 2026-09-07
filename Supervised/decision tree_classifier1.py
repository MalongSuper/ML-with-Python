# Decision Trees with Python
# Classification Tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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
# Note: The target should be a discrete variable
X_train = train_df[features]
y_train = train_df['Output (S)']
X_test = test_df[features]
y_test = test_df['Output (S)']
# Train the model with decision tree
# If criterion is not specified, criterion by default is Gini
# or just DecisionTreeClassifier() -> Gini
model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0)
model.fit(X_train, y_train)
# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")
# Draw decision tree
plt.figure(figsize=(12, 6))
tree.plot_tree(model, feature_names=features, class_names=["No", "Yes"])
plt.show()
