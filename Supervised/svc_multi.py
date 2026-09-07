# Support Vector Machine for Multi-Class Classification
# One-vs-One (OvO) and One-vs-All: OvA (One-vs-Rest: OvR)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('wine_dataset.csv')
target = 'Target'
features = df.columns.drop(target)

# 80% training and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df[features]
X_test = test_df[features]
y_train = train_df[target]
y_test = test_df[target]

print("Support Vector Machine for Multi-Class Classification")
# Train an SVC model with One-vs-One strategy
model1 = SVC(decision_function_shape='ovo')
model1.fit(X_train, y_train)
# Predict and evaluate the One-vs-One model
y_pred1 = model1.predict(X_test)
print("One-vs-One Accuracy:", accuracy_score(y_test, y_pred1))
print("Classification Report:\n", classification_report(y_test, y_pred1))
# Train an SVC model with One-vs-All strategy
model2 = SVC(decision_function_shape='ovr')
model2.fit(X_train, y_train)
# Predict and evaluate the One-vs-All model
y_pred2 = model1.predict(X_test)
print("\nOne-vs-All Accuracy:", accuracy_score(y_test, y_pred2))
print("Classification Report:\n", classification_report(y_test, y_pred2))
