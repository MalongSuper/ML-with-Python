# Softmax Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('iris.csv')
target = 'variety'

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Features and Target
X_train = train_df[df.columns.drop(target)]
y_train = train_df[target]
X_test = test_df[df.columns.drop(target)]
y_test = test_df[target]

# Train a Logistic Regression
print("Model 1: Logistic Regression (Softmax for multi-class)")
model1 = LogisticRegression(random_state=42, solver='lbfgs',
                           multi_class='multinomial', max_iter=200)

model1.fit(X_train, y_train)
# Add Accuracy Score
y_pred1 = model1.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred1)}")
print(f"Classification Report\n {classification_report(y_test, y_pred1)}")

# Print probabilities for the first 10 samples
n = 10
probs = model1.predict_proba(X_test.iloc[:10])
pred_classes = model1.predict(X_test.iloc[:10])
print("\nPredicted Probabilities for First 3 Test Samples:")
for i in range(n):
    print(f"\nSample {i + 1}:")
    print(f"True Label: {y_test.iloc[i]}")
    print(f"Predicted Class: {pred_classes[i]}")
    print("Class Probabilities:")
    for class_name, prob in zip(model1.classes_, probs[i]):
        print(f"{class_name}: {prob:.4f}", end="; ")

# Train a Stochastic Gradient Descent model
# max_iter in this case is number of epochs
print("\n\nModel 2: Stochastic Gradient Descent")
model2 = SGDClassifier(loss='log_loss', learning_rate='constant',
                      eta0=0.1, max_iter=100, random_state=42)
model2.fit(X_train, y_train)
# Add Accuracy Score
y_pred2 = model2.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred2)}")
print(f"Classification Report\n {classification_report(y_test, y_pred2)}")
