import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Read the Dataset
# Example data

# Create DataFrame
df = pd.read_csv('sample_dataset.csv').replace(['g', 'b'], [1, 0])
# Extract
target = "Class"
features = df.columns.drop(target)
# Train with Logistic Regression
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
X_train = df_train[features]
y_train = df_train[target]
X_test = df_test[features]
y_test = df_test[target]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# Confusion matrix and Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# --- Select 2 best features using RFE ---
rfe = RFE(model, n_features_to_select=2)
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]
print("Selected features:", list(selected_features))

# --- Train again with only 2 features ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train[selected_features], y_train)

# --- Predict and evaluate ---
y_pred = model.predict(X_test[selected_features])
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# --- Plot decision surface ---
# Create a meshgrid
x_min, x_max = X_train[selected_features[0]].min() - 1, X_train[selected_features[0]].max() + 1
y_min, y_max = X_train[selected_features[1]].min() - 1, X_train[selected_features[1]].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict probabilities on grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# --- Plot ---
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X_test[selected_features[0]], X_test[selected_features[1]],
            c=y_test, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.title("Decision Surface (Logistic Regression)")
plt.show()
