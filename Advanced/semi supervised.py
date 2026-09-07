# Semi-Supervised Learning
import pandas as pd
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, adjusted_rand_score)
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('breast_cancer_wisconsin.csv')
target = 'diagnosis'

unique_labels = df[target].unique()

# Replace values
df[target] = df[target].replace({unique_labels[i]: i for i in range(len(unique_labels))})

# Copy of original labels
y_unlabeled = df[target].copy()
# Randomly select 40% of the rows to be unlabeled
# Return only their indexes
unlabeled_idx = df.sample(frac=0.8, random_state=42).index
# Replace their labels to -1
y_unlabeled.loc[unlabeled_idx] = -1

# Features
X = df.drop(columns=[target]).values
y_semi_array = y_unlabeled.values  # for scikit-learn

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Label Propagation
model = LabelPropagation(kernel='rbf', gamma=20, max_iter=1000)
model.fit(X_scaled, y_semi_array)

y_pred = model.transduction_

# Evaluation (if true labels are available)
y_true = df[target].values
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("ARI:", adjusted_rand_score(y_true, y_pred))

# Mask: True for labeled samples, False for unlabeled
mask = y_semi_array != -1

# True labels for colored plot (before propagation)
y = df[target].values
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c='lightgray', s=30, label='Unlabeled')
ax[0].scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=y[mask], cmap='viridis', s=60, label='Labeled')
ax[0].set_title("Before propagation — few labels")
ax[0].legend()

ax[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', s=60)
ax[1].set_title("After propagation — all labeled")

plt.tight_layout()
plt.show()
