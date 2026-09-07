# K-Fold Cross Validation
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

# Read the Breast Cancer
df = pd.read_csv('datasets/breast_cancer_wisconsin.csv')
target = 'diagnosis'

# Extract the features and target as numpy array
X = df[df.columns.drop(target)].to_numpy()
y = df[target].map({'M': 1, 'B': 0}).to_numpy()

# Define the model
model = DecisionTreeClassifier(random_state=42)
# Define K-Fold Cross-Validation
# n_splits=5: splits the data into 5 folds
# shuffle=True: shuffles data before splitting to avoid bias
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
# Perform cross-validation
scores = cross_val_score(model, X, y, cv=k_fold,
                         scoring='f1')

print("F1-score for each fold:", scores)
print("Average F1-score:", scores.mean())
