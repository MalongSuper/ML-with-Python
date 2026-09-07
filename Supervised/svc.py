# Support Vector Machine with Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


# Read the dataframe
df = pd.read_csv('breast_cancer_wisconsin.csv')

target = 'diagnosis'
features = df.columns.drop(target)

# Label Encoding for the target
df['diagnosis'] = df['diagnosis'].map({"B": 0, "M": 1})

# 80% training and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df[features]
X_test = test_df[features]
y_train = train_df[target]
y_test = test_df[target]

print("Support Vector Machine")
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Classification report
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot the hyperplane with PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# Train SVM in PCA space
svm_pca = SVC(kernel='linear', C=1.0)
svm_pca.fit(X_train_pca, y_train)
# For a linear SVM: w_1 x_1 + w_2 x_2 + b = 0
w = svm_pca.coef_[0]
b = svm_pca.intercept_[0]
print(f"Hyperplane equation:")
print(f"{w[0]:.3f} * PC1 + {w[1]:.3f} * PC2 + {b:.3f} = 0")


# Create mesh grid
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

# Decision function
Z = svm_pca.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))

# Decision boundary and margins
plt.contour(xx, yy, Z, levels=[-1, 0, 1],
            linestyles=['--', '-', '--'])

# Scatter plot
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train,
            cmap='coolwarm', edgecolors='k', alpha=0.7)

# Support vectors
plt.scatter(svm_pca.support_vectors_[:, 0], svm_pca.support_vectors_[:, 1], s=120,
            facecolors='none', edgecolors='k', linewidths=2, label='Support Vectors')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Linear SVM Hyperplane (PCA Projection)")
plt.legend()
plt.show()
