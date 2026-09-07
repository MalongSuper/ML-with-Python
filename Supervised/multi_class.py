# Multi-Class Classification: One-vs-One and One-vs-All approach
# Model: Support Vector Machine
# Reference: https://www.nb-data.com/p/one-vs-all-vs-one-vs-one-which-multi
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('glass.csv')
target = 'Type'
features = df.columns.drop(target)

# Group by type
df_group = df.groupby('Type')
print(df_group.head(3))
# Aggregate on mean
mean_group = round(df.groupby('Type').agg('mean'), 2)
print("\n", mean_group)


# 80% training and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df[features]
X_test = test_df[features]
y_train = train_df[target]
y_test = test_df[target]

print("\nMulti-Class Classification: One-vs-One and One-vs-All approach")
# Train an SVC model
model = SVC(kernel='linear')
# One-vs-All (OvA) approach
ova_classifier = OneVsRestClassifier(model)
ova_classifier.fit(X_train, y_train)
y_pred_ova = ova_classifier.predict(X_test)
accuracy_ova = accuracy_score(y_test, y_pred_ova)
conf_matrix_ova = confusion_matrix(y_test, y_pred_ova)
print(f"One-vs-All Accuracy:", accuracy_ova)

# One-vs-One (OvO) approach
ovo_classifier = OneVsOneClassifier(model)
ovo_classifier.fit(X_train, y_train)
y_pred_ovo = ovo_classifier.predict(X_test)
accuracy_ovo = accuracy_score(y_test, y_pred_ovo)
conf_matrix_ovo = confusion_matrix(y_test, y_pred_ovo)
print(f"One-vs-One Accuracy:", accuracy_ovo)

unique_labels = df[target].unique().tolist()
# Plot the confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(conf_matrix_ova, ax=axes[0], annot=True, fmt="d", cmap="Blues",
            xticklabels=unique_labels, yticklabels=unique_labels)
axes[0].set_title("One-vs-All")


sns.heatmap(conf_matrix_ovo, ax=axes[1], annot=True, fmt="d", cmap="Blues",
            xticklabels=unique_labels, yticklabels=unique_labels)
axes[1].set_title("One-vs-One")

plt.suptitle("SVM: Confusion Matrix")
plt.tight_layout()
plt.show()
