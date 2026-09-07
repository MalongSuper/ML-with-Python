# Quadratic Discriminant Analysis with Python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


df = pd.read_csv('datasets/student.csv').set_index('StudentID')

# Replace the Yes No in Part Time Job
df['Part_Time_Job'] = df['Part_Time_Job'].map({'Yes': 1, 'No': 0})
# One-Hot Encoding on Major
df = pd.get_dummies(df, columns=['Major']).replace({True: 1, False: 0})

# Features and Target
target = 'Part_Time_Job'
features = df.columns.drop(target)

# 80% training and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
y_train = train_df[target]
y_test = test_df[target]

print("Model: QDA with all features")

X_train = train_df[features]
X_test = test_df[features]

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

y_pred_qda = qda.predict(X_test)

accuracy_qda = accuracy_score(y_test, y_pred_qda)
print("Accuracy:", accuracy_qda)
print("Classification Report:\n", classification_report(y_test, y_pred_qda))

# Select two features
plot_features = ['School Year', 'English_Level']

X = df[plot_features]
y = df['Part_Time_Job']

# Train QDA on selected features
qda_2d = QuadraticDiscriminantAnalysis()
qda_2d.fit(X, y)

# Create mesh grid
x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Predict over grid
Z = qda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision regions
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot data points
scatter = plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y,
    edgecolor='k'
)

plt.xlabel('School Year')
plt.ylabel('English Level')
plt.title('QDA Decision Boundary (School Year vs English Level)')
plt.legend(*scatter.legend_elements(), title="Part-Time Job")
plt.show()
