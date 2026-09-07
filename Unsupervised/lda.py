# Linear Discriminant Analysis with Python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


data = {
    "StudentID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "School Year": [1, 2, 2, 2, 1, 1, 2, 1, 1, 1],
    "Gpa_Score": [3.5, 2.7, 2.4, 3.0, 3.5, 3.3, 2.1, 2.9, 1.8, 2.4],
    "Fitness_Score": [9.5, 8.4, 8.5, 8.8, 8.3, 7.8, 7.8, 8.5, 9.6, 9.4],
    "English_Level": [5, 1, 2, 3, 3, 2, 3, 4, 1, 1],
    "Major": ["IT", "IT", "Business", "Marketing", "Business", "IT", "Marketing", "IT", "Business", "Marketing"],
    "Part_Time_Job": ["No", "Yes", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes"]
}


df = pd.DataFrame(data).set_index('StudentID')

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

# Model 1: Only School Year
print("Model 1: Only School Year")
features1 = ['School Year']
X_train1 = train_df[features1]
X_test1 = test_df[features1]

model1 = LinearDiscriminantAnalysis(solver='svd')
model1.fit(X_train1, y_train)
t = (model1.means_[0] + model1.means_[1]) / 2
print(f"Model 1 Decision boundary (School Year threshold): x = {t[0]:.2f}")


y_pred1 = model1.predict(X_test1)
accuracy1 = accuracy_score(y_test, y_pred1)
print("Accuracy:", accuracy1)
print("Classification Report:\n", classification_report(y_test, y_pred1))

# Model 2: School Year + English Level
print("Model 2 (School Year + English Level)")
features2 = ['School Year', 'English_Level']

X_train2 = train_df[features2]
X_test2 = test_df[features2]

model2 = LinearDiscriminantAnalysis(solver='svd')
model2.fit(X_train2, y_train)

coef = model2.coef_[0]
intercept = model2.intercept_[0]
slope = -coef[0]/coef[1]
intercept_y = -intercept/coef[1]
print(f"Model 2 (2D) Decision Boundary: English_Level = {slope:.3f}*School_Year + {intercept_y:.3f}")
print(f"Model 2 Threshold (projected mean between classes on LDA axis): {intercept_y:.3f}")

y_pred2 = model2.predict(X_test2)
accuracy2 = accuracy_score(y_test, y_pred2)
print("Accuracy:", accuracy2)
print("Classification Report:\n", classification_report(y_test, y_pred2))


# Model 1: 1D LDA (School Year)
plt.figure(figsize=(12,5))

# 1D subplot
plt.subplot(1,2,1)
# Plot train points
for label in [0,1]:
    plt.scatter(train_df['School Year'][y_train==label],
                np.zeros_like(train_df['School Year'][y_train==label]) + label,
                label=f'Part Time Job={label}', s=100)

# Decision threshold
threshold1 = (model1.means_[0] + model1.means_[1]) / 2
plt.axvline(x=threshold1, color='red', linestyle='--', label='Decision boundary')
plt.xlabel('School Year')
plt.ylabel('Class (0=No,1=Yes)')
plt.title('Model 1: LDA Decision Boundary (1D)')
plt.legend()
plt.yticks([0,1])

# Model 2: 2D LDA (School Year + English_Level)
plt.subplot(1,2,2)

# Scatter plot
for label in [0,1]:
    plt.scatter(train_df['School Year'][y_train==label],
                train_df['English_Level'][y_train==label],
                label=f'Part Time Job={label}', s=100)

# LDA decision line
coef = model2.coef_[0]
intercept = model2.intercept_[0]

# Line: coef[0]*x1 + coef[1]*x2 + intercept = 0
# Solve for x2 (English_Level)
x_vals = np.array([train_df['School Year'].min()-0.5, train_df['School Year'].max()+0.5])
y_vals = -(coef[0]*x_vals + intercept)/coef[1]
plt.plot(x_vals, y_vals, 'r--', label='Decision boundary')

plt.xlabel('School Year')
plt.ylabel('English Level')
plt.title('Model 2: LDA Decision Boundary (2D)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
