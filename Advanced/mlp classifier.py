# Multi-Layer Perceptrons
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


df = pd.read_csv("malware_dataset.csv").set_index('hash')
target = 'classification'
features = df.columns.drop(target)
# Label Encoding for the target
df[target] = df[target].map({"benign": 0, "malware": 1})

# 80% training and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df[features]
X_test = test_df[features]
y_train = train_df[target]
y_test = test_df[target]

# Feature scaling (important for Neural Networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build MLP model
model = MLPClassifier(hidden_layer_sizes=(64,32), activation='logistic',
                      solver='adam', max_iter=300, random_state=42)
# Train model
model.fit(X_train, y_train)
# Model Evaluation
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization of Training Loss
plt.plot(model.loss_curve_)
plt.title("Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
