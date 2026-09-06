# K-Nearest Neighbor

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import euclidean_distances

# Read the load_iris_datasets 
df = pd.read_csv('load_iris_dataset.csv')

# Split the data into 80% for training and 20% for testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Features and Target
target = 'target'
features = df.columns.drop(target)
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Train with KNN with only 13 neighbors
model = KNeighborsClassifier(n_neighbors=13)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report)

# Euclidean Distances
distances = euclidean_distances(X_test, X_train)
print("Distances:", distances)
