# K-Nearest Neighbors with Python
# Iris Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Prepare the data
df = pd.read_csv('iris.csv')
# Training and Testing Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Features and Target
features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
X_train = train_df[features]
y_train = train_df['variety']
X_test = test_df[features]
y_test = test_df['variety']

# Standard Scaler: Good for distance-based model
scaler = StandardScaler()
# Fit scaler on training data only
X_train_scaled = scaler.fit_transform(X_train)
# Transform test data
X_test_scaled = scaler.transform(X_test)

# Fit the KNN model using k = 1, 3, 5, 7, 9, 11, 13
for i in [1, 3, 5, 7, 9, 11, 13]:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train_scaled, y_train)
    print(f"Training KNN with {i} neighbors")
    # Predict and calculate accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy}")
