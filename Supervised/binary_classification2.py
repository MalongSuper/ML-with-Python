import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


data = {
    "ID": list(range(1, 31)),
    "City": [
        "Madrid", "Barcelona", "Seville", "Valencia", "Madrid", "Barcelona", "Bilbao", "Zaragoza", "Madrid", "Girona",
        "Malaga", "Bilbao", "Seville", "Madrid", "Barcelona", "Valencia", "Toledo", "Alicante", "Murcia", "Santander",
        "Madrid", "Barcelona", "Granada", "Seville", "Pamplona", "Zaragoza", "Barcelona", "Madrid",
        "Valladolid", "Malaga"
    ],
    "Favorite Player": [
        "Vinicius Jr", "Pedri", "Jude Bellingham", "Gavi", "Modrić",
        "Lewandowski", "Vinicius Jr", "Pedri", "Bellingham", "Gavi",
        "Modrić", "Vinicius Jr", "Pedri", "Bellingham", "Gavi", "Lewandowski",
        "Jude Bellingham", "Pedri", "Modrić", "Gavi",
        "Vinicius Jr", "Lewandowski", "Pedri", "Gavi", "Modrić",
        "Vinicius Jr", "Gavi", "Pedri", "Jude Bellingham", "Lewandowski"
    ],
    "Style Preference": [
        "Counter Attack", "Possession", "Counter Attack", "Possession",
        "Counter Attack", "Possession", "Counter Attack", "Possession", "Counter Attack", "Possession",
        "Counter Attack", "Counter Attack", "Possession", "Counter Attack",
        "Possession", "Possession", "Counter Attack", "Possession", "Counter Attack", "Possession",
        "Counter Attack", "Possession", "Possession", "Possession",
        "Counter Attack", "Possession", "Possession", "Counter Attack", "Counter Attack", "Possession"
    ],
    "Age": [
        25, 21, 28, 19, 32, 30, 24, 22, 26, 20,
        31, 23, 21, 27, 20, 29, 26, 24, 33, 19,
        28, 30, 23, 22, 31, 20, 21, 27, 25, 29
    ],
    "Club Support": [
        "Real Madrid", "Barcelona", "Real Madrid", "Barcelona",
        "Real Madrid", "Barcelona", "Real Madrid", "Barcelona", "Real Madrid", "Barcelona",
        "Real Madrid", "Real Madrid", "Barcelona",
        "Real Madrid", "Barcelona", "Barcelona", "Real Madrid",
        "Barcelona", "Real Madrid", "Barcelona",
        "Real Madrid", "Barcelona", "Barcelona", "Barcelona",
        "Real Madrid", "Real Madrid", "Barcelona", "Barcelona", "Real Madrid", "Barcelona"
    ]
}


df = pd.DataFrame(data)
target = "Club Support"
# Replace "Club Support" with Binary values
df[target] = (df[target].map({j: i for i, j in enumerate(df[target].unique().tolist())}))
# One Hot Encoding on other data
df = pd.concat((pd.get_dummies(df[df.columns.drop(target)]), df[target]), axis=1)
# Replace to binary
df = df.replace([False, True], [0, 1])
print(df.head(10))

# Now, we can train this model with Logistic Regression
features = df.columns.drop(["ID", target])
# Split the data into 80% for training and 20% for testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]
# Train with Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Confusion matrix and Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Sample prediction
print("Sample Prediction:")
print("0: Real Madrid, 1: Barcelona")
predict, actual = [], np.array(y_test)
for i in range(len(X_test)):
    sample = X_test.iloc[i]
    predicted_vc = model.predict([sample])
    predict.append(predicted_vc[0])
print("Predicted:", np.array(predict))
print("Actual:", actual)

# Create and display confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real Madrid", "Barcelona"], yticklabels=["Real Madrid", "Barcelona"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()
