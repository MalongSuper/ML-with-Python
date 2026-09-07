# Credit Card Fraud Detection: Train using four models
# Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import time

# Step 1: Read the dataset
df = pd.read_csv('creditcard.csv')
print(df.head(50))

# Take features and target
target = 'Class'
features = df.columns.tolist()
features.remove(target)
X = df[features]
y = df[target]

# Step 2: Undersampling the non-fraud to match the fraud
# 50% fraud and 50% Non-Fraud -> sampling_strategy = 5/5 = 1
# 100% Balance between classes
rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
# New dataset
df_resampled = pd.DataFrame(data=X_resampled, columns=features)
df_resampled[target] = y_resampled

# Step 3: 70% Training / 15% Validation / 15% Testing Data
train_df, val_test_df = train_test_split(df_resampled, test_size=0.3, random_state=42)
eval_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)

# Separate the features
X_train, y_train = train_df[features], train_df[target]
X_eval, y_eval = eval_df[features], eval_df[target]
X_test, y_test = test_df[features], test_df[target]


# Step 4: Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled = scaler.transform(X_eval)
X_test_scaled = scaler.transform(X_test)


# Step 5: Define a function to train a model using ML techniques
def train_evaluate_model(model, name):
    print("Model Used:", name)
    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    end_train = time.time()
    print(f"Training Complete! Training Time: {end_train - start_train}")
    # Use Validation Data -> Mock Test
    start_eval = time.time()
    y_pred = model.predict(X_eval_scaled)
    conf_matrix_eval = confusion_matrix(y_eval, y_pred)
    accuracy_eval = accuracy_score(y_eval, y_pred)
    precision_eval = precision_score(y_eval, y_pred)
    recall_eval = recall_score(y_eval, y_pred)
    f1score_eval = f1_score(y_eval, y_pred)
    end_eval = time.time()
    print(f"Validation Complete! Validation Time: {end_eval - start_eval}")
    print(f"Confusion Matrix:\n {conf_matrix_eval}\nAccuracy: {accuracy_eval} "
          f"\nPrecision: {precision_eval} \nRecall {recall_eval} \nF1-Score {f1score_eval}")
    # Testing model
    start_test = time.time()
    y_pred = model.predict(X_test_scaled)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    end_test = time.time()
    print(f"Testing Complete! Testing Time: {end_test - start_test}")
    print(f"Confusion Matrix:\n {conf_matrix}\nAccuracy: {accuracy} "
          f"\nPrecision: {precision} \nRecall {recall} \nF1-Score {f1score}")
    # Return a dictionary
    return {"Model": name, "Accuracy": accuracy, "Precision": precision,
            "Recall": recall, "F1-Score": f1score}


def make_prediction(model):  # Predict the first 10 samples
    sample = X_test_scaled[:10]
    actual = y_test[:10]
    predicted = model.predict(sample)
    print("Predicted Labels:", predicted)
    print("True Labels:", actual.values)


# Function to visualize model comparison
def plot_metrics(results):
    df_results = pd.DataFrame(results)
    df_results.set_index('Model', inplace=True)

    # Plot each metric in a grouped bar chart
    df_results.plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def main():
    models = {KNeighborsClassifier(n_neighbors=10): "KNN",
             DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42): "Decision Tree",
             RandomForestClassifier(n_estimators=100, random_state=42): "Random Forest",
             GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                        random_state=42): "Gradient Boosting"}
    results = []
    for model, name in models.items():
        res = train_evaluate_model(model, name)
        results.append(res)
        make_prediction(model)

    plot_metrics(results)


main()
