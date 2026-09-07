# Machine Learning Workflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.semi_supervised import LabelPropagation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def clustering_model(df, x, y, clusters=3):
    # Extract relevant features
    data = df[[x, y]]

    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # K-Means
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(data_scaled)

    # Assign clusters
    df['Cluster'] = kmeans.labels_

    # Evaluate clustering
    sil_score = silhouette_score(data_scaled, kmeans.labels_)
    print(f"Silhouette Score: {sil_score:.4f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x], df[y], c=df['Cluster'], cmap='viridis', s=50)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'K-Means Clustering: {x} vs {y}')
    plt.colorbar(label='Cluster')
    plt.show()



def show_info(df, target, features):
    # Get rows and columns
    print(f"+ Rows: {df.shape[0]}; Columns: {df.shape[1]}")
    print(f"+ Unique Labels: {df[target].nunique()}")

    label_counts = {label: count for label, count in df[target].value_counts().items()}
    print(f"+ Value counts of each label\n: {label_counts}")
    print(f"+ Features: {features}")

    # Show info
    df.info()


def get_columns(df, target):
    # Numerical and categorical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Identify binary and multi-class categorical columns
    binary_cols = [col for col in cat_cols if df[col].nunique() == 2 and col != target]
    multi_class_cols = [col for col in cat_cols if df[col].nunique() > 2]

    # Detect nullable integer columns
    nullable_int_cols = []

    for col in num_cols:
        try:
            df[col].astype('Int64')  # Try converting
            nullable_int_cols.append(col)
        except:
            pass


    # Remaining numerical columns
    remaining_num_cols = [col for col in num_cols if col not in nullable_int_cols]

    return num_cols, cat_cols, binary_cols, multi_class_cols, nullable_int_cols, remaining_num_cols


def imputation(df, binary_cols, multi_class_cols, nullable_int_cols, remaining_num_cols):
    # Impute binary columns with mode
    for col in binary_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in multi_class_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Impute nullable integer columns with mode
    for col in nullable_int_cols:
        df[col] = df[col].astype('Int64')
        df[col] = df[col].fillna(df[col].mode()[0])

    # Impute remaining numerical columns with mean
    for col in remaining_num_cols:
        df[col] = df[col].fillna(df[col].mean())


def sampling(df, target):
    # Combine features and target for easier manipulation
    df_combined = df.copy()

    # Separate majority and minority classes
    majority_class = df_combined[df_combined[target] == 'No']
    minority_class = df_combined[df_combined[target] == 'Yes']

    # Oversample minority class
    minority_oversampled = minority_class.sample(n=len(majority_class), replace=True, random_state=42)

    # Combine majority and oversampled minority
    df_balanced = pd.concat([majority_class, minority_oversampled])
    # Shuffle dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced


def train_evaluate_model(model, name, X_train, y_train, X_test, y_test):
    print("Model:", name)
    # Train
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)

    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1score}\n")

    return conf_matrix, {"Model": name, "Accuracy": accuracy, "Precision": precision,
                         "Recall": recall, "F1-Score": f1score}


def plot_metrics(results):
    # Convert the result to dataframe
    df_results = pd.DataFrame(results)
    print(df_results)
    # Set the metrics to plot
    x = df_results['Model']

    # Bar width
    bar_width = 0.2
    r1 = range(len(x))
    r2 = [i + bar_width for i in r1]
    r3 = [i + bar_width * 2 for i in r1]
    r4 = [i + bar_width * 3 for i in r1]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(r1, df_results['Accuracy'], width=bar_width, label='Accuracy')
    plt.bar(r2, df_results['Precision'], width=bar_width, label='Precision')
    plt.bar(r3, df_results['Recall'], width=bar_width, label='Recall')
    plt.bar(r4, df_results['F1-Score'], width=bar_width, label='F1-Score')

    # Add labels and title
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks([r + 1.5 * bar_width for r in range(len(x))], x)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.show()


# Define a function to plot confusion matrices
def plot_confusion_matrix(results, cfm_results):
    # Create subplot grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Loop through each confusion matrix
    for i in range(len(cfm_results)):

        cm = cfm_results[i]
        title = results[i]['Model']

        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, ax=axes[i],
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

        axes[i].set_title(title)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    # Adjust layout
    plt.tight_layout()
    # Show plot
    plt.show()


def label_propagation(df, target):
    # Copy labels
    y_unlabeled = df[target].copy()
    # Remove 80% of labels
    unlabeled_idx = df.sample(frac=0.8, random_state=42).index
    y_unlabeled.loc[unlabeled_idx] = -1

    # Features
    X = df.drop(columns=[target]).values
    y_semi_array = y_unlabeled.values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model
    model = LabelPropagation(kernel='rbf', gamma=20, max_iter=5000)
    model.fit(X_scaled, y_semi_array)

    # Predictions
    y_pred = model.transduction_
    # Evaluation
    y_true = df[target].values

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ARI:", adjusted_rand_score(y_true, y_pred))

    # Plot the labels
    mask = y_semi_array != -1

    y = df[target].values
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c='lightgray', s=30, label='Unlabeled')
    ax[0].scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=y[mask], cmap='viridis', s=60, label='Labeled')
    ax[0].set_title("Before propagation — few labels")
    ax[0].legend()

    ax[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', s=60)
    ax[1].set_title("After propagation — all labeled")

    plt.tight_layout()
    plt.show()


def main():
    # Read the dataset
    df = pd.read_csv('college_placement.csv')
    target = 'Placement'
    features = df.columns.drop('Placement')

    print("-" * 150)
    print(f"1. Dataset Basic Info:")
    show_info(df, target, features)
    print("-" * 150)

    print("-" * 150)
    print("2. Data Preparation: Missing Values")
    (num_cols, cat_cols, binary_cols, multi_class_cols,
     nullable_int_cols, remaining_num_cols) = get_columns(df, target)

    print(f"\n+ Number of Columns: {num_cols}")
    print(f"+ Number of Categorical Columns: {cat_cols}")
    print(f"+ Number of Binary Columns: {binary_cols}")
    print(f"+ Number of Multi-Class Columns: {multi_class_cols}")
    print(f"+ Nullable Integer Columns: {nullable_int_cols}")
    print(f" + Remaining Columns: {remaining_num_cols}")
    print("-" * 150)

    print("-" * 150)
    print("3. Data Preparation: Missing Values Imputation")
    imputation(df, binary_cols, multi_class_cols, nullable_int_cols, remaining_num_cols)
    # Check missing values again
    print(df.isnull().sum())
    # Check data types
    df.info()

    print("-" * 150)
    print("4. Data Preparation: Sampling")
    print(f"Before Sampling:\n {df[target].value_counts()}")
    df_balanced = sampling(df, target)
    print(f"\nAfter Sampling: {df_balanced[target].value_counts()}")
    print("-" * 150)

    print("-" * 150)
    print("5. Data Preparation: Scaling and Encoding")
    # One-Hot Encoding for 'Major'
    # dtype=int makes values 0 and 1 instead of True/False
    df_encoded = pd.get_dummies(df_balanced, columns=['Major'], dtype=int)
    # Map binary column
    df_encoded['Internship_Experience'] = df_encoded['Internship_Experience'].map({'Yes': 1, 'No': 0})
    df_encoded[target] = df_encoded[target].map({'Yes': 1, 'No': 0})
    df_encoded.info()
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]
    # Identify numerical columns (after encoding)
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    # Apply Standardization
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    print("-" * 150)

    print("-" * 150)
    print("6. Supervised Learning: Model Training")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training Shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing Shape: {X_test.shape}, {y_test.shape}")

    models = {
        LogisticRegression(solver='liblinear', random_state=42): 'Logistic Regression',
        DecisionTreeClassifier(criterion='entropy', random_state=42): 'Decision Tree',
        RandomForestClassifier(n_estimators=100, random_state=42): "Random Forest",
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                   random_state=42): "Gradient Boosting"
    }

    results = []
    cfm_results = []

    for model, name in models.items():
        cfm, res = train_evaluate_model(model, name, X_train, y_train, X_test, y_test)
        results.append(res)
        cfm_results.append(cfm)
    print("-" * 150)

    print("-" * 150)
    print("7. Supervised Learning: Model Evaluation")
    plot_metrics(results)
    plot_confusion_matrix(results, cfm_results)
    print("-" * 150)

    # Run clustering
    print("-" * 150)
    print("8. Unsupervised Learning: K-Means")
    clustering_model(df, x='IQ', y='CGPA', clusters=5)
    print("-" * 150)

    print("-" * 150)
    print("9. Semi-Supervised Learning: Label Propagation")
    label_propagation(df_encoded, target)
    print("-" * 150)


main()
