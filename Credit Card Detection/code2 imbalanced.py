# Credit Card Detection
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler  # Install this module

df = pd.read_csv('creditcard.csv')
# Return genuine transactions
print(df[df['Class'] == 0])
# Return fraud transactions
print(df[df['Class'] == 1])
# Count the number of genuine transactions and fraud transactions
print("Number of Genuine transactions:", (df['Class'] == 0).sum())
print("Number of Fraud transactions:", (df['Class'] == 1).sum())

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Print the shapes of the training and testing sets
print(f"Training set shape: {train_df.shape}")
print(f"Testing set shape: {test_df.shape}")
# Get features name and target (Recommended when the number of features is large)
target = 'Class'
features = df.columns.tolist()
features.remove(target)
# Separate the features (X) and target (y)
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Train a Logistic Regression model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report1 = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report1)


df = pd.read_csv('creditcard.csv')

# Random Sampling: Handling imbalanced dataset
X = df[features]
y = df[target]
# Apply Undersampling -> Reduce the number of samples in the majority class
rus = RandomUnderSampler(random_state=42)
# Fit and resample the data
X_resampled, y_resampled = rus.fit_resample(X, y)
# Convert dataframe
df_resampled = pd.DataFrame(data=X_resampled, columns=features)
df_resampled[target] = y_resampled

# Count the number of genuine transactions and fraud transactions
print("Number of Genuine transactions:", (df_resampled['Class'] == 0).sum())
print("Number of Fraud transactions:", (df_resampled['Class'] == 1).sum())

# Split data into training and testing sets
train_df, test_df = train_test_split(df_resampled, test_size=0.3, random_state=42)
# Print the shapes of the training and testing sets
print(f"Training set shape: {train_df.shape}")
print(f"Testing set shape: {test_df.shape}")
# Get features name and target (Recommended when the number of features is large)
# Separate the features (X) and target (y)
X_resampled_train = train_df[features]
y_resampled_train = train_df[target]
X_resampled_test = test_df[features]
y_resampled_test = test_df[target]

# Train a Logistic Regression model
imbalanced_model = LogisticRegression(solver='liblinear')
imbalanced_model.fit(X_train, y_train)
# Predict and calculate accuracy
y_pred = imbalanced_model.predict(X_resampled_test)
accuracy2 = accuracy_score(y_resampled_test, y_pred)
classification_report2 = classification_report(y_resampled_test, y_pred)
print("Accuracy:", accuracy2)
print("Classification Report:\n", classification_report2)
