# Credit Card Detection - Undersampling
# Download the creditcard dataset file here.
# https://drive.google.com/file/d/1ceivr15AnGuVHOjN9AaQGjmlXLWOpcTF/view?usp=sharing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler  # Install this module

df = pd.read_csv('creditcard.csv')

# Take features and target
target = 'Class'
features = df.columns.tolist()
features.remove(target)
X = df[features]
y = df[target]

# Apply Undersampling
rus = RandomUnderSampler(random_state=42)
# Fit and resample the data
X_resampled, y_resampled = rus.fit_resample(X, y)
# Convert dataframe
df_resampled = pd.DataFrame(data=X_resampled, columns=features)
df_resampled[target] = y_resampled

# Return genuine transactions
print(df_resampled[df_resampled['Class'] == 0])
# Return fraud transactions
print(df_resampled[df_resampled['Class'] == 1])
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
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# From your observations, how did undersampling affect the performance?
# Undersampling, by reducing the number of instances in the majority class,
# the model may not have access to the full range of features and patterns present in the data,
# which can result in reduced model accuracy and performance.
# Another potential issue with undersampling is that it can lead to biased models.
