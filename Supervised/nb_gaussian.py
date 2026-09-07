# Gaussian Naive Bayes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('iris.csv')

target = 'variety'
unique_columns = df[target].unique()
df[target] = df[target].map({unique_columns[i]: i
                             for i in range(len(unique_columns))})

# Display 4 entries for each color
df_group = df.groupby(target).head(4).reset_index()
print(df_group.drop(columns='index'))


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Features and Target
X_train = train_df[df.columns.drop(target)]
y_train = train_df[target]
X_test = test_df[df.columns.drop(target)]
y_test = test_df[target]

# Train the model with Gaussian Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)
# Add Accuracy Score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")
print(f"Classification Report\n {classification_report}")

# Predict 3 samples
for i in range(3):
    sample = X_test.iloc[i]
    print("Sample data for testing: \n", sample.tolist())
    # Predict
    predicted_vc = model.predict([sample])
    print(f"Predicted Output(S): {predicted_vc[0]}")
    print(f"Actual Output(S): {y_test.iloc[i]}")
