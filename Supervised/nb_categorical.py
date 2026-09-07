# Naive Bayes Classifier with Scikit-Learn
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

sample_data = {'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
               'Temp': ['Warm', 'Warm', 'Cold', 'Warm'],
               'Humid': ['Normal', 'High', 'High', 'High'],
               'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
               'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
               'Forecast': ['Same', 'Same', 'Change', 'Change'],
               'Play': ['Yes', 'Yes', 'No', 'Yes']}

df = pd.DataFrame(sample_data)

# Replace to numerical values
df['Sky'] = df['Sky'].replace({'Sunny': 1, 'Rainy': 0})
df['Temp'] = df['Temp'].replace({'Warm': 1, 'Cold': 0})
df['Humid'] = df['Humid'].replace({'Normal': 1, 'High': 0})
df['Wind'] = df['Wind'].replace({'Strong': 1})
df['Water'] = df['Water'].replace({'Warm': 1, 'Cool': 0})
df['Forecast'] = df['Forecast'].replace({'Same': 1, 'Change': 0})
df['Play'] = df['Play'].replace({'Yes': 1, 'No': 0})
print(df)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Features and Target
X_train = train_df[df.columns.drop('Play')]
y_train = train_df['Play']
X_test = test_df[df.columns.drop('Play')]
y_test = test_df['Play']


# Train the model with Naive Bayes Classifier
model = CategoricalNB(alpha=1)
model.fit(X_train, y_train)
# Add Accuracy Score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")

for i in range(test_df.shape[0]):
    sample = X_test.iloc[i]
    print("Sample data for testing: \n", sample.tolist())
    # Predict
    predicted_vc = model.predict([sample])
    print(f"Predicted Output(S): {predicted_vc[0]}")
    print(f"Actual Output(S): {y_test.iloc[i]}")
