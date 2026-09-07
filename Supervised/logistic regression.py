# Logistic Regression
import math
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def sigmoid(x):
    fx = 1 / (1 + math.exp(-x))  # Using math.exp for exponentiation
    return fx


def odd(p):
    if p <= 0 or p >= 1:
        raise ValueError("Probability must be between 0 and 1 (exclusive).")
    return p / (1 - p)


def odd_ratio(p1, p2):
    return odd(p1) / odd(p2)


def decision(predicted_p):
    if predicted_p[0][0] > predicted_p[0][1]:
        return False
    elif predicted_p[0][0] < predicted_p[0][1]:
        return True
    else:
        return "Undefined"


# X1: Suspicious Words;	X2: Contains Links;
# X3: Email Length; X4: Label (Spam or Not Spam)
sample_data = {'Suspicious Words': [5, 2, 7, 1],
               'Contains Links': [1, 0, 1, 0],
               'Email Length': [200, 50, 350, 30],
               'Label': [1, 0, 1, 0]}
# Convert to data frame
df = pd.DataFrame(sample_data)
# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Print the shapes of the training and testing sets
print(f"Training set shape: {train_df.shape}")
print(f"Testing set shape: {test_df.shape}")
# Separate the features (X) and target (y)
X_train = train_df[['Suspicious Words', 'Contains Links', 'Email Length']]
y_train = train_df['Label']
X_test = test_df[['Suspicious Words', 'Contains Links', 'Email Length']]
y_test = test_df['Label']
# Train a Logistic Regression model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
# Sample prediction with testing data
sample = X_test.iloc[0].to_frame().T  # Transpose Series to 1-row DataFrame
# Get predicted probabilities
predicted_p = model.predict_proba(sample)
print(f"Not Spam: {predicted_p[0][0]}; Spam: {predicted_p[0][1]}")
print(f"Conclusion: {'Not Spam' if decision(predicted_p) is False else 'Spam'}")
# Calculate odd and odd_ratio
print(f"- Odd Spam: {odd(predicted_p[0][1])}")
print(f"- Odd Not Spam: {odd(predicted_p[0][0])}")
# Odds Ratio
print(f"- Odd Ratio: {odd_ratio(predicted_p[0][1], predicted_p[0][0])}")
