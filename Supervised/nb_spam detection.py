# Naive bayes for Spam Detection
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import ssl
import nltk
import numpy as np
import pandas as pd

# Bypass SSL verification (safe for downloading NLTK resources only)
try:
    create_unverified_https_context = ssl.create_unverified_context
    ssl.create_default_https_context = create_unverified_https_context
except AttributeError:
    pass

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
# Define stopwords and English words
stop_words = set(stopwords.words('english'))
wordlist = set(nltk.corpus.words.words())


def preprocess(message):
    words = message.lower()  # lowercase
    words = word_tokenize(words)  # tokenization
    words = [word for word in words if len(word) > 1]
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if word in wordlist]  # english words
    words = [PorterStemmer().stem(word) for word in words]  # stemming
    return words


df = pd.read_csv('spam.csv', encoding='ISO-8859-1', usecols=[0, 1])
print("The first 20 records of the dataset:\n", df.head(20))

# Divide the dataset into Training (80%) and Testing (20%).
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training set shape: {train_df.shape}")
print(f"Testing set shape: {test_df.shape}")

# build a dictionary to count the frequencies of words in Spam dataset and
# another dictionary to count the frequencies of words in Non-Spam/Ham dataset.
# Stored as list, then return as dict
spam_words = []
ham_words = []

# Loop through training data
for index, row in train_df.iterrows():
    # Get the label and the message
    label, message = row['v1'], row['v2']
    # Processing the word
    words = preprocess(message)
    # If the next label is spam, append the word to spam_words
    # Else, append the word to ham_words
    for word in words:
        if label == 'spam':
            spam_words.append(word)
        else:
            ham_words.append(word)

# Return the counter as dictionary
spam_word_counts = dict(Counter(spam_words))
ham_word_counts = dict(Counter(ham_words))
print("Spam Word Counts:\n", spam_word_counts)
print("Ham Word Counts:\n", ham_word_counts)

# Get the unique word from both spam_words and ham_words
unique_words = list(set(spam_words + ham_words))
word_index = {word: idx for idx, word in enumerate(unique_words)}
print("Unique Words:\n", word_index)


# Turn message into vector based on word presence
# For example: If the word_index = {"a": 0; "b": 1; "c": 2}
# Message: "A A" -> Prepossessing: ["a", "a"]
# -> Vector = [2, 0, 0]
def message_to_vector(message):
    # Create a zero vector with the length of unique words
    # Length of the unique words and word_index is the same
    vector = np.zeros(len(unique_words), dtype=int)
    # Iterate through every message
    words = preprocess(message)
    # If the word appears in word index for that message
    for word in words:
        # word_index[word] get the index of the word in the word_index dict
        # Increment the position to 1, indicating the number of times that word appear
        # The other word not featured in the message, stays at 0
        if word in word_index:
            # Vector[word_index[word]] = 1 for you want binary presence
            vector[word_index[word]] = 1
    return vector, words


print("Message into Vector:")
# Print the first 10 records to showcase
for index, row in train_df.head(10).iterrows():
    label, message = row['v1'], row['v2']
    vector, words = message_to_vector(message)
    print("Label:", label, "\nMessage:", message)
    for word in words:
        # Get only the words in the message
        # Their values in the vector must be 1
        vector_dict = {word: vector[word_index[word]]}
        print(vector_dict, end=", ")


# Now, train the model with CategoricalNB
# The X for training and testing are the vectors in the previous function
# The y for training and testing is the target feature, 1 for spam, 0 for not spam
X_train = np.array([message_to_vector(msg)[0] for msg in train_df['v2']])
y_train = [1 if label == 'spam' else 0 for label in train_df['v1']]
X_test = np.array([message_to_vector(msg)[0] for msg in test_df['v2']])
y_test = [1 if label == 'spam' else 0 for label in test_df['v1']]

model = CategoricalNB(alpha=1)
model.fit(X_train, y_train)

# When testing, ignore words that are NOT in the training set.
# Already defined in the message_to_vector(message)
# Example: word_index = {"win": 0, "money": 1, "free": 2, "offer": 3}
# The word_index comes from the unique words from (spam_words + ham_words)
# Now test message: "Claim your free vacation now"
# Preprocessed: ["claim", "free", "vacation"]; "claim" and "vacation" are not in the training vocabulary.
# Only "free" is kept. Resulting vector: [0, 0, 1, 0]
# "claim" and "vacation" are ignored

# What’s the accuracy, Recall, Precision, F1?
y_pred = model.predict(X_test)
print()
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Testing Demo
for i in range(test_df.shape[0]):
    sample = X_test[i]  # Selecting any testing sample
    # Predict
    predicted_vc = model.predict([sample])
    print()
    print(f"Message: {test_df.iloc[i, 1]}")
    print(f"Predicted Vc: {'Spam' if predicted_vc[0] == 1 else 'Not Spam'}")
    print(f"Actual Vc: {'Spam' if y_test[0] else 'Not Spam'}")
