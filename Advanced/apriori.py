# Apriori Algorithm
# Install the mlxtend beforehand
# Reference: https://www.geeksforgeeks.org/machine-learning/implementing-apriori-algorithm-in-python/
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('dishes_apriori_dataset.csv')

# Group items purchased together by the same dishes
basket = df.groupby(['Dish'])['Ingredient'].apply(list).reset_index()
transactions = basket['Ingredient'].tolist()

# Apriori needs data in True/False format
# like: Did that ingredient appear in the dish?
# We use Transaction Encoder for this
encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(encoded_array, columns=encoder.columns_)

# Train Apriori Algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
print("Total Frequent Item-sets:", frequent_itemsets.shape[0])

# Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
rules = rules[rules['antecedents'].apply(lambda x: len(x) >= 1) & rules['consequents'].apply(lambda x: len(x) >= 1)]
print("Association Rules:", rules.shape[0])
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Visualize the Most Popular Ingredients
top_items = df['Ingredient'].value_counts().head(10)
top_items.plot(kind='bar', title='Top 10 Most Popular Ingredients')
plt.xlabel("Item")
plt.ylabel("Count")
plt.show()
