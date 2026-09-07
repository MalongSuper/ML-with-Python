# Training and Testing Data
import pandas as pd
from sklearn.model_selection import train_test_split

sample_data = {"price": [120, 140, 155, 190, 135, 204, 145, 176, 103, 117],
               "square": [73, 80, 86, 95, 77, 98, 82, 90, 65, 69],
               "bedrooms": [1, 1, 2, 2, 1, 1, 1, 2, 1, 2],
               "bathrooms": [1, 1, 2, 2, 1, 1, 1, 1, 1, 1]}
# Convert to data frame
df = pd.DataFrame(sample_data)
# Split the data into 80% for training and 20% for testing
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
# Print the training and testing sets
print(f"Training Data:\n {df_train}")
print(f"Testing set shape:\n {df_test}")
