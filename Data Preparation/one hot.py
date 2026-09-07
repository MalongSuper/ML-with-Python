# One-Hot Encoding
import numpy as np
import pandas as pd

sample_data = {"Color": ['red', 'yellow', 'blue', 'green', 'pink', 'purple',
         'black', 'white', 'brown', 'grey', 'orange']}

# Convert to data frame
df = pd.DataFrame(sample_data)
encoded_df = pd.get_dummies(df)
print("Features:\n", df)
print("One-Hot Encoding:\n", encoded_df)
