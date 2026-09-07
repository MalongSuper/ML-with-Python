# Z-Score Scalar with Scikit-Learn
import pandas as pd
from sklearn.preprocessing import StandardScaler

sample_data = {"price": [120, 140, 155, 190, 135, 204, 145, 176, 103, 117],
               "square": [73, 80, 86, 95, 77, 98, 82, 90, 65, 69],
               "bedrooms": [1, 1, 2, 2, 1, 1, 1, 2, 1, 2],
               "bathrooms": [1, 1, 2, 2, 1, 1, 1, 1, 1, 1]}
# Convert to data frame
df = pd.DataFrame(sample_data)
# Normalize data using MinMaxScaler()
scaler = StandardScaler()
df['square_scaled'] = scaler.fit_transform(df[['square']])
print(df[['square', 'square_scaled']])
