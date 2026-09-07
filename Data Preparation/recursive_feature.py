# Recursive Feature Selection Scikit-Learn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

sample_data = {"price": [120, 140, 155, 190, 135, 204, 145, 176, 103, 117],
               "square": [73, 80, 86, 95, 77, 98, 82, 90, 65, 69],
               "bedrooms": [1, 1, 2, 2, 1, 1, 1, 2, 1, 2],
               "bathrooms": [1, 1, 2, 2, 1, 1, 1, 1, 1, 1],
               "stories": [1, 3, 1, 4, 2, 2, 1, 2, 1, 4],
               "guestroom": [0, 1, 0, 0, 0, 1, 1, 0, 1, 1],
               "mainroad": [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]}
# Convert to data frame
df = pd.DataFrame(sample_data)
# Normalize data using MinMaxScaler()
scaler = MinMaxScaler(feature_range=(0, 1))
for i in df:
    if i != "price":
        df[f'{i}_scaled'] = scaler.fit_transform(df[[f'{i}']])
# Create a new dataframe with scaled data
scaled_data = ["price"] + [i for i in df if i.endswith("scaled")]
scaled_df = df[scaled_data]
# Split the data into 80% for training and 20% for testing
df_train, df_test = train_test_split(scaled_df, test_size=0.2, random_state=42)
# Use Recursive Feature Elimination (RFE)
# to keep only 3 features. List those 3 features.
# Estimator Linear Regression
# X are the attributes, y is the target
X_train = df_train[scaled_df.columns.drop('price')]  # Exclude the price
y_train = df_train['price']  # It is the target feature
estimator = LinearRegression()
df_rfe = RFE(estimator=estimator, n_features_to_select=3)
df_rfe.fit(X_train, y_train)
# Update the dataframe to only use the selected features
selected_features = X_train.loc[:, df_rfe.support_]
# Selected feature names
print("Selected features:", selected_features.columns.tolist())
# See rankings of all features
feature_ranks = pd.Series(df_rfe.ranking_, index=X_train.columns)
print("\nFeature rankings (1 = selected):")
print(feature_ranks.sort_values())
