import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from seaborn import heatmap
import matplotlib.pyplot as plt

# Load the dataset
input_data = "Housing2.csv"
df = pd.read_csv(input_data)
# Convert the columns of ‘Yes/No’ into 1/0
df = df.replace({"yes": 1, "no": 0})
print(df)
# Remove the last column (furnishingstatus)
df = df.drop(columns="furnishingstatus")
print("Remove furnishingstatus")
print(df)
# Normalize data using MinMaxScaler()
scaler = MinMaxScaler(feature_range=(0, 1))
# Apply scaling to the every column except "price"
df['area_scaled'] = scaler.fit_transform(df[['area']])
df['bedrooms_scaled'] = scaler.fit_transform(df[['bedrooms']])
df['bathrooms_scaled'] = scaler.fit_transform(df[['bathrooms']])
df['stories_scaled'] = scaler.fit_transform(df[['stories']])
df['mainroad_scaled'] = scaler.fit_transform(df[['mainroad']])
df['guestroom_scaled'] = scaler.fit_transform(df[['guestroom']])
df['basement_scaled'] = scaler.fit_transform(df[['basement']])
df['hotwaterheating_scaled'] = scaler.fit_transform(df[['hotwaterheating']])
df['airconditioning_scaled'] = scaler.fit_transform(df[['airconditioning']])
df['parking_scaled'] = scaler.fit_transform(df[['parking']])
df['prefarea_scaled'] = scaler.fit_transform(df[['prefarea']])
print("Data after scaled")
scaled_data = ['price', 'area_scaled', 'bedrooms_scaled', 'bathrooms_scaled',
          'stories_scaled', 'mainroad_scaled', 'guestroom_scaled',
          'basement_scaled', 'hotwaterheating_scaled', 'airconditioning_scaled',
          'parking_scaled', 'prefarea_scaled']
# Print the updated DataFrame
scaled_df = df[scaled_data]
print(scaled_df)
# Split the data into 80% for training and 20% for testing
df_train, df_test = train_test_split(scaled_df, test_size=0.2, random_state=42)
# Print the shapes of the training and testing sets
print(f"Training set shape: {df_train.shape}")
print(f"Testing set shape: {df_test.shape}")
# Use Recursive Feature Elimination (RFE)
# to keep only 5 features. List those 5 features.
# Estimator Linear Regression
# X are the attributes, y is the target
X_train = df_train[scaled_df.columns.drop('price')]  # Exclude the price
y_train = df_train['price']  # It is the target feature
estimator = LinearRegression()
df_rfe = RFE(estimator=estimator, n_features_to_select=5)
df_rfe.fit(X_train, y_train)
# Update the dataframe to only use the selected features
selected_features = X_train.loc[:, df_rfe.support_]
# Selected feature names
print("Selected features:", selected_features.columns.tolist())
# See rankings of all features
feature_ranks = pd.Series(df_rfe.ranking_, index=X_train.columns)
print("Feature rankings (1 = selected):")
print(feature_ranks.sort_values())
# 5 correlation coeﬃcients from the correlation matrix
corr_matrix = df_train.corr()
corr_with_price = corr_matrix['price'].drop('price')  # drop 'price'
# Get the absolute value, sort them and then print the top 5
corr_coeff = corr_with_price.abs().sort_values(ascending=False)
print("Correlation Coefficients ranking")
print(corr_coeff)
print("Top 5 correlation coefficients:", corr_coeff.head(5).index.tolist())
# High: > 0.75, Low: < 0.3
corr_coeff_value = corr_coeff.values.tolist()
for i in range(len(corr_coeff_value)):
    if corr_coeff_value[i] > 0.75:
        print(corr_coeff_value[i], ": High")
    elif corr_coeff_value[i] < 0.3:
        print(corr_coeff_value[i], ": Low")
    else:
        print(corr_coeff_value[i], ": Moderate")

# Now, apply Linear Regression for only the top features
X_train_features = df_train[selected_features.columns.tolist()]
y_train_features = y_train
model = LinearRegression()
model.fit(X_train_features, y_train_features)
# Get the linear equation
A = model.intercept_
B1, B2, B3, B4, B5 = (model.coef_[0], model.coef_[1],
                      model.coef_[2], model.coef_[3],
                      model.coef_[4])
print(f"Linear Equation: Y = {A} + ({B1} * X1) + ({B2} * X2) + "
      f"({B3} * X3) + ({B4} * X4) + ({B5} * X5)")
# Use testing data
X_test = df_test[selected_features.columns.tolist()]  # Exclude the price
y_test = df_test['price']  # It is the target feature
# Make predictions on the test set
y_predict = model.predict(X_test)
# Compute the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_predict)
print(f"Mean Squared Error on the test set: {mse}")
# Compute the R² score
r2 = r2_score(y_test, y_predict)
print(f"R² Score on the test set: {r2}")

# Draw a heat map for the correlation matrix (12 x 12) of training data.
heatmap(corr_matrix, annot=True, cmap="YlGnBu")
# Display the heatmap
plt.show()
