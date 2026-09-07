# Uni-variate selection
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

sample_data = {"price": [120, 140, 155, 190, 135, 204, 145, 176, 103, 117],
               "square": [73, 80, 86, 95, 77, 98, 82, 90, 65, 69],
               "bedrooms": [1, 1, 2, 2, 1, 1, 1, 2, 1, 2],
               "bathrooms": [1, 1, 2, 2, 1, 1, 1, 1, 1, 1],
               "stories": [1, 3, 1, 4, 2, 2, 1, 2, 1, 4],
               "guestroom": [0, 1, 0, 0, 0, 1, 1, 0, 1, 1],
               "mainroad": [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]}

df = pd.DataFrame(sample_data)

X = df[df.columns.drop('price')]  # Exclude the price
y = df['price']  # It is the target feature
best_features = SelectKBest(score_func=chi2, k=5)
fit = best_features.fit(X,y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
scores = pd.concat([df_columns,df_scores],axis=1)
scores.columns = ['specs','score']

# Print the 5 best features
print(scores.nlargest(5,'score'))
