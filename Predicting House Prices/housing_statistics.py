# Uni-variate selection
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

input_data = "Housing2.csv"
df = pd.read_csv(input_data)
# Convert the columns of ‘Yes/No’ into 1/0
df = df.replace({"yes": 1, "no": 0})
# Remove the last column (furnishingstatus)
df = df.drop(columns="furnishingstatus")

X = df[df.columns.drop('price')]  # Exclude the price
y = df['price']  # It is the target feature
# Alternatively, use:
# X = data.iloc[:,0:20] #independent columns
# y = data.iloc[:,-1] #pick last column for the target feature

best_features = SelectKBest(score_func=chi2, k=5)
fit = best_features.fit(X,y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
scores = pd.concat([df_columns,df_scores],axis=1)
scores.columns = ['specs','score']

# Print the 5 best features
print(scores.nlargest(5,'score'))
