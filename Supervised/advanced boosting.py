# Advanced Boosting techniques: XGBoost, LightGBM, CatBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostClassifier
import pandas as pd

# Load the dataset:
df = pd.read_csv('breast_cancer_wisconsin.csv')
target = 'diagnosis'
features = df.columns.drop(target)
# Label Encoding for the target
df[target] = df['diagnosis'].map({"B": 0, "M": 1})

# 80% training and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df[features]
X_test = test_df[features]
y_train = train_df[target]
y_test = test_df[target]

# Initialize XGBClassifier
model_xgb = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
# Fit the model (test sets are used for early stopping)
model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)])
y_pred_xgb = model_xgb.predict(X_test)
print("Accuracy (XGBoost):", accuracy_score(y_test, y_pred_xgb))


# Initialize LGBMClassifier
model_lgbm = lgbm.LGBMClassifier(objective='binary', random_state=42)
# Fit the model (test sets are used for early stopping)
model_lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)])
# Make predictions
y_pred_lgbm = model_lgbm.predict(X_test)
print("Accuracy (LightGBM):", accuracy_score(y_test, y_pred_lgbm))


# Initialize CatBoost
model_cb = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, random_seed=42)
# Fit the model (test sets are used for early stopping)
model_cb.fit(X_train, y_train, eval_set=[(X_test, y_test)])
# Make predictions
y_pred_cb = model_cb.predict(X_test)
print("Accuracy (CatBoost):", accuracy_score(y_test, y_pred_cb))
