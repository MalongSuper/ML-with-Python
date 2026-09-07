# Data Preparation with dataset.csv
import pandas as pd


data = {'WorkerID': [1, 2, 3, 4, 5], 'Age': [25, None, 35, 30, 24],
        'Salary': [50000, 54000, None, 58000, 45000]}
data = pd.DataFrame(data)
# Row Deletion
data_row_deleted = data.dropna()
# Column Deletion
data_col_deleted = data.dropna(axis=1)
print(data_row_deleted)
print(data_col_deleted)

# Dataset.csv
df = pd.read_csv('dataset.csv')
# Row Deletion
df_row_deleted = df.dropna()
# Column Deletion
df_col_deleted = df.dropna(axis=1)
print(df_row_deleted)
print(df_col_deleted)
