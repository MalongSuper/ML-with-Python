# Data Preparation with dataset.csv
import pandas as pd

# Get dataset
df = pd.read_csv('dataset.csv')
# Get shape
row, column = df.shape
# Handle dataset
print("Dataset.csv")
print(df)  # Print everything
# Check if the data is not there with df.isnull()
print("\na) Find the number of missing data in all columns.")
d_null = df.isnull()  # True if exists, False if not exist
sum_of_null = df.isnull().sum().sum()
print(f"Total number of missing values: {sum_of_null}")
print(f"Percentage: {round((sum_of_null / df.size) * 100)}%")
print(f"\n {d_null.head(10)}")  # Print the first 10 row
print("\nb) Fill in the missing values with mean.")
# mean = df.sum() / df.count()
mean = df.mean()
d_fillmean = df.fillna(mean)
print(f"\n {d_fillmean.head(10)}")  # Print the first 10 row
print("\nc) Fill in the missing values with median.")
median = df.median()
d_fillmedian = df.fillna(median)
print(f"\n {d_fillmedian.head(10)}")  # Print the first 10 row
print("\nd) Fill in the missing values with mode.")
# In case there are multiple nodes, iloc[0] selects the first node found
mode = df.mode().iloc[0]
d_fillmode = df.fillna(mode)
print(f"\n {d_fillmode.head(10)}")  # Print the first 10 row
print("\ne) What is the diﬀerence between mean, median and mode?")
print(f"- Mean: The average value of the sequence\n"
      f"- Median: The middle value of the sorted sequence \n- Mode: The most-occurred value of the sequence")
print("\nf) Drop columns with a missing value rate higher than 30%. ")
d_count = (df.isnull().sum() / row) * 100
drop_columns = d_count.index[d_count > 30].tolist()
d_drop_columns = df.drop(columns=drop_columns)
print(f"\n {d_drop_columns.head(10)}")
print("\nf) Drop rows with a missing value rate higher than 40%. ")
d_count = (df.isnull().sum(axis=1) / column) * 100
drop_rows = d_count.index[d_count > 40].tolist()
d_drop_rows = df.drop(index=drop_rows)
print(f"\n {d_drop_rows.head(10)}")
print("\ng) Normalize data columns using minmax normalization.")
d_normal = (df - df.min()) / (df.max() - df.min())
print(f"\n {d_normal.head(10)}")
print("\nh) Dropping the outlier rows with 2 * standard deviation away from the mean.")
# Find the mean of each row, then compare every element with [u + (2 * sd); u - (2 * sd)]
# If any element in the row is not within this interval, that row is dropped
row_mean = df.mean(axis=1)
row_sd = df.std(axis=1)
lower_bound = row_mean - (2 * row_sd)
upper_bound = row_mean + (2 * row_sd)
# Transpose the data and compare it with the interval, then transpose them again
outlier = (df.T < lower_bound).T | (df.T > upper_bound).T
outlier_row = outlier.any(axis=1)  # Retrieve those outlier rows, they are True
drop_outlier_row = df.drop(index=df[outlier_row].index.tolist())
print(f"\n {drop_outlier_row}")
