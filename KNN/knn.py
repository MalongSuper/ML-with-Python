import pandas as pd
import numpy as np


def k_nearest_neighbor(df, x_test, x_columns, target, k):
    cols = []
    for i in range(len(x_columns)):
        squared_diff = np.power(x_test[i] - df[x_columns[i]], 2)
        # squared_diff is a series
        cols.append(squared_diff.rename(x_columns[i]))  # keep the name
    # Concatenation
    new_df = pd.concat(cols, axis=1)
    # New sum columns
    new_df['Sum'] = new_df.sum(axis=1)
    # We will find the k smallest values
    nearest_neighbors = new_df.nsmallest(k, 'Sum')
    # Refer to the original df, find the sample satisfies the sample of nearest_neighbors
    neighbors_df = df.loc[nearest_neighbors.index]
    value_counts = neighbors_df[target].value_counts()
    # Why use index.tolist() -> We also need to handle the cases where there are ties
    # in the number of multiple classes
    top_classes = value_counts[value_counts == value_counts.max()].index.tolist()
    return new_df, neighbors_df, top_classes


df = pd.read_csv('load_iris_dataset.csv')
target = 'target'
features = df.drop(columns=target)
k = 3

new_sample = [5, 3.5, 2, 1]
new_df, smallest_df, top_classes = k_nearest_neighbor(df, new_sample, features.columns.tolist(), target, k)
print("Euclidean Distance:\n", new_df)
print("Smallest Distance:", smallest_df)
print("Top Classes (Possible predictions):", top_classes)
