The function k_nearest_neighbor() implements a manual version of the K-Nearest Neighbors (KNN) algorithm for classification. 
It calculates distances between a test sample and all samples in the dataset, then predicts the class based on the k nearest points.

## Parameters
- df – The full training dataset stored as a Pandas DataFrame.
- It must include all feature columns and the target (class) column.
- x_test – A list or array of numeric values representing the test sample (the new observation to classify).
Its length must match the number of feature columns.
- x_columns – A list of column names (strings) corresponding to the feature columns used for distance calculation.
- target – The name of the target (label) column in df that contains class values.
- k – The number of nearest neighbors to consider when making the prediction.

## Workflow / Logic

- Compute Squared Differences: For each feature in x_columns, compute the squared difference between the test value (x_test[i]) and the dataset column df[x_columns[i]]. Each result is a Pandas Series.

- Combine into One DataFrame: All squared difference Series are concatenated column-wise into a single DataFrame (new_df) with the same number of rows as the original dataset.

- Calculate the Total Distance: A new column "Sum" is added to store the sum of squared differences per row — representing the Euclidean distance (without the square root) between each training sample and the test sample.

- Find the K Smallest Distances: The k samples with the smallest "Sum" values are selected as the nearest neighbors.

- Locate the Neighboring Samples: Using their indices, the corresponding rows are retrieved from the original df into neighbors_df.

- Count Class Occurrences: The function counts the occurrences of each class label in the neighbors (value_counts).

- Handle Ties in Class Counts: If multiple classes appear the same maximum number of times, it collects all of them into a list (top_classes) to handle tie cases.

## Returns
- new_df → A DataFrame containing all squared differences per feature and the "Sum" distance column.
- neighbors_df → A subset of df containing the k nearest samples.
- top_classes → A list of the most common class(es) among those k neighbors (the prediction).
