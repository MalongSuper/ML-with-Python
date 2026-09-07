# DBSCAN with dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

input_data = "DBSCAN_data.csv"
df = pd.read_csv(input_data)
print(df.head(20))  # The first 20 lines

# Optional: Apply Min-Max Scalar
scaler = MinMaxScaler(feature_range=(0, 1))
df['X_scaled'] = scaler.fit_transform(df[['X']])
df['Y_scaled'] = scaler.fit_transform(df[['Y']])
print(df[['X_scaled', 'Y_scaled']].head(20))
# Split into Training and Testing
X = df[['X_scaled', 'Y_scaled']]

# Apply DBSCAN (Note: Adjust Epsilon to reflect the scaler)
model = DBSCAN(eps=0.1, min_samples=4)
model.fit(X)
DBSCAN_dataset = X.copy()
# Extract cluster labels
DBSCAN_dataset["Cluster"] = model.labels_

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=DBSCAN_dataset, x="X_scaled", y="Y_scaled", hue="Cluster", palette="viridis", s=100)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("DBSCAN Clustering Results")
plt.legend(title="Cluster")
plt.show()
