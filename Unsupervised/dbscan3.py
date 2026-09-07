# DBSCAN with dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

input_data = "dataset_clustering.csv"
df = pd.read_csv(input_data)
print(df.head(20))  # The first 20 lines

# Optional: Apply Standard Scaler
scaler = StandardScaler()
X = scaler.fit_transform(df[['Weight', 'Height']])

# Apply DBSCAN (Note: Adjust Epsilon to reflect the scaler)
model = DBSCAN(eps=0.2, min_samples=5)
model.fit(X)
labels = model.fit_predict(X)
# Extract cluster labels
DBSCAN_dataset = pd.DataFrame(X, columns=['Weight_scaled', 'Height_scaled'])
DBSCAN_dataset['Cluster'] = labels

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=DBSCAN_dataset, x="Weight_scaled", y="Height_scaled", hue="Cluster", palette="viridis", s=100)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("DBSCAN Clustering Results")
plt.legend(title="Cluster")
plt.show()
