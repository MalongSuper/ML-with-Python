# Density Based Clustering with Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN


sample_data = {"Height": [160, 161, 159, 162, 170, 169, 168, 172, 171, 167,
                   180, 178, 179, 181, 182, 190, 191, 150, 149, 151],
               "Weight": [55, 56, 54, 57, 70, 72, 68, 75, 74, 69,
                   90, 85, 88, 92, 91, 95, 98, 45, 46, 47],
               "Age": [25, 26, 24, 27, 35, 34, 33, 36, 37, 32,
                       45, 42, 44, 46, 47, 50, 52, 20, 19, 21]}

df = pd.DataFrame(sample_data)
print(df)


X = df[['Height', 'Weight', 'Age']]
# Apply DBSCAN
model = DBSCAN(eps=5, min_samples=4)
model.fit(X)
DBSCAN_dataset = X.copy()
# Extract cluster labels
DBSCAN_dataset["Cluster"] = model.labels_

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=DBSCAN_dataset, x="Height", y="Weight", hue="Cluster", palette="viridis", s=100)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("DBSCAN Clustering Results")
plt.legend(title="Cluster")
plt.show()
