# DBSCAN with wine dataset - Multiple Features
# Reference: https://github.com/sumony2j/DBSCAN_Clustering/blob/main/Wine_Dataset_DBSCAN.ipynb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# In this problem, we use the data description to perform DBSCAN
# Convert to only two features X, Y with PCA

input_data = "wine_dataset.csv"
df = pd.read_csv(input_data)
print(df.head(20))

df_described = df.describe().transpose()
print(df_described)  # The first 20 lines

df_corr = df.corr()
print(df_corr)

# Perform Scaling and PCA
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
pca = PCA(n_components=2)
df_scaled = pca.fit_transform(df_scaled)
df_new = pd.DataFrame(df_scaled,columns=['X','Y'])
df_new.describe()

# Apply DBSCAN (Note: Adjust Epsilon to reflect the scaler)
X = df_new[['X', 'Y']]
model = DBSCAN(eps=0.05, min_samples=4)
model.fit(X)
DBSCAN_dataset = X.copy()
# Extract cluster labels
DBSCAN_dataset["Cluster"] = model.labels_


# Draw Correlation Heatmaps
plt.style.use('dark_background')
plt.figure(figsize=(17,8))
sns.heatmap(df.corr(),annot=True,square=True,cmap='tab10')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold')
plt.show()


# Plot the clusters
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=DBSCAN_dataset, x="X", y="Y", hue="Cluster", palette="viridis", s=100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("DBSCAN Clustering Results")
plt.legend(title="Cluster")
plt.show()
