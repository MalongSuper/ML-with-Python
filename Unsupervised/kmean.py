# K-Mean Clustering for Housing
# Unsupervised Learning: No data split needed
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv('datasets/housingforKMeans.csv',
                 usecols=['longitude', 'latitude', 'median_house_value'])
# The first 20 rows
print(df.head(20))
# Cluster by geography
X = df[['longitude', 'latitude']]
scaler = StandardScaler()  # Normalize the data
X_scaled = scaler.fit_transform(X)
# Fitting the model
model = KMeans(n_clusters=3, random_state=0, n_init='auto')
model.fit(X_scaled)

# Visualize the original data
sns.scatterplot(data=df, x='longitude', y='latitude',
                hue='median_house_value', palette='viridis')
plt.show()

# Visualize the fit data
df['cluster'] = model.labels_
sns.scatterplot(data=df, x='longitude', y='latitude',
                hue='cluster', palette='viridis')
plt.show()
