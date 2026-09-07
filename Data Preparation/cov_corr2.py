# Covariance Matrix and Correlation Matrix with Python
# Heatmaps
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sample_data = {
    'Math': [84, 82, 81, 89, 73, 94, 92, 70, 88, 95],
    'Science': [85, 82, 72, 77, 75, 89, 95, 84, 77, 94],
    'History': [97, 94, 93, 95, 88, 82, 78, 84, 69, 78],
    'Geography': [88, 85, 90, 92, 80, 79, 84, 75, 89, 91],
    'English': [90, 88, 86, 85, 84, 93, 87, 78, 91, 89],
    'Biology': [82, 79, 85, 88, 81, 90, 92, 76, 83, 87],
    'Chemistry': [79, 85, 80, 83, 78, 91, 94, 77, 81, 90],
    'Physics': [84, 86, 88, 90, 79, 88, 93, 74, 85, 92]
}

# Convert to data frame
df = pd.DataFrame(sample_data)
cov_matrix = df.cov()
corr_matrix = df.corr()
print("Covariance Matrix:\n", cov_matrix)
print("Correlation Matrix:\n", corr_matrix)

# Set up the matplotlib figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Draw the correlation heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[0])
axes[0].set_title("Correlation Heatmap", fontsize=14)
# Draw the covariance heatmap
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[1])
axes[1].set_title("Covariance Heatmap", fontsize=14)
# Adjust layout and show plot
plt.tight_layout()
plt.show()
