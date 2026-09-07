# Collaborative Filtering
# Anime Recommendation
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

anime_ratings = {
    'Oregairu': [5, None, 3, None, 5, 4, None, 2, None, 5, 4, None, 5, 3, None, 4, None, 2, None, 5],
    'Gotoubun': [4, None, 4, None, 5, 3, 2, 2, None, 4, 5, None, 4, 2, None, 5, 2, 3, None, 4],
    'RentGF': [5, 4, None, 4, 5, None, 1, 3, None, None, 4, None, 5, 3, 4, None, 2, 3, None, 5],
    'SoloLeveling': [None, 5, 2, 4, None, 3, 5, None, 5, None, 2, 5, None, 4, 4, None, 5, None, 4, None],
    'ChainsawMan': [None, 4, 3, 5, None, 4, 5, None, 4, None, 3, 4, None, 3, 5, None, 5, None, 4, None],
    'SwordArtOnline': [3, None, None, 4, 2, None, 4, 3, 5, 3, None, 4, 2, None, 5, 3, 5, 4, 3, 2],
    'ReZero': [None, 3, None, None, 1, 3, None, 3, 5, None, 2, 4, None, 3, 4, None, 4, 4, 3, None],
    'OnePiece': [2, 4, None, 5, None, None, 4, 4, None, 2, None, 3, None, 2, None, 3, 4, 4, None, None],
    'Naruto': [3, 5, 2, 4, 1, 3, 5, 4, 5, 3, 2, 5, 1, None, 4, 3, 4, 4, 3, 2],
    'DragonBall': [2, None, 1, None, 2, 4, 3, 3, 4, None, 2, 4, 2, 2, 3, 2, 3, 3, 4, 3]
}


# Mean Normalization
df = pd.DataFrame(anime_ratings)
column_mean = df.mean(axis=0)
new_df = df.subtract(column_mean)
new_df = new_df.fillna(0)
print(new_df)

# Compute cosine similarity between all anime vectors (rows)
# Note: Transpose the matrix first
new_df = new_df.T
similarity_matrix = cosine_similarity(new_df)
anime_names = df.columns.tolist()
# Convert to DataFrame for readable table
similarity_df = pd.DataFrame(similarity_matrix, index=anime_names, columns=anime_names)
print("\nCosine Similarity Matrix:")
print(similarity_df)

# Find those who have rated SoloLeveling
user_df = new_df.T
target_anime = 'SoloLeveling'
watched_users = user_df[user_df[target_anime] != 0].index.tolist()
# Target Vectors
target_user_vector = user_df.iloc[0].values.reshape(1, -1)  # U1
similarities = []
for i in watched_users:
    if i == 0:  # skip comparing U1 with itself
        continue
    sim = cosine_similarity(target_user_vector, user_df.iloc[i].values.reshape(1, -1))[0][0]
    similarities.append((i, sim))
# Sort by descending similarity
similarities.sort(key=lambda x: x[1], reverse=True)
top_k = similarities[:3]

print(f"Top 3 similar users to U1 who rated {target_anime}:")
for i, sim in top_k:
    print(f"User {i + 1} → Similarity: {sim}")

# Final Rating Prediction
numerator = 0
denominator = 0
for i, sim in top_k:
    rating = user_df.iloc[i][target_anime]  # mean-centered rating
    numerator += sim * rating
    denominator += abs(sim)
if denominator != 0:
    predicted_normalized_rating = numerator / denominator
else:
    predicted_normalized_rating = 0  # fallback if no similar users
print(f"\nPredicted normalized rating of U1 for {target_anime}: {predicted_normalized_rating}")

# Add back the column mean
real_predicted_rating = predicted_normalized_rating + column_mean[target_anime]
print(f"Predicted real rating of U1 for {target_anime}: {real_predicted_rating}")
