# Collaborative Filtering
# Anime Recommendation
# Item and User-Based
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def user_based_recommend_systems(new_df, user, similar_users=2):
    target_anime = df.columns.tolist()
    # Find those who have rated the anime
    user_df = new_df.T
    for j in target_anime:
        watched_users = user_df[user_df[j] != 0].index.tolist()
        # Target Vectors
        target_user_vector = user_df.iloc[user].values.reshape(1, -1)
        similarities = []
        for i in watched_users:
            if i == user:  # skip comparing the user with itself
                continue
            sim = cosine_similarity(target_user_vector, user_df.iloc[i].values.reshape(1, -1))[0][0]
            similarities.append((i, sim))
        # Sort by descending similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[: similar_users]

        # Final Rating Prediction
        numerator = 0
        denominator = 0
        for i, sim in top_k:
            rating = user_df.iloc[i][j]  # mean-centered rating
            numerator += sim * rating
            denominator += abs(sim)
        if denominator != 0:
            predicted_normalized_rating = numerator / denominator
        else:
            predicted_normalized_rating = 0  # fallback if no similar users
        # Add back the column mean
        real_predicted_rating = predicted_normalized_rating + column_mean[j]
        print(f"Predicted real rating of user with index {user} for {j}: {real_predicted_rating}")


def item_based_recommend_systems(new_df, user_index, similar_items=3):
    # new_df: rows = anime, columns = users (mean-centered)
    target_user_ratings = new_df.iloc[:, user_index]
    for anime in new_df.index:
        # Skip anime already rated by the user
        if target_user_ratings[anime] != 0:
            continue
        similarities = []
        # Compare target anime with other anime
        target_anime_vector = new_df.loc[anime].values.reshape(1, -1)
        for other_anime in new_df.index:
            if other_anime == anime:
                continue
            # Only consider anime the user has rated
            if target_user_ratings[other_anime] == 0:
                continue
            sim = cosine_similarity(target_anime_vector,
                                    new_df.loc[other_anime].values.reshape(1, -1))[0][0]
            similarities.append((other_anime, sim))
        # Select top-K similar anime
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:similar_items]
        # Predict normalized rating
        numerator = 0
        denominator = 0
        for other_anime, sim in top_k:
            numerator += sim * target_user_ratings[other_anime]
            denominator += abs(sim)
        if denominator != 0:
            predicted_normalized_rating = numerator / denominator
        else:
            predicted_normalized_rating = 0

        # Add back anime mean
        real_rating = predicted_normalized_rating + column_mean[anime]
        print(f"Predicted real rating for User {user_index} on {anime}: {real_rating}")


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

# Note: Transpose the matrix first
new_df = new_df.T
# Execute the function
user = int(input("Enter a user: "))
user_based_recommend_systems(new_df, user, 3)
print()
# Cons: It does not predict for None rated anime for that user
item_based_recommend_systems(new_df, user, 3)
