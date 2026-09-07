# Collaborative Filtering
# Anime Recommendation
import pandas as pd

ratings = {
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


def cosine_similarity(a, b):
    dot_ab = sum([a[i] * b[i] for i in range(len(a))])
    dot_a = sum([a[i] * a[i] for i in range(len(a))])
    dot_b = sum([b[i] * b[i] for i in range(len(b))])
    cosine = dot_ab / ((dot_a ** 0.5) * (dot_b ** 0.5))
    return cosine


# Normalized (by calculation)
anime_ratings = list(ratings.values())
anime_names = list(ratings.keys())
new_anime_ratings = []
average = []
# This returns the matrix of all the corresponding lists, but remove Nones
for i in range(len(anime_ratings)):
    not_none = [anime_ratings[i][j] for j in range(len(anime_ratings[i]))
           if anime_ratings[i][j] is not None]
    new_anime_ratings.append(not_none)

# This returns the list of average
for j in range(len(new_anime_ratings)):
    avg = sum(new_anime_ratings[j]) / len(new_anime_ratings[j])
    average.append(avg)

# This subtracts every value in the list with the corresponding average
final_result = []
result = []  # Acts as a temporal list
for k in range(len(average)):
    for p in range(len(new_anime_ratings[k])):
        # Subtract the value with the average
        res = new_anime_ratings[k][p] - average[k]
        result.append(res)
        # If the length of the result reach the length of the current list
        if len(result) == len(new_anime_ratings[k]):
            final_result.append(result)  # Append that list to the final result
            result = []  # Reset the result list


print("Normalization: Mean-Based")
for key, value in enumerate(final_result):
    print(f"{anime_names[key]}:", value)

print("\nCosine Similarity")
# Fill the remaining values with 0 - For Cosine Similarity
final_matrix = []
for row in final_result:
    padded_row = row + [0] * (20 - len(row))
    final_matrix.append(padded_row)

# Cosine Similarity
similarity_matrix = []
for i in range(len(final_matrix)):
    row = []
    for j in range(len(final_matrix)):
        row.append(round(cosine_similarity(final_matrix[i], final_matrix[j]), 4))
    similarity_matrix.append(row)

# Display as DataFrame
df = pd.DataFrame(similarity_matrix, index=anime_names, columns=anime_names)
print("\nCosine Similarity Matrix:")
print(df)
