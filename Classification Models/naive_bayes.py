import pandas as pd
import calculate_probability as p


# Sample data
data = {
    "No": [1, 2, 3, 4, 5, 6, 7, 8],
    "Money": ["High", "High", "Low", "High", "High", "High", "High", "Low"],
    "In a Relationship": ["Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "No"],
    "Weather": ["Sunny", "Good", "Stormy", "Sunny", "Light Rain", "Good", "Stormy", "Sunny"],
    "Duration": ["Long", "Medium", "Short", "Long", "Medium", "Short", "Long", "Medium"],
    "Transportation": ["Easy", "Easy", "Hard", "Easy", "Hard", "Easy", "Hard", "Easy"],
    "Go on Trip": ["Yes", "Yes", "No", "Yes", "No", "Yes", "No", "No"]
}

df = pd.DataFrame(data).drop(columns='No')
target = 'Go on Trip'
prob_target = p.calculate_priors(df, target)
prob_feature_given_class = p.calculate_conditionals(df, target, prob_target)

print("-" * 50)
print("Summary:")
for k, v in prob_target.items():
    print(f"P({target} = {k}) = {v}")
print()
for i, j in prob_feature_given_class.items():
     print(f"P({i[0]} = {i[1]} | {i[2]}) = {j}")
print("-" * 50)