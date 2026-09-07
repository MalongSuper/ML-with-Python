import pandas as pd
import calculate_entropy_infogain as c

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

def recursive_info_gain(df, target_column):
    # Base cases
    if len(df[target_column].unique()) == 1:
        print(f"Leaf node reached → Class: {df[target_column].iloc[0]}")
        return
    
    if len(df.columns) == 1:  # no more features except target
        majority_class = df[target_column].mode()[0]
        print(f"No more features. Majority class: {majority_class}")
        return
    
    # Calculate best feature to split on
    info_gain, best_feature = c.information_gain(df, target_column=target_column)
    if best_feature is None:
        majority_class = df[target_column].mode()[0]
        print(f"No feature gives info gain. Majority class: {majority_class}")
        return

    print(f"Best feature: {best_feature} (Info Gain = {info_gain})")
    
    # Recurse on each branch
    for value in df[best_feature].unique():
        subset = df[df[best_feature] == value].drop(columns=best_feature)
        print(f"\n--- Splitting on {best_feature} = {value} ---")
        print(subset)
        recursive_info_gain(subset, target_column)

# Run the recursive splitting
recursive_info_gain(df, target_column='Go on Trip')
