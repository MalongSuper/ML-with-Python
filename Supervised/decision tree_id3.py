import pandas as pd
from math import log2


def entropy_target(target):
    p_list = []
    n = target.unique()  # Number of unique outcomes
    for i in range(len(n)):  
        # Calculate the p(class) of the outcome
        p = (target.value_counts().get(n[i], 0)) / (len(target))
        print(f"P(Target = {target.name} -> Outcome {n[i]}): {p}")
        p_list.append(p) 
    # Apply the formula (add p > 0 to handle entropy = 0)
    entropy = sum(- p * log2(p) for p in p_list if p > 0)
    print(f"E(Target = {target.name}): {entropy}")

    return entropy


def entropy_feature(feature, target):
    df_combined = pd.concat([feature, target], axis=1)
    entropy_list = []
    n_feature = feature.unique()  # Number of unique classes
    n_target = target.unique()  # Number of unique outcomes
    print("Feature =", feature.name)
    for i in range(len(n_feature)):
        p_list = []
        for j in range(len(n_target)):
            # Total number of samples for the class
            num_class = feature.value_counts().get(n_feature[i], 0)
            # Count the outcome for the current class, get the sub-dataset
            sub_df = df_combined[(df_combined[feature.name] == n_feature[i]) 
                                 & (df_combined[target.name] == n_target[j])]
            # Find the probability
            p = len(sub_df) / num_class if num_class != 0 else 0
            print(f"+ P({n_feature[i]} -> Outcome = {n_target[j]}): {p}")
            p_list.append(p)
        # Compute the entropy
        entropy = sum(-p * log2(p) for p in p_list if p > 0)
        print(f"- Entropy(Feature Class = {n_feature[i]}): {entropy}")
        # Append to list
        entropy_list.append(entropy)

    return entropy_list


data = {"Serial No.": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "Outlook": ["Sunny", "Sunny", "Overcast", "Rainy", "Rainy", "Rainy",
                "Overcast", "Sunny", "Sunny", "Rainy", "Sunny", "Overcast",
                "Overcast", "Rainy"],
        "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool",
                    "Cool", "Mild", "Cool", "Mild", "Mild", "Mild",
                    "Hot", "Mild"],
        "Humidity": ["High", "High", "High", "High", "Normal", "Normal",
                 "Normal", "High", "Normal", "Normal", "Normal", "High",
                 "Normal", "High"],
        "Windy": [False, True, False, False, False, True,
              True, False, False, False, True, True,
              False, True],
        "Play": ["No", "No", "Yes", "Yes", "Yes", "No",
             "Yes", "No", "Yes", "Yes", "Yes", "Yes",
             "Yes", "No"]}
df = pd.DataFrame(data).drop(columns=['Serial No.'])

for i in df['Outlook'].unique():
    print(df[df['Outlook'] == i])


# Mapping
for i in range(len(df.columns)):
    unique_columns = df[df.columns[i]].unique()
    df[df.columns[i]] = df[df.columns[i]].map({unique_columns[value]: value for value in range(len(unique_columns))})


# Get features and target
target = 'Play'
features = df.drop(columns=target).columns
x = df[features]
y = df[target]

e_target = entropy_target(y)
weight_average = []
for i in range(len(x.columns)):
    unique_class = [x[x.columns[i]].value_counts().get(j, 0) for j in x[x.columns[i]].unique()]
    entropy = entropy_feature(x[x.columns[i]], y)
    # Weight Average
    weight = sum((unique_class[j] / len(x)) * entropy[j] for j in range(len(unique_class)))
    # print(f"Weight(Feature = {x.columns[i]}):", weight)
    weight_average.append(weight)

information_gain = []
for i in range(len(x.columns)):
    ig = e_target - weight_average[i]
    print(f"Information Gain(Feature = {x.columns[i]}):", ig)
    information_gain.append(ig)

for i in range(len(information_gain)):
    if information_gain.index(max(information_gain)) == i:
        print("Best feature:", x.columns[i])
