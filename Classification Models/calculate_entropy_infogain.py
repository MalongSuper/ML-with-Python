import pandas as pd
from math import log2


def entropy_target(target):
    p_list = []
    n = target.unique()  # Unique outcomes in target
    for i in range(len(n)):
        p = target.value_counts().get(n[i], 0) / len(target)
        print(f"P(Target = {target.name} -> Outcome {n[i]}): {p}")
        p_list.append(p)
    entropy = sum(-p * log2(p) for p in p_list if p > 0)
    print(f"E(Target = {target.name}): {entropy}")
    return entropy


def entropy_feature(feature, target):
    df_combined = pd.concat([feature, target], axis=1)
    entropy_list = []
    n_feature = feature.unique()
    n_target = target.unique()
    print("Feature =", feature.name)
    for i in range(len(n_feature)):
        p_list = []
        for j in range(len(n_target)):
            num_class = feature.value_counts().get(n_feature[i], 0)
            sub_df = df_combined[
                (df_combined[feature.name] == n_feature[i]) &
                (df_combined[target.name] == n_target[j])
            ]
            p = len(sub_df) / num_class if num_class != 0 else 0
            print(f"+ P({n_feature[i]} -> Outcome = {n_target[j]}): {p}")
            p_list.append(p)
        entropy = sum(-p * log2(p) for p in p_list if p > 0)
        print(f"- Entropy(Feature Class = {n_feature[i]}): {entropy}")
        entropy_list.append(entropy)
    return entropy_list


def information_gain(df, target_column):
    # Map all categorical values to numerical values
    df = df.copy()
    for col in df.columns:
        unique_vals = df[col].unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        df[col] = df[col].map(mapping)

    features = df.drop(columns=target_column).columns
    x = df[features]
    y = df[target_column]

    # Entropy of the target
    e_target = entropy_target(y)

    info_gain_dict = {}
    for feature in x.columns:
        entropy_list = entropy_feature(x[feature], y)
        class_counts = [x[feature].value_counts().get(cls, 0) for cls in x[feature].unique()]
        weighted_entropy = sum((class_counts[i] / len(x)) * entropy_list[i] for i in range(len(class_counts)))
        ig = e_target - weighted_entropy
        info_gain_dict[feature] = ig
        print(f"Information Gain(Feature = {feature}): {ig}")

    # Identify best feature
    best_feature = max(info_gain_dict, key=info_gain_dict.get)
    print("Best feature:", best_feature)

    return info_gain_dict, best_feature
