# Naive Bayes Classifer model

def calculate_priors(df, target):
    """
    Calculate prior probabilities for each target class.
    Returns a dictionary: {class_label: P(Class=label)}
    """
    n = len(df)
    prob_target = {}

    for i in df[target].unique():
        prob_target[str(i)] = len(df[df[target] == i]) / n
        print(f"P({target} = {i}) = {prob_target[str(i)]:.4f}")
    print()
    return prob_target


def calculate_conditionals(df, target, prob_target):
    """
    Calculate conditional probabilities for each feature value given a class,
    with Laplace smoothing applied.
    """
    n = len(df)
    prob_feature_given_class = {}

    for feature in df.columns:
        if feature == target:
            continue

        print(f"--- Feature: {feature} ---")
        k = len(df[feature].unique())  # number of unique values for smoothing

        for value in df[feature].unique():
            print(f"Value: {value}")

            for cls in df[target].unique():
                subset = df[df[feature] == value]
                m = len(subset[subset[target] == cls])  # count(feature=value & class=cls)
                count_c = len(df[df[target] == cls])   # count(class=cls)

                # ✅ Laplace smoothing formula
                prob_features = (m + 1) / (count_c + k)

                # Joint probability (feature=value ∧ class=cls)
                p_joint = prob_features * prob_target[str(cls)]

                print(f"P({feature} = {value} ∧ {target} = {cls}) = {p_joint:.4f}")
                print(f"P({feature} = {value} | {target} = {cls}) = {prob_features:.4f}")
                print(f"P({feature} ≠ {value} | {target} = {cls}) = {1 - prob_features:.4f}\n")

                # Store smoothed conditional probability
                prob_feature_given_class[(feature, value, cls)] = prob_features
        print()

    return prob_feature_given_class
