# DecisionTrees.py
import pandas as pd
from math import log2
from graphviz import Digraph


# --- ENTROPY OF TARGET ---
def entropy_target(target):
    p_list = []
    n = target.unique()  # Unique outcomes
    for i in range(len(n)):  
        p = target.value_counts().get(n[i], 0) / len(target)
        print(f"P(Target = {target.name} -> Outcome {n[i]}): {p}")
        p_list.append(p)

    entropy = sum(-p * log2(p) for p in p_list if p > 0)
    print(f"E(Target = {target.name}): {entropy}")
    return entropy


# --- ENTROPY OF FEATURE ---
def entropy_feature(feature, target):
    df_combined = pd.concat([feature, target], axis=1)
    entropy_list = []
    n_feature = feature.unique()
    n_target = target.unique()
    print("\nFeature =", feature.name)
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


# --- INFORMATION GAIN ---
def information_gain_all(x, y):
    e_target = entropy_target(y)
    gains = {}
    for col in x.columns:
        unique_class = x[col].value_counts().tolist()
        entropy_values = entropy_feature(x[col], y)
        weight = sum((unique_class[j] / len(x)) * entropy_values[j] for j in range(len(unique_class)))
        ig = e_target - weight
        gains[col] = ig
        print(f"\nInformation Gain(Feature = {col}): {ig}")
    return gains


# --- BUILD DECISION TREE (Graphviz) ---
def build_tree(x, y, feature_names, graph=None, parent=None, edge_label=""):
    if graph is None:
        graph = Digraph(format='png')
        graph.attr("node", shape="ellipse")

    # If all target values are the same → stop
    if len(y.unique()) == 1:
        graph.node(str(id(y)), f"Leaf: {y.iloc[0]}")
        if parent:
            graph.edge(parent, str(id(y)), label=edge_label)
        return graph

    # If no features left → stop
    if len(feature_names) == 0:
        majority_class = y.mode()[0]
        graph.node(str(id(y)), f"Leaf: {majority_class}")
        if parent:
            graph.edge(parent, str(id(y)), label=edge_label)
        return graph

    # Compute Information Gain
    gains = information_gain_all(x, y)
    best_feature = max(gains, key=gains.get)
    node_label = f"{best_feature}\n(IG={gains[best_feature]:.3f})"
    graph.node(str(id(x)), node_label)
    if parent:
        graph.edge(parent, str(id(x)), label=edge_label)

    # Split recursively
    for value in x[best_feature].unique():
        sub_x = x[x[best_feature] == value].drop(columns=best_feature)
        sub_y = y[x[best_feature] == value]
        build_tree(sub_x, sub_y, sub_x.columns, graph, str(id(x)), f"{best_feature}={value}")

    return graph


def main():
    pass


if __name__ == "__main__":
    main()
