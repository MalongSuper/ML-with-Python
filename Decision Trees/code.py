# Code.py
import pandas as pd
import decision_trees as dt

def main():
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast'],
        'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Weak'],
        'Play_Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
    }

    df = pd.DataFrame(data)
    print("Original Data:\n", df)

    # Mapping categorical to numeric
    for col in df.columns:
        unique_values = df[col].unique()
        df[col] = df[col].map({unique_values[i]: i for i in range(len(unique_values))})

    print("\nAfter Mapping:\n", df)

    x = df.drop(columns="Play_Tennis")
    y = df["Play_Tennis"]

    # Build the tree using Graphviz
    tree = dt.build_tree(x, y, x.columns)
    tree.render("decision_tree", view=True)

if __name__ == "__main__":
    main()

