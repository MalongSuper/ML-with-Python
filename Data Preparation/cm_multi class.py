import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def get_components(confusion_matrix, k):
    tp = confusion_matrix[k, k]
    fp = confusion_matrix[:, k].sum() - tp
    fn = confusion_matrix[k, :].sum() - tp
    tn = confusion_matrix.sum() - (tp + fp + fn)
    return tp, fp, fn, tn


true_value = np.array(['A', 'B', 'A', 'C', 'C', 'B', 'A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'B'])
predict_value = np.array(['A', 'A', 'C', 'C', 'B', 'B', 'A', 'A', 'A', 'C', 'A', 'A', 'C', 'C', 'B'])
# Replace the element
true_df = pd.Series(true_value).replace({"A": 0, "B": 1, "C": 2})
pred_df = pd.Series(predict_value).replace({"A": 0, "B": 1, "C": 2})
# Get the unique labels (will be used for metrics calculation and plotting)
unique_labels = true_df.unique().tolist()
# Confusion Matrix
confusion_matrix = confusion_matrix(true_df, pred_df, labels=unique_labels)
print("Confusion Matrix:\n", confusion_matrix)


for i in unique_labels:
    print("Label", i)
    tp, fp, fn, tn = get_components(confusion_matrix, i)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1_score = (2 * recall * precision) / (recall + precision)
    print(f"TP: {tp}; FP: {fp}; FN: {fn}; TN: {tn}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Specificity: {specificity}")
    print(f"F1 Score: {f1_score}")
    print()

# Plot
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
