# Confusion Matrix
from sklearn.metrics import confusion_matrix
import pandas as pd
# "1" is positive, "0" is negative
true_data = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  # Actual labels
pred_data = [0, 1, 0, 0, 1, 0, 0, 0, 1, 1]  # Predicted labels
cm = confusion_matrix(true_data, pred_data)
# Convert the confusion matrix to a pandas DataFrame
row_labels = ["Predicted: Negative", "Predicted: Positive"]
col_labels = ["Actual: Negative", "Actual: Positive"]
cm_df = pd.DataFrame(cm, index=row_labels, columns=col_labels)
print(cm_df)
