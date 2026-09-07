# Classification Metrics with Python
# Spam Emails detection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# "1": Spam (Positive); "0": Not Spam (Negative)
true_data = ([1] * 80) + ([0] * 120)
pred_data = ([1] * 60) + ([0] * 20) + ([0] * 100) + ([1] * 20)
# Confusion matrix for Specificity
cm = confusion_matrix(true_data, pred_data)
# Classification Metrics
accuracy = accuracy_score(true_data, pred_data)
precision = precision_score(true_data, pred_data)
recall = recall_score(true_data, pred_data)
f1_score = f1_score(true_data, pred_data)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
# Find Specificity
tn, fp = cm[0][0], cm[0][1]
specificity = tn / (tn + fp)
print("Specificity:", specificity)
