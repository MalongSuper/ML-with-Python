import numpy as np

true_value = np.array(['A', 'B', 'A', 'C', 'C', 'B', 'A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'B'])
predict_value = np.array(['A', 'A', 'C', 'C', 'B', 'B', 'A', 'A', 'A', 'C', 'A', 'A', 'C', 'C', 'B'])

unique_labels = np.unique(true_value).tolist()
print("Unique Labels:", unique_labels)

# Iterate to replace the labels
for i in range(len(unique_labels)):
    true_value[true_value == unique_labels[i]] = i
    predict_value[predict_value == unique_labels[i]] = i

# Convert them to integer
true_value = np.array(true_value, dtype=int)
predict_value = np.array(predict_value, dtype=int)
print("True Value:", true_value)
print("Predict Value:", predict_value)

# Construct the confusion matrix
confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)))
unique_labels = np.unique(true_value).tolist()


    