# Neural Network using Tensorflow - Deep Learning
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST Dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Train Images length:", len(train_images))
print("Train Labels length:", len(train_labels))
print("Test Images length:", len(test_images))
print("Test Labels length:", len(test_labels))

print("\nTrain Images shape:", train_images.shape)
print("Train Labels shape:", train_labels.shape)
print("Test Images shape:", test_images.shape)
print("Test Labels shape:", test_labels.shape)


# Display images with matplotlib
for i in range(1, 2):
    for j in range(10):
        print(f"{train_images[j]} --> {train_labels[j]}")
        plt.subplot(i + 1, 5, j + 1)
        plt.imshow(train_images[j], cmap='gray')
        plt.title(f"Label: {train_labels[j]}")
        plt.axis('off')

plt.show()


# Reshape image from (28, 28) → (784,...)
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))
print("Train Images Shape:", train_images.shape)
print("Test Images Shape:", test_images.shape)

# Normalize data
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255


# Build the model
hidden_layer1 = layers.Dense(512, activation="relu")
hidden_layer2 = layers.Dense(256, activation="relu")
output_layer = layers.Dense(10, activation="softmax")

model = keras.Sequential([hidden_layer1, hidden_layer2,
                          output_layer])

# Compile the model
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Fit the model
model.fit(train_images, train_labels, epochs=20, batch_size=128, validation_split=0.2)

# Record the model in history
history = model.history

# Print the metrics
print("Train Loss:", history.history["loss"][-1])
print("Validation Loss:", history.history["val_loss"][-1])
print("Train Accuracy:", history.history["accuracy"][-1])
print("Validation Accuracy:", history.history["val_accuracy"][-1])


# Plot the Train Loss, Validation Loss, Train Accuracy, Validation Accuracy
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(epochs, loss, 'bo', label='Training loss')
ax[0].plot(epochs, val_loss, 'b', label='Validation loss')
ax[0].set_title('MNIST - Training loss vs Validation')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(epochs, acc, 'ro', label='Training accuracy')
ax[1].plot(epochs, val_acc, 'r', label='Validation accuracy')
ax[1].set_title('MNIST - Training accuracy vs Validation')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()


# Evaluate on test set
y_pred = model.predict(test_images)
y_pred = np.argmax(y_pred, axis=1)
print("Test Accuracy:", np.mean(y_pred == test_labels))
print("Test Loss:", model.evaluate(test_images, test_labels)[0])


# Predict on new images
# Get the first 10 images from the test set
new_images = test_images[:10]
new_labels = test_labels[:10]
# Create a figure and a 2x5 subplot grid
fig2, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten() # Flatten the 2x5 array of axes for easy iteration

# Predict for each image and display the result
for i in range(10):
    current_image = new_images[i]
    true_label = new_labels[i]
    # Reshape to match the model's input shape (1, 28 * 28)
    image_input = current_image.reshape(1, 28 * 28)
    prediction = model.predict(image_input, verbose=0)
    predicted_label = prediction[0].argmax()

    print(f"Prediction: {predicted_label}, True Label: {true_label}")

    # Display the image in the subplot
    axes[i].imshow(current_image.reshape(28, 28), cmap='gray')
    axes[i].set_title(f"Pred: {predicted_label}, True: {true_label}")
    axes[i].axis('off')

plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
plt.show()


