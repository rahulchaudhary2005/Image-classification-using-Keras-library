import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow  import keras


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(f"[INFO] training data shape: {x_train.shape}")
print(f"[INFO] testing data shape: {x_test.shape}")

# for visualizing some images
def plot_sample_images(x, y, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i])
        plt.xlabel(class_names[y[i][0]])
    plt.show()
    
# Define class names for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
plot_sample_images(x_train, y_train, class_names)

# Normalize pixel values to be between 0 and 1
x_train_scaled= x_train.astype('float32') / 255.0
x_test_scaled= x_test.astype('float32') / 255.0

# one-hot encode the labels
y_train_categorical= keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test_categorical= keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
print(f"[INFO] training labels shape: {y_train_categorical.shape}")
print(f"[INFO] testing labels shape: {y_test_categorical.shape}")

#Building the ann model for image classification

print("[INFO] building model...")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
    
])  

model.compile(optimizer='SGD',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print(model.summary())

# Training the model
print("[INFO] training model...")
history = model.fit(x_train_scaled, y_train_categorical,
                    validation_data=(x_test_scaled, y_test_categorical),
                    epochs=20,
                    batch_size=256)

#predict the model
print("[INFO] evaluating model...")
y_pred = model.predict(x_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
print(f"[INFO] classification report:\n {classification_report(y_test, y_pred_classes)}")

