import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("GPUs:", tf.config.list_physical_devices('GPU'))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Dummy data
import numpy as np
x = np.random.rand(1000, 20)
y = np.random.randint(0, 10, 1000)

model.fit(x, y, epochs=5, batch_size=32)
