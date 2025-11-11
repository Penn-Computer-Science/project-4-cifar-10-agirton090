import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import cifar10
                                                                                                                                               
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

class_names = [
    "airplane", "automobile","bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
]

print("Training Shape:", x_train.shape)
print("Testing Shape:", x_train.shape)
print("Any NaN Training:", np.isnan(x_train).any())
print("Any NaN Testing:", np.isnan(x_test).any())


input_shape = (32, 32, 3)
num_classes = 10

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')



    ]
)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size = 128,
    validation_data=(x_test, y_test)
)

plt.tight_layout()
plt.show()








