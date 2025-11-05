import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0




#check to make sure there are NO values that are not a number (NaN)

print("Any NaN Training:", np.isnan(x_train).any())
print("Any NaN Testing:", np.isnan(x_test).any())

#tell the model what shape to expect
input_shape = (28, 28, 1) #28x28 p0x, 1 color channel (greyscale) - 3 for RGB

#reshape the training and testing data to include the color channel
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train/255.0 #normalize the data to be between 0 and 1
#same for testing
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test = x_test/255.0 #normalize the data to be between 0 and 1

#convert our labels to be one-hot, not sparse
y_train = tf.one_hot(y_train.astype(np.int32), depth =10)
y_test = tf.one_hot(y_test.astype(np.int32), depth =10)

#show an example image from MNIST
plt.imshow(x_train[100][:,:,0], cmap='gray')
plt.show()



batch_size = 128
num_classes = 10
epochs = 20

#build the model... finally...
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')



    ]
)

model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,validation_data=(x_test, y_test))

#plot out training and validation accuracy and loss
fig, ax = plt.subplots(2, 1)

ax[1].plot(history.history['acc'], color = 'b', label="Training Accuracy")
ax[1].plot(history.history['val_acc'], color = 'r', label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()