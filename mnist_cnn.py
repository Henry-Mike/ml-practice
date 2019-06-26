import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
import numpy as np

from util import plot_history

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

x_train = np.expand_dims(train_images, -1)
x_train = x_train / 255.0
x_test = np.expand_dims(test_images, -1)
x_test = x_test / 255.0

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_acc', patience=3),
    keras.callbacks.ModelCheckpoint(filepath='./trained_models/mnist_cnn_weights.h5', monitor='val_loss', save_best_only=True),
    # keras.callbacks.TensorBoard(histogram_freq=1)
]
history = model.fit(x_train, y_train, batch_size=256, epochs=10, validation_split=0.15, callbacks=callbacks)

loss, accuracy = model.evaluate(x_test, y_test)
print('test loss: {}, test accuracy: {}'.format(loss, accuracy))

plot_history(history)
