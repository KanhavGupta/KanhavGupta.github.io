import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
import cv2 as cv
import pandas as pd
import contants


def load_data(data_file):
    data = pd.read_csv(data_file)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        faces.append(face)
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).to_numpy()
    return faces, emotions


faces, emotions = load_data(contants.train_data)
print(len(faces), len(emotions))
x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2,
                                                    random_state=1, shuffle=True)
print(len(x_train), len(y_train))
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
input_shape = (48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(7, activation=tf.nn.softmax))

# ----------------------------------------------------------------------

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10)
model.compile()
model.save('sentiment_model.h5', model)

# Epoch 10/10
# 898/898 [==============================] - 19s 22ms/step - loss: 0.3575 - accuracy: 0.8733
