from utils import *

from matplotlib import pyplot as plt
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers, models
from sklearn.metrics import accuracy_score

def build_model(X_train, X_test):
    model = models.Sequential()

    X_train = np.reshape(X_train, (X_train.shape[0], 8, 8, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 8, 8, 1))
    
    model.add(layers.Conv2D(64, (2, 2), input_shape=(8, 8, 1)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='sigmoid'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model, X_train, X_test

def run_experiment():
    X_train, X_test, y_train, y_test = load_optdigits()

    model, X_train, X_test = build_model(X_train, X_test)

    print(model.summary())
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10)
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    
    print(accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    run_experiment()
        
    # X_train, X_test, y_train, y_test = load_optdigits()
    # plt.imshow(np.reshape(X_train[0], (8, 8)), cmap='gray')
    # plt.show()
