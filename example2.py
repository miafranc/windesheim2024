from utils import *

from matplotlib import pyplot as plt
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers, models
from sklearn.metrics import accuracy_score

def build_model(X_train):
    model = models.Sequential()

    model.add(layers.Input(shape=(X_train.shape[1])))
    model.add(layers.Dense(1))
    model.add(layers.Activation(activation='sigmoid'))
    
    return model

def run_experiment():
    X_train, X_test, y_train, y_test = load_heart()
    # X_train, X_test, y_train, y_test = load_optdigits()

    model = build_model(X_train)

    print(model.summary())
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10)
    y_pred = (model.predict(X_test) > 0.5)
    
    print(accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    run_experiment()
    