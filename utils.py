from sklearn.datasets import load_svmlight_file
import numpy as np

def load_heart():
    X, y = load_svmlight_file('data/heart_scale.txt')
    
    X_train = X[:150].todense()
    y_train = (y[:150] + 1) / 2
    X_test = X[150:].todense()
    y_test = (y[150:] + 1) / 2
    
    return X_train, X_test, y_train, y_test

def load_optdigits():
    X_train = np.loadtxt('data/optdigits.tra', delimiter=',')
    y_train = X_train[:,-1]
    X_train = X_train[:,:-1]
    
    X_test = np.loadtxt('data/optdigits.tes', delimiter=',')
    y_test = X_test[:,-1]
    X_test = X_test[:,:-1]

    return X_train, X_test, y_train, y_test
