#Import Library
print('[INFO] Import Library . . .')

import platform; print(platform.platform())
import sys; print("[INFO] Python", sys.version)
import numpy; print("[INFO] NumPy", numpy.__version__)
import scipy; print("[INFO] SciPy", scipy.__version__)

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import dump
from sklearn import preprocessing

#Fungsi Untuk Training
def train():
    
    # Load, read and normalize training data
    print('[INFO] Import Data Train . . .')
    training = "./train.csv"
    data_train = pd.read_csv(training)
        
    y_train = data_train['# Letter'].values
    X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)

    print("[INFO] Shape of the training data . . .")
    print(X_train.shape)
    print(y_train.shape)
        
    # Data normalization (0,1)
    print("[INFO] Data Preprocessing . . .")
    X_train = preprocessing.normalize(X_train, norm='l2')
    
    # Models training
    # Linear Discrimant Analysis (Default parameters)
    print("[INFO] Modeling with LDA . . .")
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_train, y_train)
    
    # Neural Networks multi-layer perceptron (MLP) algorithm
    print("[INFO] Modeling with MLP . . .")
    clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=1000)
    clf_NN.fit(X_train, y_train)
       
    # Save model
    print("[INFO] Save Model . . .")
    from joblib import dump
    dump(clf_NN, 'Inference_NN.joblib')
    dump(clf_lda, 'Inference_lda.joblib')

if __name__ == '__main__':
    train()