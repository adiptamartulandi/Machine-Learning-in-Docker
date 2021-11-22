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
from joblib import dump, load
from sklearn import preprocessing

#Fungsi untuk Inference
def inference():
    
    # Load, read and normalize training data
    print('[INFO] Import Data Test . . .')
    testing = "./test.csv"
    data_test = pd.read_csv(testing)
        
    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
   
    print("[INFO] Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
    
    # Data normalization (0,1)
    print("[INFO] Data Preprocessing . . .")
    X_test = preprocessing.normalize(X_test, norm='l2')
    
    # Models training
    # Run model
    from joblib import dump, load
    print("[INFO] Inference Process . . .")
    clf_lda = load('Inference_lda.joblib')
    print("[INFO] LDA score and classification:")
    print(clf_lda.score(X_test, y_test))
    print(clf_lda.predict(X_test))
        
    # Run model
    clf_nn = load('Inference_NN.joblib')
    print("[INFO] MLP score and classification:")
    print(clf_nn.score(X_test, y_test))
    print(clf_nn.predict(X_test))

if __name__ == '__main__':
    inference()