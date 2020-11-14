"""
    Author: Eduardo Rinaldi
    Date: 12/11/2020
    Copyright ©2020
"""

from utils import *

import json
import ast

from sklearn.model_selection import *
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import numpy as np
from math import log2

if __name__ == "__main__":

    # class mapping : string -> {0,...,n}, n = 3
    classByVal = {0: 'encryption', 1 : 'sort', 2 : 'math', 3 : 'string'}
    valByClass = {'encryption' : 0, 'sort' : 1, 'math' : 2, 'string': 3}

    # reading jsonl and preparing dataset
    data = []
    print('Preparing the dataset...')
    with open('dataset.json', 'r') as f:
        for e in f:
            v = json.loads(e)
            row = get_features(ast.literal_eval(v['lista_asm']), v['cfg'], normalization_func=lambda x: log2(x+1))
            row.append(valByClass[v['semantic']])
            data += [row]

    # convert to numpy array
    data = np.array(data)
    
    # extract X and y from data
    X_all = data[:,:-1]
    y_all = data[:, -1]

    print("The dataset has been prepared.")

    # split data
    print("Splitting the data...")
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33, random_state=117)
    print("Splitting completed.")


    # SVM  ~0.863 accuracy
    model = svm.SVC(kernel='linear', C=1)

    # Bayes ~0.723 accuracy
    # model = GaussianNB()

    # Decision Tree ~0.95 accuracy
    # model = tree.DecisionTreeClassifier()

    print("\n--- Using",type(model).__name__, "model ---\n")

    print("Fitting the model...")
    model.fit(X_train, y_train)
    print("Fit completed.")
    acc = model.score(X_test, y_test)    
    print("Accuracy %.3f" %acc)


    # data2 = []
    # with open('blindtest.json', 'r') as f:
    #     for e in f:
    #         v = json.loads(e)
    #         row = get_features(ast.literal_eval(v['lista_asm']), v['cfg'], normalization_func=lambda x: log2(x+1))
    #         data2.append(row)
    # data2 = np.array(data2)

    # prediction = model.predict(data2[1:2, :])
    # print("Predicted", prediction)
    

                   