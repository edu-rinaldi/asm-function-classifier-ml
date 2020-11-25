"""
    Author: Eduardo Rinaldi
    Date: 12/11/2020
    Copyright ©2020
"""

from utils import *

import json
import ast

# sklearn
from sklearn.model_selection import *
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# sklearn metrics
from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix

# matplot
import matplotlib.pyplot as plt

# numpy arrays
import numpy as np


TEST_SIZE = 0.33
MODEL = 'SVMa'

if __name__ == "__main__":
    # dataset_path
    dataset_path = 'noduplicatedataset.json' # 'dataset.json' 'noduplicatedataset.json' 'blindtest.json' 'nodupblindtest.json'
    
    print("Parsing the dataset..")

    # get the dataset
    X_all, y_all = parseDataset(dataset_path)

    print("The dataset has been parsed.")
    print("X: ")
    print(X_all)
    print("y: ")
    print(y_all)

    # split data
    print(f"Splitting the data, test size: {TEST_SIZE}")
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=117)
    print("Splitting completed.")

    if MODEL == 'SVM':
        # SVM  ~0.863 accuracy
        model = svm.SVC(kernel='linear', C=1)
    else:
        # Decision Tree ~0.95 accuracy
        model = tree.DecisionTreeClassifier()

    print("\n--- Using",type(model).__name__, "model ---\n")

    print("Fitting the model...")
    model.fit(X_train, y_train)
    print("Fit completed.")
    
    # Prediction
    y_pred = model.predict(X_test)

    # --- Scores ---
    report = classification_report(y_test, y_pred)
    print(report)

    # Confusion matrix (terminal)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    # Confusion matrix (plt)
    matrix = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, values_format='')
    matrix.ax_.set_title("Confusion matrix", color="black")
    # plt.show()

    # --- END Scores ---


    # BLINDTEST OF MODEL
    print("Blindtest begin")
    data2 = parseDataset('blindtest.json', False)

    prediction = model.predict(data2)
    prediction = np.vectorize(classByVal.__getitem__)(prediction)
    print("Predicted", prediction)
    np.savetxt('blindtest_DT.txt', prediction, fmt="%s")
    

                   