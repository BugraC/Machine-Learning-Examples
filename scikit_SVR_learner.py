import numpy as np

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.svm import SVR


class support_vector_machines_SVR:
    def __init__(self, X, Y):
        self.split(X, Y)
        return

    def split(self, X, Y):
        split = StratifiedShuffleSplit(Y, n_iter=1, test_size=0.2, random_state=0)
        train_index, test_index = list(split)[0]
        self.trainX, self.trainY = X[train_index], Y[train_index]
        self.testX, self.testY = X[test_index], Y[test_index]

    def train(self):
        clf = SVR(C=1.0, epsilon=0.2)
        clf.fit(self.trainX, self.trainY)

        y_pred = clf.predict(self.testX)

        print(classification_report(self.testY, y_pred))