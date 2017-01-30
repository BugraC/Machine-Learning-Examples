import numpy as np

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import cross_val_score
from learning_curve import plot_learning_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import sklearn.model_selection as ms

class support_vector_machines:
    def __init__(self, X, Y, KFolds, kernelFunction,isRegression):
        self.KFolds = KFolds
        self.X = X
        self.Y = Y
        self.isRegression = isRegression
        self.split(X, Y)
        self.kernelFunction = kernelFunction
        return


    def split(self, X, Y):
        if (self.isRegression):
            split = ms.StratifiedKFold(n_splits=2, random_state=123)
            for train_index, test_index in split.split(self.X, self.Y):
                self.trainX = self.X[train_index]
                self.trainY = self.Y[train_index]
                self.testX = self.X[test_index]
                self.testY = self.Y[test_index]
        else:
            split = StratifiedShuffleSplit(Y, n_iter=100, test_size=0.3)
            train_index, test_index = list(split)[0]
            self.trainX, self.trainY = X[train_index], Y[train_index]
            self.testX, self.testY = X[test_index], Y[test_index]


    def train(self, degree, gamma):
        C = 1.0
        if(self.isRegression):
            if self.kernelFunction == 'poly':
                self.clf = svm.SVR(kernel=self.kernelFunction, degree=degree, C=C, cache_size=240000)
            elif self.kernelFunction == 'rbf':
                self.clf = svm.SVR(kernel=self.kernelFunction, gamma=gamma, C=C, cache_size=240000)
        else:
            if self.kernelFunction == 'poly':
                self.clf = svm.SVC(kernel=self.kernelFunction, degree=degree, C=C, cache_size=240000)
            elif self.kernelFunction == 'rbf':
                self.clf = svm.SVC(kernel=self.kernelFunction, gamma=gamma, C=C, cache_size=240000)
        # X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        # y = np.array([1, 1, 2, 2])
        self.clf.fit(self.trainX, self.trainY)
        # self.clf.fit(X, y)
        self.y_pred = self.clf.predict(self.testX)
        self.y_pred_train = self.clf.predict(self.trainX)
        # self.y_pred = self.clf.predict([[-0.8, -1]])




    def report(self):
        if (self.isRegression):
            # print(classification_report(self.testY, self.y_pred_with_boost))
            CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.clf, self.testX, self.testY,
                                                                   cv=self.KFolds).mean(), cross_val_score(
                self.clf,
                self.testX,
                self.testY,
                cv=self.KFolds * 2).mean(), self.clf.score(
                self.testX, self.testY)

            return CV_Score1, CV_Score2, Accuracy_Score
        else:
            # print(classification_report(self.testY, self.y_pred))
            CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.clf, self.testX, self.testY,
                                                                   cv=self.KFolds).mean(), cross_val_score(self.clf,
                                                                                                           self.testX,
                                                                                                           self.testY,
                                                                                                           cv=self.KFolds * 2).mean(), accuracy_score(
                self.testY, self.y_pred)

            return CV_Score1, CV_Score2, Accuracy_Score

    def plot_learning_curve(self):
        plot_learning_curve(self.clf, 'Learning Curves for SVC', self.X, self.Y, ylim=(0.1, 1.01), cv=5, n_jobs=4)
        plt.show()

    def return_error(self):
        error = 0
        for i in range(len(self.trainY)):
            error += (abs(self.trainY[i] - self.y_pred_train[i]) / self.trainY[i])
        error_train_percent = error / len(self.trainY) * 100
        print("Train error = "'{}'.format(error_train_percent) + " percent")

        error = 0
        for i in range(len(self.testY)):
            error += (abs(self.y_pred[i] - self.testY[i]) / self.testY[i])
        error_test_percent = error / len(self.testY) * 100
        print("Test error = "'{}'.format(error_test_percent) + " percent")
        return error_train_percent,error_test_percent