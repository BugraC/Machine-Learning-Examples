import numpy as np

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from learning_curve import plot_learning_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split

class support_vector_machines:
    def __init__(self, X, Y, KFolds, kernelFunction, C, gamma, isRegression):
        self.KFolds = KFolds
        self.X = X
        self.Y = Y
        self.isRegression = isRegression
        self.split(X, Y)
        self.kernelFunction = kernelFunction
        self.C = C
        self.gamma = gamma
        return


    def split(self, X, Y):
        # number_of_samples = len(Y)
        # np.random.seed(0)
        # random_indices = np.random.permutation(number_of_samples)
        # num_training_samples = int(number_of_samples * 0.75)
        # self.trainX = X[random_indices[:num_training_samples]]
        # y_Train = Y[random_indices[:num_training_samples]]
        # self.testX = X[random_indices[num_training_samples:]]
        # self.testY = Y[random_indices[num_training_samples:]]
        # self.trainY = list(y_Train)

        self.trainX, self.testX, self.trainY, self.testY = train_test_split(X, Y, test_size=0.30, random_state=123)

        # if (self.isRegression):
        #     split = ms.StratifiedKFold(n_splits=2, random_state=123)
        #     for train_index, test_index in split.split(self.X, self.Y):
        #         self.trainX = self.X[train_index]
        #         self.trainY = self.Y[train_index]
        #         self.testX = self.X[test_index]
        #         self.testY = self.Y[test_index]
        # else:
        #     split = StratifiedShuffleSplit(Y, n_iter=100, test_size=0.3)
        #     train_index, test_index = list(split)[0]
        #     self.trainX, self.trainY = X[train_index], Y[train_index]
        #     self.testX, self.testY = X[test_index], Y[test_index]


    def train(self,skipTrain = False):
        if(self.isRegression):
            if self.kernelFunction == 'sigmoid':
                self.parameters = {'kernel': ['sigmoid'],
                                   'C': [1.0,10.0,50.0,100.0],
                                   'gamma': np.logspace(-9, 3, 13),
                                    'cache_size': [24000000]}
                self.clf = svm.SVR(kernel=self.kernelFunction, cache_size=24000000)
            elif self.kernelFunction == 'rbf':
                self.parameters = {'kernel': ['rbf'],
                                   'C': [1.0,10.0,50.0,100.0],
                                   'gamma': np.logspace(-9, 3, 13),
                                   'cache_size': [24000000]}
                self.clf = svm.SVR(kernel=self.kernelFunction, cache_size=240000)
        else:
            if self.kernelFunction == 'sigmoid':
                self.parameters = {'kernel': ['sigmoid'],
                                   'C': [1.0,10.0,50.0,100.0],
                                   'gamma':  np.logspace(-9, 3, 13),
                                   'cache_size': [24000000]}
                self.clf = svm.SVC(kernel=self.kernelFunction, cache_size=24000000)
            elif self.kernelFunction == 'rbf':
                self.parameters = {'kernel': ['rbf'],
                                   'C': [1.0,10.0,50.0,100.0],
                                   'gamma': np.logspace(-9, 3, 13),
                                   'cache_size': [24000000]}
                self.clf = svm.SVC(kernel=self.kernelFunction, cache_size=240000)


        self.clf_tuned = GridSearchCV(self.clf, cv=5, param_grid=self.parameters)
        if(skipTrain == False):
            # Fit the model on the training data.
            self.clf_tuned.fit(self.trainX, self.trainY)
            print('Best parameters: %s' % self.clf_tuned.best_params_)
            # X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
            # y = np.array([1, 1, 2, 2])
            # self.clf.fit(self.trainX, self.trainY)
            # self.clf.fit(X, y)
            self.y_pred = self.clf_tuned.predict(self.testX)
            self.y_pred_train = self.clf_tuned.predict(self.trainX)
            # self.y_pred = self.clf.predict([[-0.8, -1]])




    def report(self):
        if (self.isRegression):
            if self.kernelFunction == 'sigmoid':
                self.clf = svm.SVR(kernel=self.kernelFunction, gamma=self.gamma, C=self.C, cache_size=24000000)
            elif self.kernelFunction == 'rbf':
                self.clf = svm.SVR(kernel=self.kernelFunction, gamma=self.gamma, C=self.C, cache_size=24000000)

            self.clf.fit(self.trainX, self.trainY)
            self.y_pred = self.clf.predict(self.testX)
            # print(classification_report(self.testY, self.y_pred_with_boost))
            CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.clf, self.testX, self.testY,
                                                                   cv=self.KFolds).mean(), cross_val_score(
                self.clf,
                self.testX,
                self.testY,
                cv=self.KFolds * 2).mean(), cross_val_score(
                self.testY, self.y_pred)

            return CV_Score1, CV_Score2, Accuracy_Score
        else:
            if self.kernelFunction == 'sigmoid':
                self.clf = svm.SVC(kernel=self.kernelFunction, gamma=self.gamma, C=self.C,
                                   cache_size=24000000)
            elif self.kernelFunction == 'rbf':
                self.clf = svm.SVC(kernel=self.kernelFunction, gamma=self.gamma, C=self.C, cache_size=24000000)
            # print(classification_report(self.testY, self.y_pred))
            self.clf.fit(self.trainX, self.trainY)
            self.y_pred = self.clf.predict(self.testX)
            CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.clf, self.testX, self.testY,
                                                                   cv=self.KFolds).mean(), cross_val_score(self.clf,
                                                                                                           self.testX,
                                                                                                           self.testY,
                                                                                                           cv=self.KFolds * 2).mean(), accuracy_score(
                self.testY, self.y_pred)

            return CV_Score1, CV_Score2, Accuracy_Score

    def plot_learning_curve(self):
        plot_learning_curve(self.clf_tuned.best_estimator_, 'Learning Curves for SVC', self.trainX, self.trainY, ylim=(0.1, 1.01), cv=5, n_jobs=4)
        plt.show()

    def return_error(self):
        # error = 0
        # for i in range(len(self.trainY)):
        #     error += (abs(self.trainY[i] - self.y_pred_train[i]) / self.trainY[i])
        # error_train_percent = error / len(self.trainY) * 100
        # print("Train error = "'{}'.format(error_train_percent) + " percent")
        error_train_percent = mean_squared_error(self.trainY, self.y_pred_train)
        print("Train error = "'{}'.format(error_train_percent) + " percent")

        # error = 0
        # for i in range(len(self.testY)):
        #     error += (abs(self.y_pred[i] - self.testY[i]) / self.testY[i])
        # error_test_percent = error / len(self.testY) * 100
        # print("Test error = "'{}'.format(error_test_percent) + " percent")

        error_test_percent = mean_squared_error(self.testY, self.y_pred)
        print("Train error = "'{}'.format(error_test_percent) + " percent")
        return error_train_percent, error_test_percent

    def confusion_matrix(self):
        # Compute confusion matrix
        cm = confusion_matrix(self.testY, self.y_pred)

        print(cm)

        # Show confusion matrix in a separate window
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()