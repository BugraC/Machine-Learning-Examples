import matplotlib.pyplot as plt

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import sklearn.model_selection as ms
from learning_curve import plot_learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

class KNNLearner:
    def __init__(self, X, Y, K, KFolds,isRegression):
        self.K = K
        self.X = X
        self.Y = Y
        self.KFolds = KFolds
        self.isRegression = isRegression
        self.split(X, Y)

    def split(self,X,Y):
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(X, Y, test_size=0.30, random_state=123)
        # if (self.isRegression):
        #     split = ms.StratifiedKFold(n_splits=2, random_state=123)
        #     for train_index, test_index in split.split(self.X):
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
        parameters = {'n_neighbors': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}

        if(self.isRegression):
            self.knn = KNeighborsRegressor(n_neighbors=self.K)
        else:
            self.knn = KNeighborsClassifier(n_neighbors=self.K)
        self.clf_tuned = GridSearchCV(self.knn, cv=5, param_grid=parameters)
        if(skipTrain == False):
            # Fit the model on the training data.
            self.clf_tuned.fit(self.trainX, self.trainY)
            print('Best parameters: %s' % self.clf_tuned.best_params_)
            # Make point predictions on the test set using the fit model.
            self.y_pred = self.clf_tuned.best_estimator_.predict(self.testX)
            self.y_pred_train = self.clf_tuned.best_estimator_.predict(self.trainX)


    def report(self):
        if (self.isRegression):
            self.knn = KNeighborsRegressor(n_neighbors=self.K)
            self.knn.fit(self.trainX, self.trainY)
            self.y_pred = self.knn.predict(self.testX)
            # print(classification_report(self.testY, self.y_pred_with_boost))
            CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.knn, self.testX, self.testY,
                                                                   cv=self.KFolds).mean(), cross_val_score(
                self.knn,
                self.testX,
                self.testY,
                cv=self.KFolds * 2).mean(), accuracy_score(
                self.testY, self.y_pred)

            return CV_Score1, CV_Score2, Accuracy_Score
        else:
            self.knn = KNeighborsClassifier(n_neighbors=self.K)
            self.knn.fit(self.trainX, self.trainY)
            self.y_pred = self.knn.predict(self.testX)
            # print(classification_report(self.testY, self.y_pred))
            CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.knn, self.testX, self.testY,
                                                                   cv=self.KFolds).mean(), cross_val_score(self.knn,
                                                                                                           self.testX,
                                                                                                           self.testY,
                                                                                                           cv=self.KFolds * 2).mean(), accuracy_score(
                self.testY, self.y_pred)

            return CV_Score1, CV_Score2, Accuracy_Score

    def plot_learning_curve(self):
        plot_learning_curve(self.clf_tuned.best_estimator_, 'Learning Curves for KNN', self.trainX, self.trainY, ylim=(0.0, 1.01), cv=5, n_jobs=4)
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