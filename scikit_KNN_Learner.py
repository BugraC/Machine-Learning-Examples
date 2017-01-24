import matplotlib.pyplot as plt

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from learning_curve import plot_learning_curve


class KNNLearner:
    def __init__(self, X, Y, K, KFolds):
        self.K = K
        self.X = X
        self.Y = Y
        self.KFolds = KFolds
        self.split(X, Y)

    def split(self,X,Y):
        split = StratifiedShuffleSplit(Y, n_iter=100, test_size=0.3)
        train_index, test_index = list(split)[0]
        self.trainX, self.trainY = X[train_index], Y[train_index]
        self.testX, self.testY = X[test_index], Y[test_index]

    def train(self):
        # Create the knn model.
        # Look at the five closest neighbors.
        self.knn = KNeighborsClassifier(n_neighbors=self.K)
        # Fit the model on the training data.
        self.knn.fit(self.trainX, self.trainY)
        # Make point predictions on the test set using the fit model.
        self.y_pred = self.knn.predict(self.testX)

    def report(self):
        # print(classification_report(self.testY, self.y_pred))
        CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.knn, self.testX, self.testY,
                                                               cv=self.KFolds).mean(), cross_val_score(self.knn,
                                                                                                       self.testX,
                                                                                                       self.testY,
                                                                                                       cv=self.KFolds * 2).mean(), accuracy_score(
            self.testY, self.y_pred)

        return CV_Score1, CV_Score2, Accuracy_Score

    def plot_learning_curve(self):
        plot_learning_curve(self.knn, 'Learning Curves for KNN', self.X, self.Y, ylim=(0.1, 1.01), cv=5, n_jobs=4)
        plt.show()