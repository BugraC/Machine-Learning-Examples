from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

class KNNLearner:
    def __init__(self, X, Y, K, KFolds):
        self.K = K
        self.KFolds = KFolds
        self.split(X, Y)

    def split(self,X,Y):
        split = StratifiedShuffleSplit(Y, n_iter=1, test_size=0.2, random_state=0)
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

