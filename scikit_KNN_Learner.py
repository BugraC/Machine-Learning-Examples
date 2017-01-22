from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report
from sklearn import datasets

class KNNLearner:
    def __init__(self, X, Y, K):
        self.K = K
        self.split(X, Y)

    def split(self,X,Y):
        split = StratifiedShuffleSplit(Y, n_iter=1, test_size=0.2, random_state=0)
        train_index, test_index = list(split)[0]
        self.trainX, self.trainY = X[train_index], Y[train_index]
        self.testX, self.testY = X[test_index], Y[test_index]

    def train(self):
        # Create the knn model.
        # Look at the five closest neighbors.
        knn = KNeighborsClassifier(n_neighbors=self.K)
        # Fit the model on the training data.
        knn.fit(self.trainX, self.trainY)
        # Make point predictions on the test set using the fit model.
        self.y_pred = knn.predict(self.testX)

    def report(self):
        print(classification_report(self.testY, self.y_pred))
