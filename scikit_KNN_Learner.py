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
        y_pred = knn.predict(self.testX)

        print(classification_report(self.testY, y_pred))

        # # load iris the datasets
        # dataset = datasets.load_iris()
        # # fit a k-nearest neighbor model to the data
        # model = KNeighborsClassifier()
        # model.fit(dataset.data, dataset.target)
        # print(model)
        # # make predictions
        # expected = dataset.target
        # predicted = model.predict(dataset.data)
        # # summarize the fit of the model
        # print(metrics.classification_report(expected, predicted))
        # print(metrics.confusion_matrix(expected, predicted))

