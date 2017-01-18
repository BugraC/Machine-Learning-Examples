from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsRegressor

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
        knn = KNeighborsRegressor(n_neighbors=5)
        # Fit the model on the training data.
        knn.fit(self.trainX, self.trainY )
        # Make point predictions on the test set using the fit model.
        predictions = knn.predict(self.testX)
        print(predictions)

