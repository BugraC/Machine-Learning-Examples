from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import pydotplus

class neural_network:
    def __init__(self, X, Y, KFolds):
        self.KFolds = KFolds
        self.split(X, Y)
        return

    def split(self,X,Y):
        split = StratifiedShuffleSplit(Y, n_iter=1, test_size=0.2, random_state=0)
        train_index, test_index = list(split)[0]
        self.trainX, self.trainY = X[train_index], Y[train_index]
        self.testX, self.testY = X[test_index], Y[test_index]

    def train(self):
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        self.clf.fit(self.trainX, self.trainY)
        self.y_pred = self.clf.predict(self.testX)
        # dot_data = MLPClassifier.export_graphviz(clf, out_file=None)
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # graph.write_pdf("iris_neuralnetwork.pdf")


    def report(self):
        # print(classification_report(self.testY, self.y_pred))
        CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.clf, self.testX, self.testY,
                                                               cv=self.KFolds).mean(), cross_val_score(self.clf,
                                                                                                       self.testX,
                                                                                                       self.testY,
                                                                                                       cv=self.KFolds * 2).mean(), accuracy_score(
            self.testY, self.y_pred)

        return CV_Score1, CV_Score2, Accuracy_Score