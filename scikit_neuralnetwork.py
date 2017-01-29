from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from learning_curve import plot_learning_curve
import pydotplus
import matplotlib.pyplot as plt

class neural_network:
    def __init__(self, X, Y, KFolds, hiddenLayerSizes):
        self.X = X
        self.Y = Y
        self.KFolds = KFolds
        self.split(X, Y)
        self.hiddenLayerSizes = hiddenLayerSizes
        return

    def split(self,X,Y):
        split = StratifiedShuffleSplit(Y, n_iter=100, test_size=0.3)
        train_index, test_index = list(split)[0]
        self.trainX, self.trainY = X[train_index], Y[train_index]
        self.testX, self.testY = X[test_index], Y[test_index]

    def train(self):
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(self.hiddenLayerSizes,), random_state=1)
        self.clf.fit(self.trainX, self.trainY)
        self.y_pred = self.clf.predict(self.testX)
        self.y_pred_train = self.clf.predict(self.trainX)
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

    def plot_learning_curve(self):
        plot_learning_curve(self.clf, 'Learning Curves Neural Network', self.X, self.Y, ylim=(0.1, 1.01), cv=5, n_jobs=4)
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