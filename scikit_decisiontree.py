from sklearn import tree
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.datasets import load_iris
import pydotplus
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


class decision_tree:
    def __init__(self,X,Y):
        self.split(X, Y)
        return

    def train(self):
        self.result_with_boost()
        self.result_without_boost()

    def split(self,X,Y):
        split = StratifiedShuffleSplit(Y, n_iter=1, test_size=0.2, random_state=0)
        train_index, test_index = list(split)[0]
        self.trainX, self.trainY = X[train_index], Y[train_index]
        self.testX, self.testY = X[test_index], Y[test_index]

    def result_with_boost(self):
        estimator = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=1, random_state=0)
        estimator.fit(self.trainX, self.trainY)
        y_pred = estimator.predict(self.testX)
        print(classification_report(self.testY, y_pred))


    def result_without_boost(self):
        tree_benchmark = DecisionTreeClassifier(max_depth=3, class_weight='auto')
        tree_benchmark.fit(self.trainX, self.trainY)
        y_pred_benchmark = tree_benchmark.predict(self.testX)
        print(classification_report(self.testY, y_pred_benchmark))

        dot_data = tree.export_graphviz(tree_benchmark,class_names=True, out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("wine-quality-red.pdf")
