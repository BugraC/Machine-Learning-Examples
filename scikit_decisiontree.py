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


    def split(self,X,Y):
        split = StratifiedShuffleSplit(Y, n_iter=1, test_size=0.2, random_state=0)
        train_index, test_index = list(split)[0]
        self.trainX, self.trainY = X[train_index], Y[train_index]
        self.testX, self.testY = X[test_index], Y[test_index]

    def train_with_boosting(self):
        estimator = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=1, random_state=0)
        estimator.fit(self.trainX, self.trainY)
        self.y_pred_with_boost = estimator.predict(self.testX)



    def train_without_boosting(self):
        self.tree_benchmark = DecisionTreeClassifier(max_depth=3, class_weight='auto')
        self.tree_benchmark.fit(self.trainX, self.trainY)
        self.y_pred_benchmark = self.tree_benchmark.predict(self.testX)

    def report_with_boosting(self):
        print(classification_report(self.testY, self.y_pred_with_boost))

    def report_without_boosting(self):
        print(classification_report(self.testY, self.y_pred_benchmark))

        dot_data = tree.export_graphviz(self.tree_benchmark, class_names=True, out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("wine-quality-red.pdf")

