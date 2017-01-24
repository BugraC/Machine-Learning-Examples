import matplotlib.pyplot as plt

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from learning_curve import plot_learning_curve


class decision_tree:
    def __init__(self,X,Y,KFolds,NEstimators):
        self.KFolds = KFolds
        self.X = X
        self.Y = Y
        self.split(X, Y)
        self.NEstimators = NEstimators
        return


    def split(self,X,Y):
        split = StratifiedShuffleSplit(Y, n_iter=100, test_size=0.3)
        train_index, test_index = list(split)[0]
        self.trainX, self.trainY = X[train_index], Y[train_index]
        self.testX, self.testY = X[test_index], Y[test_index]

    def train_with_boosting(self):
        print self.NEstimators
        self.estimator = GradientBoostingClassifier(n_estimators=self.NEstimators, learning_rate=0.1)
        self.estimator.fit(self.trainX, self.trainY)
        self.y_pred_with_boost = self.estimator.predict(self.testX)



    def train_without_boosting(self):
        self.tree_benchmark = DecisionTreeClassifier(class_weight='auto')
        self.tree_benchmark.fit(self.trainX, self.trainY)
        self.y_pred_benchmark = self.tree_benchmark.predict(self.testX)


    def report_with_boosting(self):
        # print(classification_report(self.testY, self.y_pred_with_boost))
        CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.estimator , self.testX, self.testY,
                                                               cv=self.KFolds).mean(), cross_val_score(self.estimator,
                                                                                                       self.testX,
                                                                                                       self.testY,
                                                                                                       cv=self.KFolds * 2).mean(), accuracy_score(
            self.testY, self.y_pred_with_boost)

        return CV_Score1, CV_Score2, Accuracy_Score

    def report_without_boosting(self):
        # print(classification_report(self.testY, self.y_pred_benchmark))
        #
        # dot_data = tree.export_graphviz(self.tree_benchmark, class_names=True, out_file=None)
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # graph.write_pdf("wine-quality-red.pdf")
        CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.tree_benchmark, self.testX, self.testY,
                                                               cv=self.KFolds).mean(), cross_val_score(self.tree_benchmark,
                                                                                                       self.testX,
                                                                                                       self.testY,
                                                                                                       cv=self.KFolds * 2).mean(), accuracy_score(
            self.testY, self.y_pred_benchmark)

        return CV_Score1, CV_Score2, Accuracy_Score

    def plot_learning_curve_without_boosting(self):
        plot_learning_curve(self.tree_benchmark, 'Learning Curves Decision Tree', self.X, self.Y, ylim=(0.1, 1.01), cv=5, n_jobs=4)
        plt.show()


    def plot_learning_curve_with_boosting(self):
        plot_learning_curve(self.estimator, 'Learning Curves Decision Tree With Boosting', self.X, self.Y, ylim=(0.1, 1.01), cv=5, n_jobs=4)
        plt.show()

