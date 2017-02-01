import matplotlib.pyplot as plt

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit
from learning_curve import plot_learning_curve
from scikit_prunedregressiontree import dtclf_pruned_regressor
from scikit_prunedtree import dtclf_pruned
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

class decision_tree:
    def __init__(self,X,Y,KFolds,NEstimators,isRegression):
        self.KFolds = KFolds
        self.X = X
        self.Y = Y
        self.isRegression = isRegression
        self.split(X, Y)
        self.NEstimators = NEstimators

        return


    def split(self,X,Y):
        # split = None
        # if(self.isRegression):
        # number_of_samples = len(Y)
        # np.random.seed(0)
        # random_indices = np.random.permutation(number_of_samples)
        # num_training_samples = int(number_of_samples * 0.75)
        # self.trainX = X[random_indices[:num_training_samples]]
        # y_Train = Y[random_indices[:num_training_samples]]
        # self.testX = X[random_indices[num_training_samples:]]
        # self.testY = Y[random_indices[num_training_samples:]]
        # self.trainY = list(y_Train)

        self.trainX, self.testX, self.trainY, self.testY  = train_test_split(X, Y, test_size=0.30, random_state=2)

            # split = ShuffleSplit(n_splits=100,random_state=123, test_size=0.3)
            # for train_index, test_index in split.split(self.X, self.Y):
            #     self.trainX = self.X[train_index]
            #     self.trainY = self.Y[train_index]
            #     self.testX = self.X[test_index]
            #     self.testY = self.Y[test_index]
        # else:
        #     split = StratifiedShuffleSplit(Y, n_iter=100, test_size=0.3)
        #     train_index, test_index = list(split)[0]
        #     self.trainX, self.trainY = X[train_index], Y[train_index]
        #     self.testX, self.testY = X[test_index], Y[test_index]

    def train_with_boosting(self):
        rng = np.random.RandomState(1)
        print self.NEstimators
        if (self.isRegression):
            self.estimator = AdaBoostRegressor(base_estimator=dtclf_pruned_regressor(),random_state=rng,n_estimators=self.NEstimators, learning_rate=0.1)
        else:
            self.estimator = AdaBoostClassifier(base_estimator=dtclf_pruned(class_weight='auto'), random_state=rng,
                                                n_estimators=self.NEstimators, learning_rate=0.1)

        parameters = {'base_estimator__criterion': ['gini'],
                      # 'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],
                      # "base_estimator__splitter": ["best", "random"],
                      "n_estimators": [1,2,3,4,5,6,7,8,9,10]
                      }
        self.clf_tuned = GridSearchCV(self.estimator, cv=5, param_grid=parameters)
        self.tree = self.clf_tuned.fit(self.trainX, self.trainY)
        print('Best parameters: %s' % self.clf_tuned.best_params_)
        # self.estimator.fit(self.X, self.Y)
        self.tree =self.clf_tuned.best_estimator_.estimators_[-1]
        self.testX = self.tree.valX
        self.testY = self.tree.valY
        self.trainX = self.tree.trgX
        self.trainY = self.tree.trgY

        self.y_pred_with_boost = self.clf_tuned.best_estimator_.predict(self.testX)
        self.y_pred_training_with_boost = self.clf_tuned.best_estimator_.predict(self.trainX)





    def train_without_boosting(self):
        if(self.isRegression):
            self.tree_benchmark = dtclf_pruned_regressor()
        else:
            self.tree_benchmark = dtclf_pruned(class_weight='auto')

        parameters = {'criterion': ['gini', 'entropy'],
                      'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}
        self.clf_tuned = GridSearchCV(self.tree_benchmark, cv=5, param_grid=parameters)

        self.tree = self.clf_tuned.fit(self.trainX, self.trainY)
        print('Best parameters: %s' % self.clf_tuned.best_params_)
        self.testX = self.tree.best_estimator_.valX
        self.testY = self.tree.best_estimator_.valY
        self.trainX = self.tree.best_estimator_.trgX
        self.trainY = self.tree.best_estimator_.trgY

        self.y_pred_benchmark = self.clf_tuned.best_estimator_.predict(self.testX)
        self.y_pred_training_without_boost = self.clf_tuned.best_estimator_.predict(self.trainX)




    def train_without_boosting_without_pruning(self):
        if (self.isRegression):
            self.tree_benchmark = DecisionTreeRegressor(max_depth=1)
        else:
            self.tree_benchmark = DecisionTreeClassifier(class_weight='balanced')
        # self.tree = self.tree_benchmark.fit(self.trainX, self.trainY)

        # self.y_pred_benchmark = self.tree_benchmark.predict(self.testX)
        # self.y_pred_training_without_boost = self.tree_benchmark.predict(self.trainX)


        # parameters = {'class_weight': ('balanced')}
        parameters = {'criterion': ['gini', 'entropy'],
                     'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}
        self.clf_tuned = GridSearchCV(self.tree_benchmark, cv=5, param_grid=parameters)
        self.clf_tuned.fit(self.trainX, self.trainY)
        print('Best parameters: %s' % self.clf_tuned.best_params_)
        self.y_pred_benchmark = self.clf_tuned.best_estimator_.predict(self.testX)
        self.y_pred_training_without_boost = self.clf_tuned.best_estimator_.predict(self.trainX)

    def report_without_boosting_without_pruning(self):
        if (self.isRegression):
            # print(classification_report(self.testY, self.y_pred_with_boost))
            CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.clf_tuned.best_estimator_, self.testX, self.testY,
                                                                   cv=self.KFolds).mean(), cross_val_score(
                self.clf_tuned.best_estimator_,
                self.testX,
                self.testY,
                cv=self.KFolds * 2).mean(), self.tree_benchmark.score(
                self.testY, self.y_pred_benchmark)

            return CV_Score1, CV_Score2, Accuracy_Score
        else:
            # print(classification_report(self.testY, self.y_pred_with_boost))
            CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.clf_tuned.best_estimator_, self.testX, self.testY,
                                                                   cv=self.KFolds, scoring='accuracy').mean(), cross_val_score(self.clf_tuned.best_estimator_,
                                                                                                           self.testX,
                                                                                                           self.testY,
                                                                                                           cv=self.KFolds * 2, scoring='accuracy').mean(), \
                                                   accuracy_score(self.testY, self.y_pred_benchmark)

            return CV_Score1, CV_Score2, Accuracy_Score

    def report_with_boosting(self):
        rng = np.random.RandomState(1)
        if (self.isRegression):
            self.estimator = AdaBoostRegressor(base_estimator=dtclf_pruned_regressor(), random_state=rng,
                                               n_estimators=self.NEstimators, learning_rate=0.1)
            # print(classification_report(self.testY, self.y_pred_with_boost))
            CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.estimator, self.testX, self.testY,
                                                                   cv=self.KFolds).mean(), cross_val_score(
                self.estimator,
                self.testX,
                self.testY,
                cv=self.KFolds * 2).mean(), accuracy_score(
                self.testY, self.y_pred_benchmark)

            return CV_Score1, CV_Score2, Accuracy_Score
        else:
            self.estimator = AdaBoostClassifier(base_estimator=dtclf_pruned(class_weight='auto'), random_state=rng,
                                                n_estimators=self.NEstimators, learning_rate=0.1)
            # print(classification_report(self.testY, self.y_pred_with_boost))
            CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.estimator, self.testX, self.testY,
                                                                   cv=self.KFolds).mean(), cross_val_score(self.estimator,
                                                                                                           self.testX,
                                                                                                           self.testY,
                                                                                                           cv=self.KFolds * 2).mean(), accuracy_score(
                self.testY, self.y_pred_benchmark)

            return CV_Score1, CV_Score2, Accuracy_Score

    def report_without_boosting(self):
        if (self.isRegression):
            CV_Score1, CV_Score2, Accuracy_Score = cross_val_score(self.tree_benchmark, self.testX, self.testY,
                                                                   cv=self.KFolds).mean(), cross_val_score(
                self.tree_benchmark,
                self.testX,
                self.testY,
                cv=self.KFolds * 2).mean(), accuracy_score(
                self.testY, self.y_pred_benchmark)

            return CV_Score1, CV_Score2, Accuracy_Score
        else:
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
        plot_learning_curve(self.clf_tuned.best_estimator_, 'Learning Curves Decision Tree', self.trainX, self.trainY, ylim=(0.0, 1.01), cv=5, n_jobs=4)
        plt.show()

    def plot_learning_curve_without_boosting_without_pruning(self):
        # plot_learning_curve(self.tree_benchmark, 'Learning Curves Decision Tree', self.trainX, self.trainY, ylim=(0.0, 1.01),
        #                         cv=5, n_jobs=4)

        plot_learning_curve(self.clf_tuned.best_estimator_, 'Learning Curves Decision Tree', self.trainX, self.trainY, ylim=(0.0, 1.01))
        # plot_learning_curve(self.tree_benchmark, 'Learning Curves Decision Tree', self.trainX, self.trainY,
        #                     train_sizes=np.linspace(.05, 1.0, 5))

        plt.show()


    def plot_learning_curve_with_boosting(self):
        plot_learning_curve(self.clf_tuned.best_estimator_, 'Learning Curves Decision Tree With Boosting', self.trainX, self.trainY, ylim=(0.0, 1.01), cv=5, n_jobs=4)
        plt.show()


    def return_error_without_boosting_without_pruning(self):
        error = 0.0
        for i in range(len(self.trainY)):
            error += (abs(self.y_pred_training_without_boost[i] - self.trainY[i]) / self.trainY[i])
        error_train_percent = error / len(self.trainY) * 100
        print("Train error = "'{}'.format(error_train_percent) + " percent")

        error = 0.0
        for i in range(len(self.testY)):
            error += (abs(self.y_pred_benchmark[i] - self.testY[i]) / self.testY[i])
        error_test_percent = error / len(self.testY) * 100
        print("Test error = "'{}'.format(error_test_percent) + " percent")
        return error_train_percent,error_test_percent

    def return_error_with_boost(self):
        error = 0.0
        for i in range(len(self.trainY)):
            error += (abs(self.y_pred_training_with_boost[i] - self.trainY[i]) / self.trainY[i])
        error_train_percent = error / len(self.trainY) * 100
        print("Train error = "'{}'.format(error_train_percent) + " percent")

        error = 0.0
        for i in range(len(self.testY)):
            error += (abs(self.y_pred_with_boost[i] - self.testY[i]) / self.testY[i])
        error_test_percent = error / len(self.testY) * 100
        print("Test error = "'{}'.format(error_test_percent) + " percent")
        return error_train_percent,error_test_percent