import scikit_decisiontree
import scikit_neuralnetwork
import numpy as np
import os
import scikit_KNN_Learner
import matplotlib.pyplot as plt
import pandas as pd
from scikit_SVM_learner import support_vector_machines
import sys


def main():
    # We are using two files here listed:
    # 'data/winequality-white.csv'
    # 'data/movie_metadata.csv'
    dataFile = sys.argv[1]
    isRegressor = (True if sys.argv[2] == 'True' else False)


    inf = os.path.join(os.path.dirname(__file__), dataFile)
    f = pd.read_csv(inf)
    data = pd.DataFrame(f)
    X_data = data.dtypes[data.dtypes != 'object'].index
    X = data[X_data]
    X = X.fillna(0)
    columns = X.columns.tolist()
    Y = X['imdb_score']
    X.drop(['imdb_score'], axis=1, inplace=True)
    X = X.values
    X = np.asarray(X)

    Y = Y.values
    Y = np.asarray(Y)

    # data = genfromtext(inf,float)
    # X = data[:, 0:-1]
    # Y = data[:, -1]

    # print "Decision tree without boosting without pruning"
    # CV1_Scores = []
    # CV2_Scores = []
    # Accuracy_Scores = []
    # decisionTree = scikit_decisiontree.decision_tree(X, Y, 5, 0, isRegressor)
    # decisionTree.train_without_boosting_without_pruning()
    #
    # CVScore1, CVScore2, Accuracy = decisionTree.report_without_boosting_without_pruning()
    # CV1_Scores = np.append(CV1_Scores, CVScore1)
    # CV2_Scores = np.append(CV2_Scores, CVScore2)
    # Accuracy_Scores = np.append(Accuracy_Scores, Accuracy)
    #
    # df = pd.DataFrame(
    #     data={'Cross Validation Score with 5 folds': CV1_Scores, 'Cross Validation Score with 10 folds': CV2_Scores,
    #           'Accuracy Score:': Accuracy_Scores}, index=range(1, 5))
    # df.plot()
    # plt.show()
    # decisionTree.plot_learning_curve_without_boosting_without_pruning()
    # train_decisionTreeErrorWithoutBoostingWithoutPruning,test_decisionTreeErrorWithoutBoostingWithoutPruning = decisionTree.return_error_without_boosting_without_pruning()
    # decisionTree.confusion_matrix_without_boost()

    # print "Decision tree without boosting"
    # CV1_Scores = []
    # CV2_Scores = []
    # Accuracy_Scores = []
    # decisionTree = scikit_decisiontree.decision_tree(X, Y, 5, 0, isRegressor)
    # decisionTree.train_without_boosting()
    #
    # CVScore1, CVScore2, Accuracy = decisionTree.report_without_boosting()
    # CV1_Scores = np.append(CV1_Scores, CVScore1)
    # CV2_Scores = np.append(CV2_Scores, CVScore2)
    # Accuracy_Scores = np.append(Accuracy_Scores, Accuracy)
    #
    # df = pd.DataFrame(
    #     data={'Cross Validation Score with 5 folds': CV1_Scores, 'Cross Validation Score with 10 folds': CV2_Scores,
    #           'Accuracy Score:': Accuracy_Scores}, index=range(1, 5))
    # df.plot()
    # plt.show()
    # decisionTree.plot_learning_curve_without_boosting()
    # train_decisionTreeErrorWithoutPruning,test_decisionTreeErrorWithoutPruning = decisionTree.return_error_without_boosting_without_pruning()
    # decisionTree.confusion_matrix_without_boost()

    # print "Decision tree with boosting"
    # CV1_Scores = []
    # CV2_Scores = []
    # Accuracy_Scores = []
    # limit = 5
    #
    # print "finished training start reporting"
    # for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #     print i
    #     decisionTree = scikit_decisiontree.decision_tree(X, Y, 5, i, isRegressor)
    #     decisionTree.train_with_boosting(True)
    #     CVScore1, CVScore2, Accuracy = decisionTree.report_with_boosting()
    #     CV1_Scores = np.append(CV1_Scores, CVScore1)
    #     CV2_Scores = np.append(CV2_Scores, CVScore2)
    #     Accuracy_Scores = np.append(Accuracy_Scores, Accuracy)
    #
    # df = pd.DataFrame(
    #     data={'Cross Validation Score with 5 folds': CV1_Scores, 'Cross Validation Score with 10 folds': CV2_Scores,
    #           'Accuracy Score:': Accuracy_Scores}, index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #
    # decisionTree = scikit_decisiontree.decision_tree(X, Y, 5, 0, isRegressor)
    # decisionTree.train_with_boosting()
    #
    # df.plot()
    # plt.show()
    # decisionTree.plot_learning_curve_with_boosting()
    # train_decisionTreeWithBoostingError,test_decisionTreeWithBoostingError = decisionTree.return_error_with_boost()
    # decisionTree.confusion_matrix_with_boost()

    # print "Neural Network"
    # CV1_Scores = []
    # CV2_Scores = []
    # Accuracy_Scores = []
    # limit = 100
    #
    # for i in [10,20,40,80]:
    #     print i
    #     neuralNetwork = scikit_neuralnetwork.neural_network(X, Y, 5, i, isRegressor)
    #     neuralNetwork.train(True)
    #     CVScore1, CVScore2, Accuracy = neuralNetwork.report()
    #     CV1_Scores = np.append(CV1_Scores, CVScore1)
    #     CV2_Scores = np.append(CV2_Scores, CVScore2)
    #     Accuracy_Scores = np.append(Accuracy_Scores, Accuracy)
    #
    # df = pd.DataFrame(
    #     data={'Cross Validation Score with 5 folds': CV1_Scores, 'Cross Validation Score with 10 folds': CV2_Scores,
    #           'Accuracy Score:': Accuracy_Scores}, index=[10, 20, 40, 80])
    # df.plot()
    # plt.show()
    #
    # neuralNetwork = scikit_neuralnetwork.neural_network(X, Y, 5, 0, isRegressor)
    # neuralNetwork.train()
    #
    # neuralNetwork.plot_learning_curve()
    # train_neuralNetworkError,test_neuralNetworkError = neuralNetwork.return_error()
    # neuralNetwork.confusion_matrix()

    # print "KNN"
    # CV1_Scores = []
    # CV2_Scores = []
    # Accuracy_Scores = []
    # limit = 100
    # knnLearner = None
    # for i in [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]:
    #     knnLearner = scikit_KNN_Learner.KNNLearner(X, Y, i, 5, isRegressor)
    #     knnLearner.train(True)
    #     CVScore1, CVScore2, Accuracy = knnLearner.report()
    #     CV1_Scores = np.append(CV1_Scores, CVScore1)
    #     CV2_Scores = np.append(CV2_Scores, CVScore2)
    #     Accuracy_Scores = np.append(Accuracy_Scores,Accuracy)
    #
    # df = pd.DataFrame(data = {'Cross Validation Score with 5 folds' : CV1_Scores, 'Cross Validation Score with 10 folds' : CV2_Scores, 'Accuracy Score:' : Accuracy_Scores }, index=[4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150])
    # df.plot()
    # plt.show()
    #
    # knnLearner = scikit_KNN_Learner.KNNLearner(X, Y, 0, 5, isRegressor)
    # knnLearner.train()
    # knnLearner.plot_learning_curve()
    # train_knnLearnerError,test_knnLearnerError = knnLearner.return_error()
    # knnLearner.confusion_matrix()

    # print "SVC rbf"
    # CV1_Scores = []
    # CV2_Scores = []
    # Accuracy_Scores = []
    # limit = 100
    # gammaSpace = np.logspace(-9, 3, 4)
    # supportVectorMachines = None
    # # gammaSpace:
    # for i,j in zip(gammaSpace,[1.0,10.0,50.0,100.0]):
    #     print i
    #     print j
    #     supportVectorMachines = support_vector_machines(X, Y, 5, "rbf",j,i, isRegressor)
    #     supportVectorMachines.train(True)
    #     CVScore1, CVScore2, Accuracy = supportVectorMachines.report()
    #     CV1_Scores = np.append(CV1_Scores, CVScore1)
    #     CV2_Scores = np.append(CV2_Scores, CVScore2)
    #     Accuracy_Scores = np.append(Accuracy_Scores,Accuracy)
    #
    # df = pd.DataFrame(data = {'Cross Validation Score with 5 folds' : CV1_Scores, 'Cross Validation Score with 10 folds' : CV2_Scores, 'Accuracy Score:' : Accuracy_Scores }, index=gammaSpace)
    # df.plot()
    # plt.show()
    #
    # supportVectorMachines = support_vector_machines(X, Y, 5, "rbf",0,0, isRegressor)
    # supportVectorMachines.train()
    # supportVectorMachines.plot_learning_curve()
    # train_supportVectorMachinesRbfError,test_supportVectorMachinesRbfError = supportVectorMachines.return_error()
    # supportVectorMachines.confusion_matrix()


    # print "SVC sigmoid"
    # CV1_Scores = []
    # CV2_Scores = []
    # Accuracy_Scores = []
    # limit = 40
    # gammaSpace =  np.logspace(-9, 3, 4)
    # supportVectorMachines = None
    # for i,j in zip(gammaSpace,[1.0,10.0,50.0,100.0]):
    #     print i
    #     print j
    #     supportVectorMachines = support_vector_machines(X, Y, 5, 'sigmoid',j,i, isRegressor)
    #     supportVectorMachines.train(True)
    #     CVScore1, CVScore2, Accuracy = supportVectorMachines.report()
    #     CV1_Scores = np.append(CV1_Scores, CVScore1)
    #     CV2_Scores = np.append(CV2_Scores, CVScore2)
    #     Accuracy_Scores = np.append(Accuracy_Scores, Accuracy)
    #
    # df = pd.DataFrame(data={'Cross Validation Score with 5 folds': CV1_Scores,
    #                         'Cross Validation Score with 10 folds': CV2_Scores,
    #           'Accuracy Score:': Accuracy_Scores}, index=gammaSpace)
    # df.plot()
    # plt.show()
    # supportVectorMachines = support_vector_machines(X, Y, 5, 'sigmoid', 0, 0, isRegressor)
    # supportVectorMachines.train()
    # supportVectorMachines.plot_learning_curve()
    # train_supportVectorMachinesPolyError,test_supportVectorMachinesPolyError = supportVectorMachines.return_error()
    # supportVectorMachines.confusion_matrix()

    # train_error = [train_decisionTreeErrorWithoutBoostingWithoutPruning,train_decisionTreeErrorWithoutPruning,train_decisionTreeWithBoostingError,train_neuralNetworkError,train_knnLearnerError,train_supportVectorMachinesRbfError,train_supportVectorMachinesPolyError]
    # test_error=[test_decisionTreeErrorWithoutBoostingWithoutPruning,test_decisionTreeErrorWithoutPruning,test_decisionTreeWithBoostingError,test_neuralNetworkError,test_knnLearnerError,test_supportVectorMachinesRbfError,test_supportVectorMachinesPolyError]
    #
    # col={'Train Error':train_error,'Test Error':test_error}
    # models=['Decision Tree Without Boosting and Pruning','Decision Tree Without Pruning','Decision Tree With Boosting And Pruning','Neural Network','KNN','SVM RBF','SVM Poly']
    # df=pd.DataFrame(data=col,index=models)
    # print df

def genfromtext(fname,formatType):
    with open(fname, 'r') as file:
        r = np.array([])
        i = 0
        for line in file:
            row_value = []
            for lineValue in line.split(','):
                try:
                    row_value.append(formatType(lineValue))
                except:
                    pass  # Fail silently on this line since we hit an error
            # r.append([row_value])
            if(i == 0):
                r = np.hstack((r, np.array(row_value)))
            else:
                r = np.vstack((r, np.array(row_value)))
            # np.concatenate((r, row_value), axis=0)
            i += 1
    return r

if __name__ == "__main__": main()