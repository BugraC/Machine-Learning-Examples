import scikit_decisiontree
import scikit_neuralnetwork
import numpy as np
import os
import scikit_KNN_Learner
import matplotlib.pyplot as plt
import pandas as pd
from scikit_SVC_learner import support_vector_machines
import sys


def main():
    dataFile = sys.argv[0]
    isRegressor = False
    if(dataFile == 'data/winequality-white.csv'):
        isRegressor = False
    elif(dataFile == 'data/movie_metadata.csv'):
        isRegressor = True


    # inf = open(os.path.join(os.path.dirname(__file__), 'data/winequality-white.csv'))
    inf = os.path.join(os.path.dirname(__file__), dataFile)
    # inf = os.path.join(os.path.dirname(__file__), 'data/winequality-white.csv')

    data = genfromtext(inf,float)

    X = data[:, 0:-1]
    Y = data[:, -1]

    print "Decision tree without boosting without pruning"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    decisionTree = scikit_decisiontree.decision_tree(X, Y, 5, 0, isRegressor)
    decisionTree.train_without_boosting_without_pruning()

    CVScore1, CVScore2, Accuracy = decisionTree.report_without_boosting_without_pruning()
    CV1_Scores = np.append(CV1_Scores, CVScore1)
    CV2_Scores = np.append(CV2_Scores, CVScore2)
    Accuracy_Scores = np.append(Accuracy_Scores, Accuracy)

    df = pd.DataFrame(
        data={'Cross Validation Score with 5 folds': CV1_Scores, 'Cross Validation Score with 10 folds': CV2_Scores,
              'Accuracy Score:': Accuracy_Scores}, index=range(1, 5))
    df.plot()
    plt.show()
    decisionTree.plot_learning_curve_without_boosting_without_pruning()
    train_decisionTreeErrorWithoutBoostingWithoutPruning,test_decisionTreeErrorWithoutBoostingWithoutPruning = decisionTree.return_error_without_boosting_without_pruning()


    print "Decision tree without boosting"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    decisionTree = scikit_decisiontree.decision_tree(X, Y, 5, 0, isRegressor)
    decisionTree.train_without_boosting()

    CVScore1, CVScore2, Accuracy = decisionTree.report_without_boosting()
    CV1_Scores = np.append(CV1_Scores, CVScore1)
    CV2_Scores = np.append(CV2_Scores, CVScore2)
    Accuracy_Scores = np.append(Accuracy_Scores, Accuracy)

    df = pd.DataFrame(
        data={'Cross Validation Score with 5 folds': CV1_Scores, 'Cross Validation Score with 10 folds': CV2_Scores,
              'Accuracy Score:': Accuracy_Scores}, index=range(1, 5))
    df.plot()
    plt.show()
    decisionTree.plot_learning_curve_without_boosting()
    train_decisionTreeErrorWithoutPruning,test_decisionTreeErrorWithoutPruning = decisionTree.return_error_without_boosting_without_pruning()

    print "Decision tree with boosting"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    limit = 5
    decisionTree = None
    for i in range(1, limit):
        decisionTree = scikit_decisiontree.decision_tree(X, Y, 5, i, isRegressor)
        decisionTree.train_with_boosting()

        CVScore1, CVScore2, Accuracy = decisionTree.report_with_boosting()
        CV1_Scores = np.append(CV1_Scores, CVScore1)
        CV2_Scores = np.append(CV2_Scores, CVScore2)
        Accuracy_Scores = np.append(Accuracy_Scores, Accuracy)

    df = pd.DataFrame(
        data={'Cross Validation Score with 5 folds': CV1_Scores, 'Cross Validation Score with 10 folds': CV2_Scores,
              'Accuracy Score:': Accuracy_Scores}, index=range(1, limit))
    df.plot()
    plt.show()
    decisionTree.plot_learning_curve_with_boosting()
    train_decisionTreeWithBoostingError,test_decisionTreeWithBoostingError =  decisionTree.return_error_with_boost()

    print "Neural Network"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    limit = 100
    neuralNetwork = None
    for i in range(1, limit,10):
        print i
        neuralNetwork = scikit_neuralnetwork.neural_network(X,Y,5,i, isRegressor)
        neuralNetwork.train()
        CVScore1, CVScore2, Accuracy = neuralNetwork.report()
        CV1_Scores = np.append(CV1_Scores, CVScore1)
        CV2_Scores = np.append(CV2_Scores, CVScore2)
        Accuracy_Scores = np.append(Accuracy_Scores, Accuracy)

    df = pd.DataFrame(
        data={'Cross Validation Score with 5 folds': CV1_Scores, 'Cross Validation Score with 10 folds': CV2_Scores,
              'Accuracy Score:': Accuracy_Scores}, index=range(1, 11))
    df.plot()
    plt.show()
    neuralNetwork.plot_learning_curve()
    train_neuralNetworkError,test_neuralNetworkError = neuralNetwork.return_error()

    print "KNN"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    limit = 100
    knnLearner = None
    for i in range(1,limit):
        knnLearner = scikit_KNN_Learner.KNNLearner(X, Y, i, 5, isRegressor)
        knnLearner.train()
        CVScore1, CVScore2, Accuracy = knnLearner.report()
        CV1_Scores = np.append(CV1_Scores, CVScore1)
        CV2_Scores = np.append(CV2_Scores, CVScore2)
        Accuracy_Scores = np.append(Accuracy_Scores,Accuracy)

    df = pd.DataFrame(data = {'Cross Validation Score with 5 folds' : CV1_Scores, 'Cross Validation Score with 10 folds' : CV2_Scores, 'Accuracy Score:' : Accuracy_Scores }, index=range(1,limit))
    df.plot()
    plt.show()
    knnLearner.plot_learning_curve()
    train_knnLearnerError,test_knnLearnerError = knnLearner.return_error()

    print "SVC rbf"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    limit = 100
    gammaSpace = np.logspace(-9, 3, 13)
    supportVectorMachines = None
    for i in gammaSpace:
        supportVectorMachines = support_vector_machines(X, Y, 5, "rbf", isRegressor)
        supportVectorMachines.train(0,i)
        CVScore1, CVScore2, Accuracy = supportVectorMachines.report()
        CV1_Scores = np.append(CV1_Scores, CVScore1)
        CV2_Scores = np.append(CV2_Scores, CVScore2)
        Accuracy_Scores = np.append(Accuracy_Scores,Accuracy)
    df = pd.DataFrame(data = {'Cross Validation Score with 5 folds' : CV1_Scores, 'Cross Validation Score with 10 folds' : CV2_Scores, 'Accuracy Score:' : Accuracy_Scores }, index=gammaSpace)
    df.plot()
    supportVectorMachines.plot_learning_curve()
    train_supportVectorMachinesRbfError,test_supportVectorMachinesRbfError = supportVectorMachines.return_error()


    print "SVC poly"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    limit = 40
    supportVectorMachines = None
    for i in range(1, limit):
        supportVectorMachines = support_vector_machines(X, Y, 5, 'poly', isRegressor)
        supportVectorMachines.train(i,0)
        CVScore1, CVScore2, Accuracy = supportVectorMachines.report()
        CV1_Scores = np.append(CV1_Scores, CVScore1)
        CV2_Scores = np.append(CV2_Scores, CVScore2)
        Accuracy_Scores = np.append(Accuracy_Scores, Accuracy)

    df = pd.DataFrame(data={'Cross Validation Score with 5 folds': CV1_Scores,
                            'Cross Validation Score with 10 folds': CV2_Scores,
              'Accuracy Score:': Accuracy_Scores}, index=range(1, limit))
    df.plot()
    supportVectorMachines.plot_learning_curve()
    train_supportVectorMachinesPolyError,test_supportVectorMachinesPolyError = supportVectorMachines.return_error()

    train_error = [train_decisionTreeErrorWithoutBoostingWithoutPruning,train_decisionTreeErrorWithoutPruning,train_decisionTreeWithBoostingError,train_neuralNetworkError,train_knnLearnerError,train_supportVectorMachinesRbfError,train_supportVectorMachinesPolyError]
    test_error=[test_decisionTreeErrorWithoutBoostingWithoutPruning,test_decisionTreeErrorWithoutPruning,test_decisionTreeWithBoostingError,test_neuralNetworkError,test_knnLearnerError,test_supportVectorMachinesRbfError,test_supportVectorMachinesPolyError]

    col={'Train Error':train_error,'Test Error':test_error}
    models=['Decision Tree Without Boosting and Pruning','Decision Tree Without Pruning','Decision Tree With Boosting And Pruning','Neural Network','KNN','SVM RBF','SVM Poly']
    df=pd.DataFrame(data=col,index=models)
    print df

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