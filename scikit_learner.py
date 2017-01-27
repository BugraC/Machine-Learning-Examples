import scikit_decisiontree
import scikit_neuralnetwork
import numpy as np
import os
import scikit_KNN_Learner
import matplotlib.pyplot as plt
import pandas as pd
from scikit_SVC_learner import support_vector_machines


def main():
    inf = open(os.path.join(os.path.dirname(__file__), 'data/winequality-white.csv'))
    # inf = open(os.path.join(os.path.dirname(__file__), 'data/movie_metadata.csv'))

    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])
    X = data[:, 0:-1]
    Y = data[:, -1]

    print "Decision tree without boosting"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    decisionTree = scikit_decisiontree.decision_tree(X, Y, 5, 0)
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

    print "Decision tree with boosting"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    limit = 5
    decisionTree = None
    for i in range(1, limit):
        decisionTree = scikit_decisiontree.decision_tree(X, Y, 5, i)
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

    print "Neural Network"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    limit = 100
    neuralNetwork = None
    for i in range(1, limit,10):
        print i
        neuralNetwork = scikit_neuralnetwork.neural_network(X,Y,5,i)
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

    print "KNN"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    limit = 100
    knnLearner = None
    for i in range(1,limit):
        knnLearner = scikit_KNN_Learner.KNNLearner(X, Y, i, 5)
        knnLearner.train()
        CVScore1, CVScore2, Accuracy = knnLearner.report()
        CV1_Scores = np.append(CV1_Scores, CVScore1)
        CV2_Scores = np.append(CV2_Scores, CVScore2)
        Accuracy_Scores = np.append(Accuracy_Scores,Accuracy)

    df = pd.DataFrame(data = {'Cross Validation Score with 5 folds' : CV1_Scores, 'Cross Validation Score with 10 folds' : CV2_Scores, 'Accuracy Score:' : Accuracy_Scores }, index=range(1,limit))
    df.plot()
    plt.show()
    knnLearner.plot_learning_curve()

    print "SVC rbf"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    limit = 100
    gammaSpace = np.logspace(-9, 3, 13)
    supportVectorMachines = None
    for i in gammaSpace:
        supportVectorMachines = support_vector_machines(X, Y, 5, "rbf")
        supportVectorMachines.train(0,i)
        CVScore1, CVScore2, Accuracy = supportVectorMachines.report()
        CV1_Scores = np.append(CV1_Scores, CVScore1)
        CV2_Scores = np.append(CV2_Scores, CVScore2)
        Accuracy_Scores = np.append(Accuracy_Scores,Accuracy)
    df = pd.DataFrame(data = {'Cross Validation Score with 5 folds' : CV1_Scores, 'Cross Validation Score with 10 folds' : CV2_Scores, 'Accuracy Score:' : Accuracy_Scores }, index=gammaSpace)
    df.plot()
    supportVectorMachines.plot_learning_curve()

    print "SVC poly"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    limit = 40
    supportVectorMachines = None
    for i in range(1, limit):
        supportVectorMachines = support_vector_machines(X, Y, 5, 'poly')
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




if __name__ == "__main__": main()