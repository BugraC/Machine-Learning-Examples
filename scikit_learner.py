import scikit_decisiontree
import scikit_neuralnetwork
import numpy as np
import os
import scikit_KNN_Learner
import matplotlib.pyplot as plt
import pandas as pd
from scikit_SVC_learner import support_vector_machines_SVC

from scikit_SVR_learner import support_vector_machines_SVR


def main():
    inf = open(os.path.join(os.path.dirname(__file__), 'data/winequality-white.csv'))
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])
    X = data[:, 0:-1]
    Y = data[:, -1]

    print "Decision tree without boosting"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    decisionTree = scikit_decisiontree.decision_tree(X,Y,5)
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

    print "Decision tree with boosting"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    decisionTree = scikit_decisiontree.decision_tree(X, Y,5)
    decisionTree.train_with_boosting()

    CVScore1, CVScore2, Accuracy = decisionTree.report_with_boosting()
    CV1_Scores = np.append(CV1_Scores, CVScore1)
    CV2_Scores = np.append(CV2_Scores, CVScore2)
    Accuracy_Scores = np.append(Accuracy_Scores, Accuracy)

    df = pd.DataFrame(
        data={'Cross Validation Score with 5 folds': CV1_Scores, 'Cross Validation Score with 10 folds': CV2_Scores,
              'Accuracy Score:': Accuracy_Scores}, index=range(1, 5))
    df.plot()
    plt.show()

    print "Neural Network"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    neuralNetwork = scikit_neuralnetwork.neural_network(X,Y,5)
    neuralNetwork.train()
    CVScore1, CVScore2, Accuracy = neuralNetwork.report()
    CV1_Scores = np.append(CV1_Scores, CVScore1)
    CV2_Scores = np.append(CV2_Scores, CVScore2)
    Accuracy_Scores = np.append(Accuracy_Scores, Accuracy)

    df = pd.DataFrame(
        data={'Cross Validation Score with 5 folds': CV1_Scores, 'Cross Validation Score with 10 folds': CV2_Scores,
              'Accuracy Score:': Accuracy_Scores}, index=range(1, 5))
    df.plot()
    plt.show()

    print "KNN"
    CV1_Scores = []
    CV2_Scores = []
    Accuracy_Scores = []
    limit = 100
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

    # supportVectorMachinesSVC = support_vector_machines_SVC(X, Y)
    # supportVectorMachinesSVC.train()
    # supportVectorMachinesSVC.report()
    #
    # supportVectorMachinesSVR = support_vector_machines_SVR(X, Y)
    # supportVectorMachinesSVR.train()
    # supportVectorMachinesSVR.report()


if __name__ == "__main__": main()