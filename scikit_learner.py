import scikit_decisiontree
import scikit_neuralnetwork
import numpy as np
import os
import scikit_KNN_Learner
from scikit_SVC_learner import support_vector_machines_SVC

from scikit_SVR_learner import support_vector_machines_SVR


def main():
    inf = open(os.path.join(os.path.dirname(__file__), 'data/winequality-white.csv'))
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])
    X = data[:, 0:-1]
    Y = data[:, -1]
    decisionTree = scikit_decisiontree.decision_tree(X,Y)
    decisionTree.train()

    neuralNetwork = scikit_neuralnetwork.neural_network(X,Y)
    neuralNetwork.train()


    knnLearner = scikit_KNN_Learner.KNNLearner(X, Y, 5)
    knnLearner.train()

    supportVectorMachinesSVC = support_vector_machines_SVC(X, Y)
    supportVectorMachinesSVC.train()

    supportVectorMachinesSVR = support_vector_machines_SVR(X, Y)
    supportVectorMachinesSVR.train()


if __name__ == "__main__": main()