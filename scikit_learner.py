import scikit_decisiontree
import scikit_neuralnetwork
import numpy as np
import os
import scikit_KNN_Learner

def main():
    inf = open(os.path.join(os.path.dirname(__file__), 'data/winequality-red.csv'))
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])
    X = data[:, 0:-1]
    Y = data[:, -1]
    decisionTree = scikit_decisiontree.decision_tree(X,Y)
    decisionTree.train()

    neuralNetwork = scikit_neuralnetwork.neural_network()
    neuralNetwork.train()


    knnLearner = scikit_KNN_Learner.KNNLearner(X, Y, 5)
    knnLearner.train()



if __name__ == "__main__": main()