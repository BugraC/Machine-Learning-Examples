from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
import pydotplus

class neural_network:
    def __init__(self):
        return

    def train(self):
        iris = load_iris()
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        clf.fit(iris.data, iris.target)
        # dot_data = MLPClassifier.export_graphviz(clf, out_file=None)
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # graph.write_pdf("iris_neuralnetwork.pdf")