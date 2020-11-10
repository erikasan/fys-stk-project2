import numpy as np
from neural_network import NeuralNetwork


n_nodes = [3, 5, 3]
NN = NeuralNetwork(n_nodes = n_nodes)
NN.set_weights()
NN.set_bias()

X = np.zeros((3, 1))
X[:, 0] = [1, 2, 3]

y = np.zeros((3, 1))
y[:, 0] = [2, 3, 4]

NN.back_propagation(X, y)
