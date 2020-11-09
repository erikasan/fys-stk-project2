import numpy as np
from neural_network import NeuralNetwork


n_nodes = [3, 1]
NN = NeuralNetwork(n_nodes = n_nodes)
NN.set_weights()
NN.set_bias()

X = np.array([1, 1, 1])
X.shape = (3, 1)

res = NN.feed_forward(X)
