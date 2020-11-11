import numpy as np
from neural_network import NeuralNetwork


n_nodes = [1, 1]
NN = NeuralNetwork(n_nodes = n_nodes, output_func = lambda x: x)
NN.set_weights()
NN.set_bias()

def f(x):
    return np.pi + np.exp(1)*x + 42*x**2

x = [0]
y = [f(0)]

weights_gradient, bias_gradient = NN.back_propagation(x, y)

NN.gradient_descent(weights_gradient, bias_gradient, iterations = 100000)
