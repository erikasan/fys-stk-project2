import numpy as np
from copy import copy

sigmoid = lambda x: 1/(1 + np.exp(-x))

class Layer:
    """
    Args:

        n_inputs (int)             - number of inputs

        n_nodes  (int)             - number of nodes/outputs

        weights  (numpy.ndarray)   - matrix of weights
                                     must have shape (n_nodes, n_inputs)

        bias     (numpy.ndarray)   - column vector of biases
                                     must have shape (n_nodes, 1)

        activation_func (callable) - activation function of nodes
                                     default is the sigmoid function
    """

    def __init__(self,
                 n_inputs,
                 n_nodes,
                 weights        =None,
                 bias           =None,
                 activation_func=None):

        self.n_inputs        = n_inputs
        self.n_nodes         = n_nodes
        self.weights         = weights
        self.bias            = bias
        self.activation_func = activation_func


    def set_weights(self):
        self.weights = np.random.randn(self.n_nodes, self.n_inputs)
        return self.weights


    def set_bias(self):
        self.bias = 0.01*np.ones((self.n_nodes, 1))
        return self.bias

class NeuralNetwork:
    """
    Args:

        n_inputs    (int)                       - the number of inputs of the neural network

        n_outputs   (int)                       - the number of outputs of the neural network

        output_func (callable)                  - the output function of the neural network
                                                  default is the softmax function

        layers      (array-like, Layer)         - array/list of Layer objects

        n_nodes     (array-like, int)           - array/list of the number of nodes
                                                  the first element is the number of nodes in the first layer
                                                  the second element is the number of nodes in the second layer, and so on

        activation_funcs (array-like, callable) - array/list of functions
                                                  the first element is the activation function of the first layer
                                                  the second element is the activation function of the second layer, and so on
    """

    def __init__(self,
                 n_nodes,
                 activation_funcs=None,
                 output_func=lambda z: np.exp(z)/np.sum(np.exp(z), axis=0, keepdims=True),
                 weights=None,
                 bias   =None):

        self.n_nodes          = np.array(n_nodes)
        self.activation_funcs = activation_funcs
        self.output_func      = output_func

        self.weights = weights
        self.bias    = bias

        if activation_funcs is None:
            activation_funcs     = len(n_nodes)*[sigmoid]
            activation_funcs[0]  = lambda x: x
            activation_funcs[-1] = output_func

        layers = []

        first_layer = Layer(n_inputs=n_nodes[0], n_nodes=n_nodes[0], activation_func=activation_funcs[0])
        layers.append(first_layer)

        for i in range(1, len(n_nodes)):
            layer = Layer(n_inputs=n_nodes[i-1], n_nodes=n_nodes[i], activation_func=activation_funcs[i])
            layers.append(layer)

        self.layers = layers


    def set_weights(self):
        self.weights    = [layer.set_weights() for layer in self.layers]
        self.weights[0] = np.ones(self.n_nodes[0])

    def set_bias(self):
        self.bias    = [layer.set_bias() for layer in self.layers]
        self.bias[0] = np.zeros(self.n_nodes[0])

    def feed_forward(self, X, output_only=True):
        """
        Each column of X is a vector of inputs.
        Returns an activation matrix 'a' where each column is a vector of outputs
        of the corresponding inputs.

        Set output_only=True if you only want the output of the neural network (i.e the activations of the nodes in the last layer).
        Set output_false=True to return the activities and activations of all the layers (necessary for backpropagation)

        """
        X = np.array(X)
        if len(X.shape) < 2:
            X.shape = (X.shape[0], 1)

        weights = self.weights
        bias    = self.bias

        a = len(self.layers)*[0]
        z = len(self.layers)*[0]
        a[0] = z[0] = X

        for l in range(1, len(self.layers)):
            layer = self.layers[l]
            z[l]  = weights[l]@a[l-1] + bias[l]
            a[l]  = layer.activation_func(z[l])

            if l >= len(self.layers) - 1:
                break

        if output_only:
            return a[-1]
        else:
            return a, z


    def back_propagation(self, x, y):
        y = np.array(y)
        weights_gradient = copy(self.weights)
        bias_gradient    = copy(self.bias)
        delta            = copy(self.bias)

        L = len(self.layers)
        aa, z = self.feed_forward(x, output_only=False)

        for j in range(len(aa[-1])):                                     # Loop over neurons in the last layer
            delta[-1][j] = aa[-1][j] - y[j]
            bias_gradient[-1][j] = delta[-1][j]

            for n in range(len(self.weights[-1][j])):                   # Loop over the weights of each neuron in the last layer
                weights_gradient[-1][j, n] = delta[-1][j]*aa[-2][n]

        for l in range(L - 2, 1, -1):                                   # Loop over the other layers
            for j in range(len(aa[l])):                                  # Loop over the neurons
                bias_gradient[l][j] = delta[l][j]

                for n in range(len(self.weights[l][j])):               # Loop over the weights of each neuron
                    weights_gradient[l][j, n] = delta[l][j]*aa[l-1][n]

                delta[l-1][j] = aa[l-1][j]*(1 - aa[l-1][j])*np.sum(delta[l+1]*self.weights[l+1][:][j])

        return weights_gradient, bias_gradient

    def gradient_descent(self, weights_gradient, bias_gradient, eta = 0.01, iterations = 100):
        for l, (weights_grad, bias_grad) in enumerate(zip(weights_gradient[1:], bias_gradient[1:]), start = 1):
            self.weights[l] -= eta*weights_grad
            self.bias[l] -= eta*bias_grad


    def predict(self, X):
        probabilities = self.feed_forward(X)
        return np.argmax(probabilities, axis = 0)
