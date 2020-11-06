import numpy as np

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
                 weights = None,
                 bias    = None,
                 activation_func = lambda x: 1/(1 + np.exp(-x))):

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
                 n_inputs,
                 output_func = lambda z: np.exp(z)/np.sum(np.exp(z)),
                 layers           = None,
                 n_nodes          = None,
                 weights          = None,
                 activation_funcs = None):

        self.n_inputs    = n_inputs
        self.output_func = output_func
        self.layers      = np.array(layers)
        self.n_nodes     = np.array(n_nodes)

        if n_nodes is not None:
            layers = []
            for n_node in n_nodes:
                layer = Layer(n_inputs, n_node)
                layers.append(layer)
            self.layers = np.array(layers)

        if activation_funcs is not None:
            pass

    def set_weights(self):
        weights      = [layer.set_weights() for layer in self.layers]
        self.weights = np.array(weights)

    def set_bias(self):
        bias      = [layer.set_bias() for layer in self.layers]
        self.bias = np.array(bias)


    def feed_forward(self, X):
        X = np.array(X)

        if self.layers is None:
            return output_func(X)

        weights = self.weights
        bias    = self.bias

        a = X
        for l, layer in enumerate(self.layers):
            z = weights[l]@a + bias[l]
            a = layer.activation_func(z)

        return self.output_func(a)
