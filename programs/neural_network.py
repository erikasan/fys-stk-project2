import numpy as np

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
                 output_func=lambda z: np.exp(z)/np.sum(np.exp(z), axis=0, keepdims=True)):

        self.n_nodes          = np.array(n_nodes)
        self.activation_funcs = activation_funcs
        self.output_func      = output_func

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
        the corresponding inputs.

        Set output_only=True if you only want the output of the neural network.
        Set output_false=True to return the activities and activations of all the layers (necessary for backpropagation)

        """
        X = np.array(X)

        weights = self.weights
        bias    = self.bias

        aa = np.zeros((len(self.layers),) + X.shape)
        z = np.zeros((len(self.layers),) + X.shape)
        aa[0] = z[0] = X

        for l in range(1, len(self.layers)):
            layer = self.layers[l]
            z[l]  = weights[l]@aa[l-1] + bias[l]
            aa[l]  = layer.activation_func(z[l])
            if l >= len(self.layers) - 1:
                break

        if output_only:
            return aa[-1]
        else:
            return aa, z


    def back_propagation(self):
        pass


    def predict(self, X):
        probabilities = self.feed_forward(X)
        return np.argmax(probabilities, axis = 0)
