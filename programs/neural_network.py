import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0, keepdims=True)


class NeuralNetwork:

    def __init__(self, layers, functions=None):
        """Initialize a NeuralNetwork object.

        Parameters:

            layers : list of ints
                List of the number of nodes in each layer. The first and last
                elements are the number of nodes in the input and output layers respectively.
                The other elements are the number of nodes in the hidden layers.

            functions : list of callables, optional
                List of the activation function of the nodes in each layer. Must have
                length len(layers - 1). The first element is the activation function
                of the first hidden layer. The last element is the activation function
                of the output layer. By the default the activation function of the
                hidden layers is the sigmoid function, while the activation function
                of the output layer is the softmax function."""

        self.layers     = layers
        self.num_layers = len(layers)

        self.weights    = [np.random.randn(m, n) for m, n in zip(layers[1:], layers[:-1])]
        self.biases     = [0.01*np.ones((m, 1)) for m in layers[1:]]

        if functions is None:
            functions  = [sigmoid]*(len(layers) - 2)
            functions += [softmax]
        self.functions = functions

        def feedforward(self, a):
            """Return the output of the neural network.

            Parameters:

                a : ndarray
                    Input column vector. Must have shape (m, 1) where
                    m is the number of nodes in the input layer. Alternatively
                    several inputs can be fed forward simultaneously by providing
                    an ndarray of shape (m, n)

            Returns:

                out : ndarray
                    The output of the neural network. If an input of shape (m, n)
                    is provided, an output of shape (M, n) is given where M is
                    the number of nodes in the output layer."""

            for w, b, f in zip(self.weights, self.biases, self.functions):
                a = f(w@a + b)
            return a

        def SGD(self, training_data, epochs, mini_batch_size, eta):
            """Docstring to be updated.

            Parameters:

                training_data : list of tuples of ndarrays
                    List on the form [(x1, y1), (x2, y2), ...] where e.g.
                    x1, y1 are column vectors of an input and the corresponding desired
                    output of the neural network respectively.

                epochs : int
                    The number of epochs.

                mini_batch_size : int
                    The size of each mini_batch.

                eta : int or float
                    The learning rate when applying gradient descent."""

            n = len(training_data)
            for j in range(epochs):
                mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                for mini_batch in mini_batches:
                    self.GD(mini_batch, eta)

        def GD(self):
            pass

        def backprop(self):
            pass
