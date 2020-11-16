import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0, keepdims=True)

def softmax_derivative(x):
    return softmax(x)*(1 - softmax(x))

def MSE(a, y):
    return np.sum((a - y)**2)/len(a)

def MSE_derivative(a, y):
    return a - y

def cross_entropy():
    pass

def cross_entropy_derivative(a, y):
    return (a - y)/(a*(1 - a))


class NeuralNetwork:

    def __init__(self, layers, mode='classification'):
        """Initialize a NeuralNetwork object.

        Parameters:

            layers : list of ints
                List of the number of nodes in each layer. The first and last
                elements are the number of nodes in the input and output layers respectively.
                The other elements are the number of nodes in the hidden layers.

            functions : list of callables, optional
                List of the activation function of the nodes in each layer. Must have
                length len(layers - 1). The first element is the activation function
                of the first hidden layer, and so on. The last element is the activation
                function of the output layer. By the default the activation function of the
                hidden layers is the sigmoid function, while the activation function
                of the output layer is the softmax function."""

        self.layers     = layers
        self.num_layers = len(layers)

        self.weights    = [np.random.randn(m, n) for m, n in zip(layers[1:], layers[:-1])]
        self.biases     = [0.01*np.ones((m, 1)) for m in layers[1:]]

        assert mode in ['classification', 'regression'], 'mode must be "classification" or "regression"'

        if mode == 'classification':
            functions             = [sigmoid]*(len(layers) - 2)
            functions            += [softmax]
            functions_derivative  = [sigmoid_derivative]*(len(layers) - 2)
            functions_derivative += [softmax_derivative]
            cost_function         = cross_entropy
            cost_derivative       = cross_entropy_derivative

        elif mode == 'regression':
            functions            = [sigmoid]*(len(layers) - 1)
            functions_derivative = [sigmoid_derivative]*(len(layers) - 1)
            cost_function        = MSE
            cost_derivative      = MSE_derivative

        self.mode                 = mode
        self.functions            = functions
        self.functions_derivative = functions_derivative
        self.cost_function        = cost_function
        self.cost_derivative      = cost_derivative


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
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for k, mini_batch in enumerate(mini_batches):
                self.GD(mini_batch, eta)
            print(f'epoch {j+1}/{epochs} complete')

    def GD(self, mini_batch, eta):
        """Docstring to be updated."""
        mini_batch_size = len(mini_batch)
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - eta*nw/mini_batch_size for w, nw in zip(self.weights, nabla_w)]
        self.bias    = [b - eta*nb/mini_batch_size for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Docstring to be updated."""

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        a = [x]
        z = [x]
        for w, b, f in zip(self.weights, self.biases, self.functions):
            z.append(w@a[-1] + b)
            a.append(f(z[-1]))

        if self.mode == 'classification':
            delta = a[-1] - y
        elif self.mode == 'regression':
            delta = self.cost_derivative(a[-1], y)*self.functions_derivative[-1](z[-1])

        nabla_w[-1] = delta@a[-2].T
        nabla_b[-1] = delta
        for l in range(self.num_layers-2, 0, -1):
            delta = self.weights[l].T@delta * self.functions_derivative[l](z[l])
            nabla_w[l-1] = delta@a[l-1].T
            nabla_b[l-1] = delta
        return nabla_w, nabla_b

    def predict(self, x):
        a = self.feedforward(x)
        return np.argmax(a, axis = 0)

    def evaluate(self, test_data):
        test_results = [(self.predict(x), y) for x, y in test_data]
        return np.sum([int(x == y) for x, y in test_results])
