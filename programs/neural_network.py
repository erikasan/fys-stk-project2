import numpy as np

# Each node in the first hidden layer has a number of weights
# equal to the number of inputs

class NeuralNetwork:

    def __init__(self,
                 W = None,
                 b = None,
                 activation_func = lambda x: 1/(1 + np.exp(-x)),
                 output_func = None):
        """
        Args:
            W (np.ndarray)             - multidimensional array of weights.
                                         W must have shape (num_nodes, num_weights_per_node, num_hidden_layers)

            b (np.ndarray)             - multidimensional array of biases
                                         b must have shape (num_nodes, num_hidden_layers)

            activation_func (callable) - activation function of the nodes in the hidden layers
                                         default activation function is the sigmoid function

            output_func (callable)     - output function. */()&¤#¤%#""&¤¤%#
        """

        self.W = W
        self.b = b
        self.activation_func = activation_func
        self.output_func = output_func

    def set_weights(self,
                    num_nodes,
                    num_weights_per_node,
                    num_hidden_layers):

        self.W = np.random.randn(num_nodes,
                                 num_weights_per_node,
                                 num_hidden_layers)


    def set_biases(self,
                   num_nodes,
                   num_hidden_layers):

        self.b = 0.01*np.ones((num_nodes, num_hidden_layers))


    def set_activation_function(self,
                                activation_func):

        self.activation_func = activation_func


    def set_output_function(self,
                            output_func):

        self.output_func = output_func

    def feed_forward(self, X):

        W = self.W
        b = self.b
        activation_func = self.activation_func
        num_hidden_layers = W.shape[2]

        a = X
        for l in range(num_hidden_layers):
            z = W[:, :, l]@a + b[:, l]
            a = activation_func(z)

        return output_func(a)


    def return_model(self):
        return self.feed_forward


    def back_propagation(self):
        pass
