from neural_network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# training_data = training_data[:1000]
# test_data = test_data[:100]



@np.vectorize
def sigmoid(x):
    if x < 0:
        return np.exp(x)/(1 + np.exp(x))
    else:
        return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

@np.vectorize
def relu_derivative(x):
    if x <= 0:
        return 0
    else:
        return 1

def leaky(x):
    if x < -10:
        return 0.01*(-10)
    else:
        return np.maximum(0.01*x, x)

@np.vectorize
def leaky_derivative(x):
    if x < -10:
        return 0
    elif x <= 0:
        return 0.01
    else:
        return 1

# For classification only
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0, keepdims=True)
# For classification only
def softmax_derivative(x):
    return softmax(x)*(1 - softmax(x))
# For classification only
def cross_entropy_derivative(a, y):
    return (a - y)/(a*(1 - a))


sigmoid_functions    = [sigmoid]*(len(layers) - 2)
sigmoid_functions   += [softmax]
sigmoid_derivatives  = [sigmoid_derivative]*(len(layers) - 2)
sigmoid_derivatives += [softmax_derivative]

relu_functions     = [relu]*(len(layers) - 2)
relu_functions    += [softmax]
relu_derivatives   = [relu_derivative]*(len(layers) - 2)
relu_derivatives  += [softmax_derivative]

leaky_functions    = [leaky]*(len(layers) - 2)
leaky_functions   += [softmax]
leaky_derivatives  = [leaky_derivative]*(len(layers) - 2)
leaky_derivatives += [softmax_derivative]


def accuracy_vs_hidden_layers():
    num_hidden_layers = np.arange(5)
    sigmoid_accuracy  = np.zeros(len(num_hidden_layers))
    relu_accuracy     = np.zeros(len(num_hidden_layers))
    leaky_accuracy    = np.zeros(len(num_hidden_layers))
    for i in num_hidden_layers:
        layers  = [784]
        layers += i*[30]
        layers += [10]

        net = NeuralNetwork(layers=layers, functions=sigmoid_functions, functions_derivatives=sigmoid_derivatives, cost_derivative=cross_entropy_derivative, mode='classification')
        net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, lmbda=0)
        sigmoid_accuracy[i] = net.evaluate(test_data)/len(test_data)*100

        # net = NeuralNetwork(layers=layers, functions=relu_functions, functions_derivatives=relu_derivatives, cost_derivative=cross_entropy_derivative, mode='classification')
        # net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, lmbda=0)
        # relu_accuracy[i] = net.evaluate(test_data)/len(test_data)*100
        #
        # net = NeuralNetwork(layers=layers, functions=leaky_functions, functions_derivatives=leaky_derivatives, cost_derivative=cross_entropy_derivative, mode='classification')
        # net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, lmbda=0)
        # leaky_accuracy[i] = net.evaluate(test_data)/len(test_data)*100

        print(f'{i+1}/{len(num_hidden_layers)} hidden layers complete')

    np.save('accuracy_hidden_layers_sigmoid.npy', sigmoid_accuracy)
    # np.save('accuracy_hidden_layers_relu.npy',    relu_accuracy)
    # np.save('accuracy_hidden_layers_leaky.npy',   leaky_accuracy)


def plot_accuracy_vs_hidden_layers():
    sns.set()
    num_hidden_layers = np.arange(5)
    sigmoid_accuracy  = np.load('accuracy_hidden_layers_sigmoid.npy')

    plt.plot(num_hidden_layers, sigmoid_accuracy, 'o-', label='sigmoid')
    plt.xlabel(r'Number of hidden layers')
    plt.ylabel(r'Accuracy %')
    plt.legend()
    plt.tight_layout()
    plt.show()

def accuracy_vs_nodes():
    nodes            = 10*np.arange(1, 9)
    sigmoid_accuracy = np.zeros(len(nodes))
    relu_accuracy    = np.zeros(len(nodes))
    leaky_accuracy   = np.zeros(len(nodes))
    for i, n in enumerate(nodes):
        layers = [784, n, 10]

        net = NeuralNetwork(layers=layers, functions=sigmoid_functions, functions_derivatives=sigmoid_derivatives, cost_derivative=cross_entropy_derivative, mode='classification')
        net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, lmbda=0)
        sigmoid_accuracy[i] = net.evaluate(test_data)/len(test_data)*100

        net = NeuralNetwork(layers=layers, functions=relu_functions, functions_derivatives=relu_derivatives, cost_derivative=cross_entropy_derivative, mode='classification')
        net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, lmbda=0)
        relu_accuracy[i] = net.evaluate(test_data)/len(test_data)*100

        net = NeuralNetwork(layers=layers, functions=leaky_functions, functions_derivatives=leaky_derivatives, cost_derivative=cross_entropy_derivative, mode='classification')
        net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, lmbda=0)
        leaky_accuracy[i] = net.evaluate(test_data)/len(test_data)*100

    np.save('accuracy_nodes_sigmoid.npy', sigmoid_accuracy)
    np.save('accuracy_nodes_relu.npy',    relu_accuracy)
    np.save('accuracy_nodes_leaky.npy',   leaky_accuracy)


def plot_accuracy_vs_nodes():
    sns.set()
    nodes = 10*np.arange(1, 9)

    sigmoid_accuracy  = np.load('accuracy_nodes_sigmoid.npy')
    relu_accuracy     = np.load('accuracy_nodes_relu.npy')
    leaky_accuracy    = np.load('accuracy_nodes_leaky.npy')

    plt.plot(nodes, sigmoid_accuracy, '-o', label='sigmoid')
    plt.plot(nodes, relu_accuracy,    '-o', label='relu')
    plt.plot(nodes, leaky_accuracy,   '-o', label='leaky relu')
    plt.xlabel('Number of nodes')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.tight_layout()
    plt.show()

def accuracy_vs_eta_lambda():
    layers = [784, 30, 10]
    # etas =
    # lmbdas =
    accuracy = np.zeros((len(eta), len(lmbda)))
    for i, eta in enumerate(etas):
        for j, lmbda in enumerate(lmbdas):
            net = NeuralNetwork(layers=layers, functions=sigmoid_functions, functions_derivatives=sigmoid_derivatives, cost_derivative=cross_entropy_derivative, mode='classification')
            net.SGD(training_data, epochs=30, mini_batch_size=10, eta=eta, lmbda=lmbda)
            accuracy[i, j] = net.evaluate(test_data)/len(test_data)*100

    np.save('accuracy_eta_lambda.npy', accuracy)


def plot_accuracy_vs_eta_lambda():
    pass
