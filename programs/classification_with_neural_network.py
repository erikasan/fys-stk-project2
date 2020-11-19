import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = training_data[:100]
test_data = test_data[:10]


from neural_network import NeuralNetwork

# layers = [784, 30, 10]
#
#
#
# net = NeuralNetwork(layers=layers, mode='classification')
#
# net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, lmbda=0)
#
# print(f'{net.evaluate(test_data)/len(test_data)*100:.1f}% success rate')

@np.vectorize
def sigmoid(x):
    if x < 0:
        return np.exp(x)/(1 + np.exp(x))
    else:
        return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

def relu(x):
    pass

def relu_derivative(x):
    pass

def leaky(x):
    pass

def leaky_derivative(x):
    pass


sigmoid_functions = []
sigmoid_derivatives = []

relu_functions = []
relu_derivatives = []

leaky_functions = []
leaky_derivatives = []

def accuracy_vs_hidden_layers():
    num_hidden_layers = np.arange(5)
    accuracy = np.zeros(len(num_hidden_layers))
    for i in num_hidden_layers:
        layers  = [784]
        layers += i*[30]
        layers += [10]
        net = NeuralNetwork(layers=layers, mode='classification')
        net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, lmbda=0)
        accuracy[i] = net.evaluate(test_data)/len(test_data)*100

        print(f'{i+1}/{len(num_hidden_layers)} hidden layers complete')

    np.save('accuracy_hidden_layers.npy', accuracy)

def plot_accuracy_vs_hidden_layers():
    sns.set()
    num_hidden_layers = np.arange(5)
    accuracy = np.load('accuracy_hidden_layers.npy')
    plt.plot(num_hidden_layers, accuracy, 'o-')
    plt.xlabel(r'Number of hidden layers')
    plt.ylabel(r'Accuracy %')
    plt.show()

def accuracy_vs_nodes():
    pass

def plot_accuracy_vs_nodes():
    pass
