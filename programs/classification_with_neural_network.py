import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = training_data[:500]
test_data = test_data[:50]


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
    plt.plot(num_hidden_layers, accuracy)
    plt.show()
