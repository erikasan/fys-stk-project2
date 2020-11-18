import numpy as np

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = training_data[:500]
test_data = test_data[:50]


from neural_network import NeuralNetwork

layers = [784, 30, 10]



net = NeuralNetwork(layers=layers, mode='classification')

net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, lmbda=0)

print(f'{net.evaluate(test_data)/len(test_data)*100:.1f}% success rate')
