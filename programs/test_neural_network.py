# from neural_network import NeuralNetwork
# import numpy as np
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
#
# digits = datasets.load_digits()
#
# inputs = digits.images[:10]
# labels = digits.target[:10]
# labels.shape += (1,)
#
# n_inputs = len(inputs)
# inputs = inputs.reshape(n_inputs, -1)
# inputs.shape += (1,)
# inputs /= 255
#
# X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=0.8)
#
# def to_categorical_numpy(integer_vector):
#     n_inputs = len(integer_vector)
#     n_categories = 10
#     onehot_vector = np.zeros((n_inputs, n_categories))
#     onehot_vector[range(n_inputs), integer_vector] = 1
#     onehot_vector.shape += (1,)
#     return onehot_vector
#
# Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(Y_test)
#
# training_data = [(x, y) for x, y in zip(X_train, Y_train_onehot)]
# test_data     = [(x, y) for x, y in zip(X_test, Y_test)]
#
# net = NeuralNetwork([64, 30, 10], mode='classification')
#
# net.SGD(training_data, epochs=30, mini_batch_size=2, eta=3)
#
# print(f'{net.evaluate(test_data)/len(test_data)*100:.1f}% success rate')

import sys
sys.path.append("../")

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

from neural_network import NeuralNetwork

net = NeuralNetwork([784, 30, 10], mode='classification')
net.SGD(training_data, 30, 10, 3.0)

print(f'{net.evaluate(test_data)/len(test_data)*100:.1f}% success rate')
