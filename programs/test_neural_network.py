from neural_network import NeuralNetwork
import numpy as np

x = np.array([1, 2, 3])
x.shape = (x.shape[0], 1)
y = 2*x
net = NeuralNetwork([3, 3, 4])

training_data = [(x, y)]
net.SGD(training_data, epochs=1, mini_batch_size=1, eta=1)
