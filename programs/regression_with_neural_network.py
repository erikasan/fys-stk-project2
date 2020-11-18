import numpy as np
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

def FrankeFunction(x, y):
    term1 =  0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 =  0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 =   0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = - 0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def prepare_training_data(x, y, z):
    assert len(x) == len(y) == len(z), "shape of x, y, z do not match"
    input_data = [np.array([i, j]) for i, j in zip(x, y)]
    for arr in input_data:
        arr.shape = (arr.shape[0], 1)

    output_data = [np.array([k]) for k in z]
    for arr in output_data:
        arr.shape = (arr.shape[0], 1)

    training_data = [(i, o) for i, o in zip(input_data, output_data)]
    return input_data, output_data, training_data


x = np.random.random(1000)
y = np.random.random(1000)
z = FrankeFunction(x, y)

input_data, output_data, training_data = prepare_training_data(x, y, z)

layers = [2, 20, 1]
net = NeuralNetwork(layers=layers, mode='regression')
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, lmbda=0)


ztilde  = net.feedforward(input_data)
ztilde  = ztilde.flatten()
z_exact = z.flatten()

print(f"MSE = {MSE(z_exact, ztilde)}")
