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

def prepare_data(x, y, z):
    """Prepare data on a form that the neural network can handle.
       Only works on scalar functions of two variables, e.g. the Frankefunction."""
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

min_z, max_z = np.min(z), np.max(z)
if min_z < 0:
    z += min_z
if max_z > 1:
    z /= max_z

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, train_size=0.8)

input_train, output_train, training_data = prepare_data(x_train, y_train, z_train)
input_test, output_test, testing_data    = prepare_data(x_test, y_test, z_test)

layers = [2, 30, 1] # Input layer must be 2 nodes, output layer must be 1 node, since Franke is F : R^2 -> R^1
                    # Anything inbetween is arbitrary

weights = [np.random.normal(0, 1/m, (m,n)) for m, n in zip(layers[1:], layers[:-1])]
net = NeuralNetwork(layers=layers, weights=None, mode='regression')
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, lmbda=0)

ztilde  = net.feedforward(input_test)
ztilde  = ztilde.flatten()
z_exact = z_test.flatten()

print(f"MSE = {MSE(z_exact, ztilde)}")
