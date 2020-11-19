import numpy as np
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
sns.set(rc={'figure.figsize':(10,7)})


np.random.seed(40)

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

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, train_size=0.8)

input_train, output_train, training_data = prepare_data(x_train, y_train, z_train)
input_test, output_test, testing_data    = prepare_data(x_test, y_test, z_test)

#layers = [2, 30, 1] # Input layer must be 2 nodes, output layer must be 1 node, since Franke is F : R^2 -> R^1
                    # Anything inbetween is arbitrary

#weights = [np.random.normal(0, 1/m, (m,n)) for m, n in zip(layers[1:], layers[:-1])]
# net = NeuralNetwork(layers=layers, weights=None, mode='regression')
# net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3, lmbda=0)
#
# ztilde  = net.feedforward(input_test)
# ztilde  = ztilde.flatten()
# z_exact = z_test.flatten()
#
# print(f"MSE = {MSE(z_exact, ztilde)}")
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

@np.vectorize
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

def linear(x):
    return x

def linear_derivative(x):
    return 1

def MSE_(a, y):
    return np.sum((a - y)**2)/len(a)

def MSE_derivative(a, y):
    return (a - y)



def MSE_vs_hidden_layers():
    h_layers = np.arange(5)
    accuracy = np.zeros(len(h_layers))
    for i in h_layers:
        layers = [2]
        layers += i*[30]
        layers += [1]
        NN = NeuralNetwork(layers=layers, mode='regression')
        NN.SGD(training_data, epochs=30, mini_batch_size=10, eta=0.3, lmbda=1e-6)
        z_tilde = NN.feedforward(input_test)
        z_tilde = z_tilde.flatten()
        z_exact = z_test.flatten()
        accuracy[i] = MSE(z_exact, z_tilde)

        print(f'{i+1}/{len(h_layers)} hidden layers complete')

    np.save('MSE_hidden_layers.npy', accuracy)

def plot_MSE_vs_hidden_layers():
    h_layers = np.arange(5)
    accuracy = np.load('MSE_hidden_layers.npy')
    plt.figure(figsize=(12, 7))
    plt.plot(h_layers, accuracy, 'o-')
    plt.xlabel(r'Number of hidden layers')
    plt.ylabel(r'MSE')
    plt.title('MSE as a function of hidden layers')
    plt.show()

#MSE_vs_hidden_layers()
#plot_MSE_vs_hidden_layers()


def MSE_vs_nodes():
    nodes       = 10*np.arange(1, 9)
    sigmoid_MSE = np.zeros(len(nodes))
    relu_MSE    = np.zeros(len(nodes))
    leaky_MSE   = np.zeros(len(nodes))
    for i, n in enumerate(nodes):
        layers = [2, n, 1]

        NN = NeuralNetwork(layers=layers, mode='regression')
        NN.SGD(training_data, epochs=30, mini_batch_size=10, eta=0.3, lmbda=1e-5)
        z_tilde = NN.feedforward(input_test)
        z_tilde = z_tilde.flatten()
        z_exact = z_test.flatten()
        sigmoid_MSE[i] = MSE(z_exact, np.nan_to_num(z_tilde))


        relu_functions     = [relu]*(len(layers) - 2)
        relu_functions    += [linear]
        relu_derivatives   = [relu_derivative]*(len(layers) - 2)
        relu_derivatives  += [linear_derivative]

        NN = NeuralNetwork(layers=layers, functions=relu_functions, functions_derivatives=relu_derivatives, cost_derivative=MSE_derivative, mode='regression')
        NN.SGD(training_data, epochs=30, mini_batch_size=10, eta=0.3, lmbda=1e-5)
        z_tilde = NN.feedforward(input_test)
        z_tilde = z_tilde.flatten()
        z_exact = z_test.flatten()
        relu_MSE[i] = MSE(z_exact, np.nan_to_num(z_tilde))

        leaky_functions    = [leaky]*(len(layers) - 2)
        leaky_functions   += [linear]
        leaky_derivatives  = [leaky_derivative]*(len(layers) - 2)
        leaky_derivatives += [linear_derivative]

        NN = NeuralNetwork(layers=layers, functions=leaky_functions, functions_derivatives=leaky_derivatives, cost_derivative=MSE_derivative, mode='regression')
        NN.SGD(training_data, epochs=30, mini_batch_size=10, eta=0.3, lmbda=1e-5)
        z_tilde = NN.feedforward(input_test)
        z_tilde = z_tilde.flatten()
        z_exact = z_test.flatten()
        leaky_MSE[i] = MSE(z_exact, np.nan_to_num(z_tilde))

    np.save('MSE_nodes_sigmoid_reg.npy', sigmoid_MSE)
    np.save('MSE_nodes_relu.npy',    relu_MSE)
    np.save('MSE_nodes_leaky.npy',   leaky_MSE)


def plot_MSE_vs_nodes():
    sns.set()
    nodes = 10*np.arange(1, 9)

    sigmoid_accuracy  = np.load('MSE_nodes_sigmoid_reg.npy')
    relu_accuracy     = np.load('MSE_nodes_relu.npy')
    leaky_accuracy    = np.load('MSE_nodes_leaky.npy')

    plt.plot(nodes, sigmoid_accuracy,'-o', label='sigmoid')
    plt.plot(nodes, relu_accuracy,  ls='-.' ,label='relu')
    plt.plot(nodes, leaky_accuracy,  ls=':', label='leaky relu')
    plt.xlabel('Number of nodes')
    plt.ylabel('MSE')
    plt.title('MSE as function of number of nodes')
    plt.legend()
    plt.tight_layout()
    plt.show()

#MSE_vs_nodes()
#plot_MSE_vs_nodes()


def MSE_vs_eta_lambda():
    layers = [2, 30, 1]
    etas = np.array([0.05, 0.1, 0.15, 0.2, 0.3])
    lmbdas = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    accuracy = np.zeros((len(etas), len(lmbdas)))
    for i, eta in enumerate(etas):
        for j, lmbda in enumerate(lmbdas):
            NN = NeuralNetwork(layers=layers, mode='regression')
            NN.SGD(training_data, epochs=30, mini_batch_size=10, eta=eta, lmbda=lmbda)
            z_tilde = NN.feedforward(input_test)
            z_tilde = z_tilde.flatten()
            z_exact = z_test.flatten()
            accuracy[i,j] = MSE(z_exact, z_tilde)
    np.save('accuracy_eta_lambda.npy', accuracy)

def plot_MSE_vs_eta_lambda():
    etas = np.array([0.05, 0.1, 0.15, 0.2, 0.3])
    lmbdas = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    sns.set(rc={'figure.figsize':(12,9)})
    accuracy = np.load('accuracy_eta_lambda.npy')
    acc_df = pd.DataFrame(accuracy, index=etas, columns=lmbdas)
    fig, ax = plt.subplots()
    ax = sns.heatmap(acc_df, annot = True, cbar_kws={'label': 'MSE'})
    ax.set_title(r'MSE for changing $\lambda$ and $\eta$')
    ax.set_ylabel(r'$\eta$')
    ax.set_xlabel(r'$\lambda$')
    plt.show()

#MSE_vs_eta_lambda()
plot_MSE_vs_eta_lambda()